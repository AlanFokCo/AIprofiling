// src/plugins/unix_handler.rs
use crate::collector::collector_name::CollectorName;
use crate::collector::event_handler::{EventHandler, SchedulerEvent};
use crate::r#const;
use anyhow::{anyhow, Result};
use std::fs;
use std::path::Path;
use std::time::Duration;
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::broadcast;
use tracing as log;
pub struct UnixSocketHandler;

/// Upper bound for the liveness probe in [`UnixSocketHandler::bind_listener_at`].
///
/// `connect(2)` on a SOCK_STREAM unix socket completes as soon as the listener's
/// backlog has room; it does not wait for the peer to call `accept`. A live
/// listener therefore answers in microseconds even when its accept loop is
/// starved - the exact shape `scheduler::start_event_loop` documents for a
/// container under a 1-CPU quota. The bound covers the pathological cases
/// instead (backlog already full, or a peer wedged mid-handshake): without it a
/// `std::os::unix::net::UnixStream::connect` with no timeout would park a tokio
/// worker thread indefinitely and hang startup for every pid behind it.
const SOCKET_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// What the liveness probe proved about a socket path that is already occupied.
enum SocketProbe {
    /// Something answered, so the inode is in use and must not be unlinked.
    Live,
    /// `connect(2)` said ECONNREFUSED: a socket file with no listener behind it,
    /// i.e. the residue of a run that died without unlinking. Safe to reclaim.
    /// Linux refuses a connect to a path that is not a socket at all the same
    /// way, so a stray regular file squatting on the reserved path lands here
    /// too - nothing can be serving lifecycle events through it.
    Dead(String),
    /// The probe timed out or failed in a way that says nothing about liveness
    /// (permission denied, path vanished, ...). Treated as live: reclaiming on an
    /// answer we cannot interpret is how the socket theft this module exists to
    /// prevent would creep back in.
    Inconclusive(String),
}

/// Map a lifecycle message received on a target's unix socket to the scheduler
/// event it stands for.
///
/// These strings are a wire protocol with the injected collectors, matched by
/// exact equality:
///
///   * cuprof (vendored at `src/plugins/cuprof`) emits `CUPTIProfilingStart` /
///     `Stop` / `WriterOver` / `Failed` from `src/cupti_sink.cc`. One connection
///     per message, no framing - see `src/plugins/cuprof/docs/embedding.md`.
///   * the pyki loader emits `STARTPYKICOLLECTOR` / `STOPPYKICOLLECTOR` from the
///     Python snippet built in `main.rs`.
///
/// The `drift_guard_*` tests at the bottom of this file fail if a re-sync of the
/// vendored cuprof tree renames any of them, which is the check
/// `src/plugins/cuprof/VENDOR.md` otherwise asks a human to perform by hand.
pub fn scheduler_event_for(received: &str, pid: i32) -> Option<SchedulerEvent> {
    match received {
        r#const::START_PYKI_COLLECTOR => Some(SchedulerEvent::StartPendingCollector(
            CollectorName::Pyki,
            pid,
        )),
        r#const::STOP_PYKI_COLLECTOR => {
            Some(SchedulerEvent::StopCollector(CollectorName::Pyki, pid))
        }
        r#const::START_CUPTI_COLLECTOR => Some(SchedulerEvent::StartPendingCollector(
            CollectorName::CUPTI,
            pid,
        )),
        r#const::STOP_CUPTI_COLLECTOR => {
            Some(SchedulerEvent::StopCollector(CollectorName::CUPTI, pid))
        }
        r#const::FINISH_CUPTI_COLLECTOR => {
            Some(SchedulerEvent::WritingFinish(CollectorName::CUPTI, pid))
        }
        // cuprof sends this *instead of* WriterOver when it could not write the
        // trace file. embedding.md asks consumers to treat it as terminal for
        // the window and not to wait for a file that will never appear.
        // CollectFailed drives writer::handle_collect_failed, which sets
        // CollectorState::Failed (8); all_collectors_finished() only requires
        // >= WrittingOver (7), so the run finalizes immediately instead of
        // idling until the duration + 60s watchdog fires.
        r#const::FAILED_CUPTI_COLLECTOR => {
            Some(SchedulerEvent::CollectFailed(CollectorName::CUPTI, pid))
        }
        _ => None,
    }
}

impl UnixSocketHandler {
    /// Host-visible path of a target's unix socket: `<root><CF_UNIXSOCK><pid>`.
    ///
    /// Production passes [`Self::proc_root`] as `root`, which is the target's `/`
    /// as seen from the host, so the injected collector (which connects to
    /// `<CF_UNIXSOCK><pid>` from inside) and this listener meet on one inode.
    /// `root` is a parameter only so tests can aim the same logic at a temp dir
    /// instead of procfs. Always pair it with `proc_root` in production code:
    /// hand-rolling the `/proc/<pid>/root` prefix is how this path drifts.
    pub fn socket_path_under(root: &str, pid: i32) -> String {
        format!("{}{}{}", root, r#const::CF_UNIXSOCK, pid)
    }

    /// The procfs root prefix of a target: the container's `/` as seen from the
    /// host. Pinned by `socket_path_matches_the_deployed_layout`, because the
    /// injected collectors build the same string independently.
    ///
    /// Public so every caller that needs the production root
    /// (`collector_manager::init_event_thread`, the tests below) goes through
    /// this one definition instead of re-deriving it, which would leave
    /// `socket_path_under` looking like the single constructor while the prefix
    /// it is fed silently diverges.
    pub fn proc_root(pid: i32) -> String {
        format!("/proc/{}/root", pid)
    }

    /// Bind this pid's listener, or explain why we may not.
    ///
    /// A file already sitting at the socket path is either a *live* socket served
    /// by another CollectionFramework instance profiling the same pid, or the
    /// corpse of an earlier run that died without unlinking it. Unlinking
    /// unconditionally - what this used to do - silently kills the other
    /// instance's event stream, so probe first and only reclaim an inode the
    /// probe proves dead.
    ///
    /// Async because the probe is a `tokio` connect under
    /// [`SOCKET_PROBE_TIMEOUT`]; this runs on a tokio worker (it is called from
    /// `collector_manager::init_collectors`), where an unbounded `std` connect
    /// would wedge startup.
    pub async fn bind_listener_at(root: &str, pid: i32) -> Result<UnixListener> {
        let socket_path = Self::socket_path_under(root, pid);

        match UnixListener::bind(&socket_path) {
            Ok(listener) => Ok(listener),
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
                Self::resolve_occupied(&socket_path, pid).await
            }
            Err(e) => Err(anyhow!(
                "failed to bind unix socket {} for pid {}: {}",
                socket_path,
                pid,
                e
            )),
        }
    }

    /// Decide what to do about a path [`Self::bind_listener_at`] found occupied.
    /// Split out of the bind so this decision can be driven from a test without
    /// having to win a race against a real listener.
    async fn resolve_occupied(socket_path: &str, pid: i32) -> Result<UnixListener> {
        match Self::probe_socket(socket_path).await {
            SocketProbe::Live => Err(anyhow!(
                "unix socket {} for pid {} is already in use and a live listener \
                 answered our probe, so lifecycle events for pid {} cannot be collected \
                 here. The probe only proves that *something* is listening, not that it \
                 is AIProf: check for a second aiprof-client or CollectionFramework \
                 profiling this pid, and for a socket file left behind by an earlier run \
                 that something is still serving. The socket was left untouched",
                socket_path,
                pid,
                pid
            )),
            SocketProbe::Inconclusive(reason) => {
                // RACE: the owner can exit - and unlink its own socket -
                // between the bind that reported AddrInUse and the probe
                // above, so a path that is in fact free lands here. Retry
                // once rather than failing a run a plain retry would have
                // saved; once, not in a loop, because a second failure
                // means an owner really is there now and guessing again is
                // how socket theft creeps back in.
                if !Path::new(socket_path).exists() {
                    if let Ok(listener) = UnixListener::bind(socket_path) {
                        log::warn!(
                            "unix socket {} for pid {} vanished before the liveness probe \
                             could answer ({}); rebound it on the now-free path",
                            socket_path,
                            pid,
                            reason
                        );
                        return Ok(listener);
                    }
                }
                Err(anyhow!(
                    "unix socket {} for pid {} is already in use and its liveness could not \
                     be determined ({}), so it was left untouched rather than unlinked",
                    socket_path,
                    pid,
                    reason
                ))
            }
            SocketProbe::Dead(reason) => {
                log::warn!(
                    "Reclaiming stale unix socket {} for pid {} ({}). cuprof opens one \
                     connection per lifecycle message and keeps none open, so any message \
                     it sent between the old listener dying and this rebind is lost: a \
                     window that was already running will end in the watchdog, not in \
                     WriterOver",
                    socket_path,
                    pid,
                    reason
                );
                // RACE WINDOW: another instance can bind this path
                // between the probe above and the rebind below, so the
                // unlink can in principle delete a socket that just
                // became live. It is kept to one unlink + one rebind
                // (no retry loop) and the rebind surfaces AddrInUse, so
                // the worst case is a loud failure instead of a
                // silently stolen socket.
                fs::remove_file(socket_path).map_err(|e| {
                    anyhow!("failed to remove stale unix socket {}: {}", socket_path, e)
                })?;
                UnixListener::bind(socket_path).map_err(|e| {
                    anyhow!(
                        "failed to bind unix socket {} for pid {}: {}",
                        socket_path,
                        pid,
                        e
                    )
                })
            }
        }
    }

    /// Ask an occupied socket path whether anybody is still listening, with a
    /// hard time limit. See [`SocketProbe`] for how each answer is treated.
    async fn probe_socket(socket_path: &str) -> SocketProbe {
        match tokio::time::timeout(SOCKET_PROBE_TIMEOUT, UnixStream::connect(socket_path)).await {
            Ok(Ok(_stream)) => SocketProbe::Live,
            Ok(Err(e)) if e.kind() == std::io::ErrorKind::ConnectionRefused => {
                SocketProbe::Dead(format!("connect was refused: {}", e))
            }
            Ok(Err(e)) => SocketProbe::Inconclusive(format!("probe failed: {}", e)),
            Err(_elapsed) => SocketProbe::Inconclusive(format!(
                "probe did not answer within {:?}",
                SOCKET_PROBE_TIMEOUT
            )),
        }
    }

    /// Serve an already-bound listener: spawn the accept loop for `pid`.
    pub fn serve(
        listener: UnixListener,
        pid: i32,
        shutdown_sender: broadcast::Sender<()>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            log::debug!("Starting Listen Unixsock for pid: {}", pid);

            // Create a receiver to listen for the shutdown signal
            let mut shutdown_receiver = shutdown_sender.subscribe();

            // Loop forever, listening for new connections
            loop {
                tokio::select! {
                    // Accept a new connection and handle messages
                    result = listener.accept() => {
                        match result {
                            Ok((socket, _addr)) => {
                                log::debug!("Client connected to Unix socket for pid: {}", pid);

                                // Subscribe to the shutdown signal for each connection
                                let mut conn_shutdown_receiver = shutdown_sender.subscribe();

                                // Spawn an async task per connection to handle multiple messages
                                tokio::spawn(async move {
                                    loop {
                                        tokio::select! {
                                            result = socket.readable() => {
                                                match result {
                                                    Ok(()) => {
                                                        let mut buffer = vec![0; 1024];
                                                        match socket.try_read(&mut buffer) {
                                                            Ok(0) => {
                                                                // Connection closed
                                                                log::debug!("Client disconnected from Unix socket for pid: {}", pid);
                                                                break;
                                                            }
                                                            Ok(n) => {
                                                                // Convert the read data to a string and strip trailing NUL bytes
                                                                let received_data = String::from_utf8_lossy(&buffer[..n]);
                                                                let received = received_data.trim_end_matches('\0');
                                                                log::debug!("Received UnixSocket message: {} for pid: {}", received, pid);

                                                                // Dispatch the matching event based on the received message
                                                                if received == r#const::FAILED_CUPTI_COLLECTOR {
                                                                    // Raw messages are only logged at debug, so surface
                                                                    // the one that means this window did not complete.
                                                                    // cuprof sends it when it could not write the trace,
                                                                    // when its CUPTI teardown timed out, and when it
                                                                    // refuses to start a window in a process that an
                                                                    // earlier timeout already retired (patches/0005). In
                                                                    // the second case a truncated trace does exist and is
                                                                    // still worth copying out, so do not claim there is
                                                                    // no output.
                                                                    log::warn!(
                                                                        "cuprof reported pid {}'s collection window as failed; any trace it did write is incomplete",
                                                                        pid
                                                                    );
                                                                }
                                                                let event = scheduler_event_for(received, pid);

                                                                // If there is a matching event, send it
                                                                if let Some(evt) = event {
                                                                    if let Err(e) = EventHandler::global_sender().send(evt) {
                                                                        log::error!("Failed to send scheduler event: {}", e);
                                                                    }
                                                                }
                                                            }
                                                            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                                                                continue;
                                                            }
                                                            Err(e) => {
                                                                log::error!("Failed to read from Unix socket for pid {}: {}", pid, e);
                                                                break;
                                                            }
                                                        }
                                                    }
                                                    Err(e) => {
                                                        log::error!("Failed to make Unix socket readable for pid {}: {}", pid, e);
                                                        break;
                                                    }
                                                }
                                            }
                                            // Listen for a per-connection shutdown signal
                                            _ = conn_shutdown_receiver.recv() => {
                                                log::info!("Received shutdown signal, stopping connection handler for pid: {}", pid);
                                                break;
                                            }
                                        }
                                    }
                                });
                            }
                            Err(e) => log::error!("Failed to accept Unix socket connection: {}", e),
                        }
                    }
                    // Listen for the shutdown signal
                    _ = shutdown_receiver.recv() => {
                        log::error!("Received shutdown signal, stopping Unix socket listener for pid: {}", pid);
                        break;
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuprof_lifecycle_messages_map_to_scheduler_events() {
        assert!(matches!(
            scheduler_event_for(r#const::START_CUPTI_COLLECTOR, 42),
            Some(SchedulerEvent::StartPendingCollector(
                CollectorName::CUPTI,
                42
            ))
        ));
        assert!(matches!(
            scheduler_event_for(r#const::STOP_CUPTI_COLLECTOR, 42),
            Some(SchedulerEvent::StopCollector(CollectorName::CUPTI, 42))
        ));
        assert!(matches!(
            scheduler_event_for(r#const::FINISH_CUPTI_COLLECTOR, 42),
            Some(SchedulerEvent::WritingFinish(CollectorName::CUPTI, 42))
        ));
    }

    #[test]
    fn cuprof_failed_write_is_terminal_and_not_silently_dropped() {
        // Before this mapping existed the message fell through to `None`, so no
        // event was emitted and the scheduler waited out its whole
        // duration + 60s watchdog for a trace that cuprof had already said it
        // could not produce.
        assert!(matches!(
            scheduler_event_for(r#const::FAILED_CUPTI_COLLECTOR, 42),
            Some(SchedulerEvent::CollectFailed(CollectorName::CUPTI, 42))
        ));
    }

    #[test]
    fn pyki_lifecycle_messages_map_to_scheduler_events() {
        assert!(matches!(
            scheduler_event_for(r#const::START_PYKI_COLLECTOR, 42),
            Some(SchedulerEvent::StartPendingCollector(
                CollectorName::Pyki,
                42
            ))
        ));
        assert!(matches!(
            scheduler_event_for(r#const::STOP_PYKI_COLLECTOR, 42),
            Some(SchedulerEvent::StopCollector(CollectorName::Pyki, 42))
        ));
    }

    #[test]
    fn the_socket_pid_is_carried_into_the_event() {
        // The socket is per-target; attributing an event to the wrong pid would
        // credit one process's kernels to another.
        assert!(matches!(
            scheduler_event_for(r#const::FINISH_CUPTI_COLLECTOR, 4242),
            Some(SchedulerEvent::WritingFinish(CollectorName::CUPTI, 4242))
        ));
    }

    #[test]
    fn unknown_or_empty_message_is_ignored() {
        assert!(scheduler_event_for("", 7).is_none());
        assert!(scheduler_event_for("CUPTIProfilingWhatever", 7).is_none());
        assert!(scheduler_event_for("cuptiprofilingstart", 7).is_none());
    }

    // ---- drift guards against the vendored cuprof tree ----
    // VENDOR.md asks whoever re-syncs cuprof to re-check the message names and
    // the config keys by hand. These two tests make `cargo test` fail instead.

    #[test]
    fn drift_guard_vendored_cuprof_emits_every_message_we_match_on() {
        let sink = include_str!("plugins/cuprof/src/cupti_sink.cc");
        for msg in [
            r#const::START_CUPTI_COLLECTOR,
            r#const::STOP_CUPTI_COLLECTOR,
            r#const::FINISH_CUPTI_COLLECTOR,
            r#const::FAILED_CUPTI_COLLECTOR,
        ] {
            assert!(
                sink.contains(msg),
                "vendored cuprof no longer emits {:?}: update src/const.rs and \
                 scheduler_event_for (see src/plugins/cuprof/VENDOR.md)",
                msg
            );
        }
    }

    #[test]
    fn drift_guard_vendored_cuprof_reads_every_config_key_we_write() {
        let cfg = include_str!("plugins/cuprof/src/config.cc");
        // The keys cupti_plugin_wrapper::write_cuprof_config emits into
        // /tmp/cuprof_<container-pid>.cfg, plus CUPROF_CONFIG, which
        // cuprof/VENDOR.md documents as part of the same contract.
        for key in [
            "CUPROF_OUTPUT",
            "CUPROF_DURATION",
            "CUPROF_VERBOSE",
            "CUPROF_SOCKET",
            "CUPROF_CONFIG",
        ] {
            assert!(
                cfg.contains(key),
                "vendored cuprof no longer reads {}: update \
                 cupti_plugin_wrapper::write_cuprof_config",
                key
            );
        }
    }

    #[test]
    fn drift_guard_vendored_cuprof_keeps_the_cfg_filename_convention() {
        // write_cuprof_config writes /proc/<pid>/root/tmp/cuprof_<ns-pid>.cfg
        // and depends on cuprof's LoadCfgFile looking for exactly that name.
        // A rename here is the quietest failure of the three: the injected
        // library finds no config and silently falls back to its defaults
        // (./cuprof_<pid>.json, no socket), so the run produces no trace and
        // no lifecycle message at all.
        let cfg = include_str!("plugins/cuprof/src/config.cc");
        assert!(
            cfg.contains("/tmp/cuprof_"),
            "vendored cuprof no longer reads /tmp/cuprof_<pid>.cfg: update \
             cupti_plugin_wrapper::write_cuprof_config"
        );
    }

    // ---- bind_listener_at ----
    // These aim bind_listener_at at a temp root because /proc/<pid>/root is
    // neither present nor writable for a dummy pid on a dev host; the
    // production path itself is pinned by socket_path_matches_the_deployed_layout.

    /// Stand-in for `/proc/<pid>/root`. `CF_UNIXSOCK` is an absolute path under
    /// `/tmp`, which a real container root always has but a bare temp dir does
    /// not, so create it here.
    fn fake_container_root() -> tempfile::TempDir {
        let root = tempfile::tempdir().expect("temp root");
        fs::create_dir(root.path().join("tmp")).expect("create <root>/tmp");
        root
    }

    #[test]
    fn socket_path_matches_the_deployed_layout() {
        // The injected collectors build this string independently (see
        // cupti_plugin_wrapper::write_cuprof_config), so it must not drift.
        assert_eq!(UnixSocketHandler::proc_root(4242), "/proc/4242/root");
        assert_eq!(
            UnixSocketHandler::socket_path_under("/proc/4242/root", 4242),
            "/proc/4242/root/tmp/.cf_sock_4242"
        );
    }

    #[test]
    fn the_liveness_probe_is_bounded() {
        // A `std` connect with no timeout runs on a tokio worker here, so an
        // unbounded probe would wedge startup instead of failing it. Pin the
        // bound: long enough for a starved accept loop to answer, short enough
        // that a wedged peer cannot stall the run.
        assert!(SOCKET_PROBE_TIMEOUT > Duration::ZERO);
        assert!(SOCKET_PROBE_TIMEOUT <= Duration::from_secs(10));
    }

    #[tokio::test]
    async fn bind_listener_binds_a_fresh_path() {
        let root = fake_container_root();
        let listener = UnixSocketHandler::bind_listener_at(root.path().to_str().unwrap(), 7)
            .await
            .expect("a fresh socket path binds");
        drop(listener);
    }

    #[tokio::test]
    async fn bind_listener_refuses_to_steal_a_live_socket() {
        let root = fake_container_root();
        let root_str = root.path().to_str().unwrap();
        // Kept alive on purpose: it stands for a second CollectionFramework
        // instance already profiling this pid.
        let first = UnixSocketHandler::bind_listener_at(root_str, 7)
            .await
            .expect("first bind");

        let err = UnixSocketHandler::bind_listener_at(root_str, 7)
            .await
            .expect_err("a live socket must not be stolen");
        // Assert on the pid and on what the probe actually proved, not on the
        // host-absolute temp path: the message must not claim to know that the
        // listener is another AIProf instance, only that one is there.
        let msg = err.to_string();
        assert!(
            msg.contains("pid 7")
                && msg.contains("live listener")
                && msg.contains("already in use"),
            "error should name the pid and say a live listener answered, got: {}",
            msg
        );
        assert!(
            msg.contains("second aiprof-client") && msg.contains("left untouched"),
            "error should hint at the two real causes and say nothing was deleted, got: {}",
            msg
        );

        // The live listener still works, which proves the failed bind did not
        // unlink the socket out from under it.
        let path = UnixSocketHandler::socket_path_under(root_str, 7);
        std::os::unix::net::UnixStream::connect(&path).expect("live socket still connectable");
        let _accepted = first.accept().await.expect("live socket still accepting");
    }

    #[tokio::test]
    async fn a_starved_listener_still_answers_the_probe() {
        // The listener below never calls accept, which is the starvation shape
        // scheduler.rs documents under a 1-CPU quota. connect(2) only needs
        // backlog room, so the probe must still report it live well inside the
        // timeout instead of blocking a tokio worker until the watchdog fires.
        let root = fake_container_root();
        let root_str = root.path().to_str().unwrap();
        let _starved = UnixSocketHandler::bind_listener_at(root_str, 7)
            .await
            .expect("first bind");

        let started = std::time::Instant::now();
        let err = UnixSocketHandler::bind_listener_at(root_str, 7)
            .await
            .expect_err("a starved listener is still live");
        let elapsed = started.elapsed();
        assert!(
            elapsed < SOCKET_PROBE_TIMEOUT,
            "probe took {:?} against a live listener, so it is not bounded by the backlog",
            elapsed
        );
        assert!(
            err.to_string().contains("live listener"),
            "a starved listener must be reported as live, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn the_probe_only_calls_a_socket_dead_on_refusal() {
        // Dead is the only answer that licenses the unlink, so it must stay
        // reserved for "connect was refused". Note Linux refuses a connect to a
        // path that is not a socket at all, which is why a stray file on the
        // reserved path also counts as dead (see the next test).
        let root = fake_container_root();
        let root_str = root.path().to_str().unwrap();
        let path = UnixSocketHandler::socket_path_under(root_str, 7);

        // No file at all: says nothing about a dead listener, so it must not be
        // reported as one.
        assert!(
            matches!(
                UnixSocketHandler::probe_socket(&path).await,
                SocketProbe::Inconclusive(_)
            ),
            "a missing path must not be reported dead"
        );

        // A socket file whose listener is gone: the crash residue to reclaim.
        let stale = UnixSocketHandler::bind_listener_at(root_str, 7)
            .await
            .expect("first bind");
        drop(stale);
        assert!(
            matches!(
                UnixSocketHandler::probe_socket(&path).await,
                SocketProbe::Dead(_)
            ),
            "a socket with no listener behind it must be reported dead"
        );

        // And a listener that is there.
        let live = UnixSocketHandler::bind_listener_at(root_str, 7)
            .await
            .expect("rebind");
        assert!(
            matches!(
                UnixSocketHandler::probe_socket(&path).await,
                SocketProbe::Live
            ),
            "a bound listener must be reported live"
        );
        drop(live);
    }

    #[tokio::test]
    async fn bind_listener_reclaims_a_path_that_is_not_a_socket() {
        // Pins the platform behaviour the reclaim branch relies on: connect(2)
        // to a non-socket inode is refused, so a stray regular file squatting on
        // the reserved socket path is replaced instead of failing every run on
        // that host. Nothing can be serving lifecycle events through it.
        let root = fake_container_root();
        let root_str = root.path().to_str().unwrap();
        let path = UnixSocketHandler::socket_path_under(root_str, 7);
        fs::write(&path, b"not a socket").expect("write a placeholder file");

        let listener = UnixSocketHandler::bind_listener_at(root_str, 7)
            .await
            .expect("a non-socket placeholder is reclaimed");
        std::os::unix::net::UnixStream::connect(&path).expect("the path is a socket now");
        drop(listener);
    }

    #[tokio::test]
    async fn bind_listener_reclaims_a_stale_socket() {
        let root = fake_container_root();
        let root_str = root.path().to_str().unwrap();
        let path = UnixSocketHandler::socket_path_under(root_str, 7);

        let stale = UnixSocketHandler::bind_listener_at(root_str, 7)
            .await
            .expect("first bind");
        drop(stale);
        // Dropping a listener does not unlink it, which is exactly the crash
        // residue this branch exists to clean up.
        assert!(
            std::path::Path::new(&path).exists(),
            "expected the stale socket file to survive the drop"
        );

        let reclaimed = UnixSocketHandler::bind_listener_at(root_str, 7)
            .await
            .expect("a stale socket is reclaimed");
        std::os::unix::net::UnixStream::connect(&path).expect("reclaimed socket is live");
        let _accepted = reclaimed.accept().await.expect("reclaimed socket accepts");
    }

    #[tokio::test]
    async fn a_path_that_vanished_before_the_probe_is_rebound() {
        // The owner can exit - and unlink its own socket - between the bind that
        // reported AddrInUse and the probe. The probe then answers about a path
        // that is already gone, which is Inconclusive, and the run used to fail
        // permanently on a path that was in fact free. `resolve_occupied` is the
        // post-AddrInUse half of `bind_listener_at`, so driving it against a
        // missing path reproduces that end state without having to win the race.
        let root = fake_container_root();
        let root_str = root.path().to_str().unwrap();
        let path = UnixSocketHandler::socket_path_under(root_str, 7);
        assert!(
            !Path::new(&path).exists(),
            "the path must start out free, as it is once the owner unlinks it"
        );

        let listener = UnixSocketHandler::resolve_occupied(&path, 7)
            .await
            .expect("a path that vanished before the probe must be rebound");
        std::os::unix::net::UnixStream::connect(&path).expect("the rebound path is a live socket");
        drop(listener);
    }

    #[tokio::test]
    async fn the_vanished_path_retry_is_a_single_bounded_attempt() {
        // The retry must not become a loop and must not paper over a path that
        // cannot be bound at all: with no parent directory the one rebind fails
        // too, so the caller still gets the inconclusive-probe error, and the
        // attempt creates nothing on the filesystem.
        let root = fake_container_root();
        let missing_root = format!("{}/no-such-dir", root.path().to_str().unwrap());
        let path = UnixSocketHandler::socket_path_under(&missing_root, 7);

        let err = UnixSocketHandler::resolve_occupied(&path, 7)
            .await
            .expect_err("an unbindable path stays an error");
        assert!(
            err.to_string().contains("liveness could not be determined"),
            "the retry must fall through to the inconclusive error, got: {}",
            err
        );
        assert!(
            !Path::new(&missing_root).exists(),
            "the retry binds, it does not create the path it needs"
        );
    }
}
