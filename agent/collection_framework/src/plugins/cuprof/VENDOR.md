# Vendored cuprof — provenance and local patches

This directory is a **vendored copy** of cuprof, not a git submodule and not a
subtree. Do not edit files here ad hoc: either change upstream and re-sync, or
add a patch under `patches/` and record it below.

| | |
|---|---|
| Upstream | `cuprof` — clone URL is maintained internally and deliberately **not** published here; ask an AIProf maintainer for it |
| Base branch | `main` |
| Base commit | `84cb9d8` — "fix: keep libcuprof.so loaded; rewrite the unload contract" (2026-08-12) |
| Last synced | 2026-09-10 — re-verified *internally* that upstream `main` is still `84cb9d8`. Not independently checkable, because the clone URL above is deliberately unpublished. |
| License | Apache-2.0. `LICENSE` and `NOTICE` here are upstream's, unmodified. |

Everything in this tree is byte-identical to `cuprof@84cb9d8` **except** the
ten files touched by the five patches below, plus the two files patch 0005
adds, `src/bounded_call.h` and `src/bounded_call.cc`, which do not exist
upstream. Each of those twelve files carries an Apache-2.0 section 4(b) notice
in its first lines: a code comment in the nine source files, reading "differs
from upstream" for the seven that were modified and "does not exist in
upstream" for the two that were added, and an HTML comment in `README.md`,
`docs/embedding.md` and `docs/design.md` so those rendered pages are
unchanged.

Reproducibility is checked by applying `patches/*` to a pristine
`cuprof@84cb9d8` checkout and diffing against this tree — the result must be
empty apart from `patches/` and this file. That base commit's tree is
`55309c542a5fd60662281c15b53c459f174acbb3`, so whoever has the internal clone
can confirm they are diffing the same contents; nothing in this repository
pins them, and the check cannot be run without that clone.

## Local patches

### `patches/0001-rpath-origin-for-ptrace-injection.patch`

`Makefile`. Puts `$ORIGIN` ahead of `$(CUPTI_HOME)/lib64` in `libcuprof.so`'s
RPATH.

Upstream only supports the `CUDA_INJECTION64_PATH` launch flow, where the
library is loaded in the build host's own namespace. CollectionFramework instead
ptrace-injects it into an **already-running** target process, so `dlopen` and
symbol resolution happen in the target's address space using the target's search
paths — the build-time CUPTI path may not exist there (different CUDA major
version, or a container build against a bare-metal target). `$ORIGIN` resolves
to the directory the injected copy was staged into (`/tmp` inside the target's
mount namespace), where `CUPTIPluginWrapper::stage_cupti_runtime()` drops a
vendored `libcupti.so.<major>` matching the library's `DT_NEEDED`.

Consumer: `src/plugins/cupti_plugin_wrapper.rs`.

### `patches/0002-align-cupti-epoch-to-clock-monotonic.patch`

`src/config.{h,cc}`, `src/trace_writer.{h,cc}`, `src/cupti_sink.cc`. Converts
CUPTI's `CLOCK_REALTIME` (epoch) nanosecond timestamps to `CLOCK_MONOTONIC`.

pyki / `torch.profiler` timestamps are `CLOCK_MONOTONIC`. AIProf renders both
sources on a single Perfetto timeline, so an unadjusted CUPTI trace lands
decades away from the Python-stack lanes and the fused view is unusable. The
offset `(CLOCK_REALTIME - CLOCK_MONOTONIC)` is computed once in `LoadConfig()`
and subtracted per event in `WriteChromeTrace()`. A timestamp at or below the
offset is left unchanged rather than subtracted, so it can never wrap `uint64`
(it does *not* clamp to 0). The event **duration** is *not* derived from the two
shifted endpoints — see `patches/0004-duration-from-raw-endpoint-pair.patch`.

Note this implements "framework/Python stack fusion", which upstream lists as a
**non-goal** in `docs/design.md`. Keep it local unless upstream adopts it.

### `patches/0003-readme-aiprof-cupti-alignment-note.patch`

`README.md`. Puts a note at the top of the vendored README that sends a builder
to `../../third_party/cupti/README.md`, and records that upstream's "no CUDA
libraries are vendored" claims describe upstream cuprof as a stand-alone
repository rather than this tree.

Upstream links against the system CUPTI and vendors nothing, so its README says
nothing about which CUPTI a build ends up needing. AIProf ptrace-injects the
compiled `libcuprof.so` into arbitrary target processes, where the CUPTI ABI is
tied to the build machine's CUDA major and a mismatch with the target host's
driver silently SIGSEGVs the injected process. That constraint is documented one
level up in `src/third_party/cupti/`, which a reader of this README has no reason
to open.

Upstream's `NOTICE` ends with the same "No NVIDIA libraries or headers are
distributed with this repository" claim, which is likewise true of upstream and
false of AIProf. It is deliberately **left unmodified**: it is upstream's
attribution file, and the correction belongs in the AIProf-local README note, not
in an edit to an upstream licence document.

Consumer: `src/third_party/cupti/README.md`.

### `patches/0004-duration-from-raw-endpoint-pair.patch`

`src/trace_writer.cc`. Takes each event's `"dur"` from the raw, unshifted
endpoint pair instead of subtracting the two values patch 0002 shifted.

Patch 0002 shifts `start_ns` and `end_ns` separately, which is only sound while
both endpoints land on the same side of the offset. The offset is computed once
in `LoadConfig()` and is roughly the wall-clock time at boot, so it is only
valid while `CLOCK_REALTIME` keeps advancing. A step **backwards** by more than
the uptime — a VM resumed from a snapshot with a stale RTC, an explicit
`date -s`, or a large `chronyc makestep` on a host that booted with a badly
wrong clock — puts later CUPTI timestamps below an offset derived from the
pre-step clock. (Ordinary NTP slewing cannot: it would have to move the clock
back past boot time.) An event that straddles the offset then gets one endpoint
shifted and the other left alone, so
`adj_end - adj_start` underflows `uint64` to roughly 1.8e19 ns. The emitted
duration becomes ~1.8e16 us, which stretches Perfetto's time bounds by six
orders of magnitude and flattens every real kernel into a single pixel, while
`WriteChromeTrace()` still returns `true` and CollectionFramework still reports
a successful collection. Nothing downstream can tell the trace is corrupt.

An `end_ns` of `0` underflows the same way but cannot reach the writer through
cuprof's own path: `IngestRecord` in `src/cupti_sink.cc` already drops records
with `end == 0 || end < start`, for exactly this anti-wrap reason. It is fixed
here anyway because `WriteChromeTrace()` is exported in `src/trace_writer.h` and
cannot assume that every embedder filtered its records first.

The trigger is narrow, so treat this as hardening an exported function rather
than as a field bug: `WriteChromeTrace()` is declared in `src/trace_writer.h` and
has no business assuming its caller kept the two endpoints on one side of a shift
that the caller never sees. A duration is invariant under a uniform shift, so the
raw pair is the correct source; a reversed or unset pair clamps to zero. Timestamps keep patch 0002's
semantics exactly.

Unlike 0002 this is not an AIProf-specific policy but a latent bug in the
conversion 0002 introduces, so if 0002 is ever offered upstream this fix should
travel with it.

Consumer: `test/native/test_cuprof_trace_writer.cc`, which covers both shapes (a
straddling event and an unset `end_ns`) and is the only automated coverage of
either. Run it with `make test-native`; against the unfixed writer those two
cases report durations of 1.67e16 us and 1.84e16 us.

### `patches/0005-bound-the-cupti-teardown-in-stop.patch`

`src/cupti_sink.{h,cc}`, `src/config.{h,cc}`, `README.md`,
`docs/embedding.md`, `docs/design.md` and `Makefile`, plus new files
`src/bounded_call.h` and `src/bounded_call.cc`. Runs the CUPTI teardown in
`CuptiSink::Stop()` under a deadline instead of calling it inline. On timeout
it reports `CUPTIProfilingFailed`, keeps whatever CUPTI had already delivered,
and retires the instance, so no later window is collected in a process whose
driver state is unknown.

`Stop()` called `cuptiActivityDisable()` three times and then
`cuptiActivityFlushAll(1)` directly on the duration thread, and that teardown
can wedge inside the driver and never return. `gdb` on a hung target shows
`Stop()` blocked in `pthread_rwlock_wrlock` inside `libcuda.so.1` under
`cuptiActivityDisable`, the application's own thread blocked in
`pthread_rwlock_wrlock` under `cuLaunchKernel` on a *different* lock, and
CUPTI's worker threads parked in `sem_wait` inside `libcupti.so.12`. Measured
on a 4x A10 host with driver 580.126.09 against a workload that launches
kernels continuously, roughly a third of collection windows hung there. A
library built from the tree without this patch, alternated window by window
against the patched one on the same host, the same workload and the same
CollectionFramework binary, hung at the same rate: 7 successes in 12 windows
against 8 in 12, Fisher's exact two-sided p = 1.0. Every failure on the
unpatched side was the 65s idle watchdog with no file at all, while three of
the patched side's four ended in 37s with a 44-byte trace, a line on the
target's stderr and a `CollectFailed`, and the fourth was a `Start()` wedge,
which this patch deliberately does not bound. Two earlier alternating batches
agree in direction (6 of 8 against 5 of 8, and 9 of 12 against 10 of 12), and
a batch that halved the target's launch rate scored 5 of 8, inside the range
the others scored, so neither this tree nor launch pressure moves the rate.

What makes it worth patching is not the hang but what follows from it.
`Stop()` never reached `WriteChromeTrace`, so no trace file appeared, and it
never reached either terminal `Notify`, so an orchestrator waiting for
`CUPTIProfilingWriterOver` got nothing and fell back to its own idle watchdog.
CollectionFramework's watchdog then exits 0, so a window that had collected
nothing reported success. With the bound in place the same wedge produces one
line on the target's stderr naming the timeout, a trace file holding whatever
CUPTI had delivered so far, and `CUPTIProfilingFailed`, the terminal message
that replaces `WriterOver`. That message used to be documented in
`docs/embedding.md` as "no file at all", which this patch made untrue, so the
patch updates the table too. In the observed wedge the file is an empty trace,
because the hang is at the first `cuptiActivityDisable`, before any buffer was
flushed, so the gain is not the data: it is that the window ends in 37s with a
terminal message and the collector marked `Failed` in the report, instead of
running out the orchestrator's 60s idle budget in silence. Measured with a 5s
window, where the floor below applies: cuprof printed `CUPTI teardown did not
finish within 30000 ms`, CollectionFramework handled `CollectFailed`, copied
the file out and exited in 37s.

The bound is 30s by default and `CUPROF_TEARDOWN_TIMEOUT_MS` overrides it,
clamped to 1000..600000 ms. It is sized as a wedge detector rather than as an
allowance for slow teardowns, and it is flat rather than derived from the
window length. On the 4x A10 host above, the heaviest teardown measured - a
20s window that collected 4475153 events into a 1.24 GB trace - took 82 ms of
`cuptiActivityDisable` plus `cuptiActivityFlushAll`, so the default is roughly
370x the worst healthy case observed. It does not scale with the window
because the teardown only drains what CUPTI is still holding at `Stop()`, and
buffers are delivered continuously while the window runs, so that residual is
not a function of duration.

What does grow with the window is the `WriteChromeTrace` that follows, which
took 7.8s for that same 1.24 GB trace, and this bound does not cover it. That
is worth stating plainly because it limits what the bound buys:
CollectionFramework waits `duration + 60s` for a terminal message, and on a
long enough window the write alone can exceed that slack, patched or not.
Bounding the teardown keeps the wedged case inside the budget; it does not
make every healthy case fit one.

`RunBounded` gets its own translation unit rather than living in
`cupti_sink.cc` because it has no CUDA or CUPTI dependency, which is what
makes it testable without a GPU. On timeout the worker thread, its mutex, its
condvar and the copied `std::function` are leaked deliberately: the thread may
still lock that mutex and signal at any later point, so none of it may be
freed, and a thread stuck inside the driver cannot be cancelled without
risking the target's address space. The leak is one thread per wedged window
and nothing leaks from a window that ends normally, so a target that profiles
for days without a wedge leaks nothing, and one that wedges does not reach a
second window (below). CUPTI's activity calls are process-global, so moving
them onto another thread does not change what they observe. The wait is on
`CLOCK_MONOTONIC` wherever the platform accepts it, because a wall clock step
is the same hazard patch 0004 defends the timestamps against, and would
otherwise collapse the bound on a forward jump or stretch it on a backward
one.

A flush that finishes after the timeout appends to an `events_` vector that
`Stop()` has already swapped empty, and those records belong to a window that
was already reported failed. Were a later window to reuse the instance, they
would be written into *its* trace and announced as a normal `WriterOver`,
which is a wrong result rather than a lost one, so `Start()` clears `events_`
before enabling anything. That keeps "one trace holds one window" true on its
own, independent of the retiring below. For the same reason `Start()` is
refused while a stop is still in flight, which before the bound could not
happen: the teardown now runs for up to `Config::teardown_timeout_ms` with the
mutex released, and a window started inside that interval would have its
records swapped out and written by the previous window's stop, and its
activity kinds disabled by a teardown it never asked for.

This bounds `Stop()` only. The same driver lock can also wedge `Start()`,
inside `cuptiActivityEnable`, which runs within `InitializeInjection()` on the
injector's ptrace call; on the same host that appeared in roughly one window
in ten, with `InitializeInjection()` never returning and CollectionFramework's
own watchdog killing the injector after 65s. It is deliberately not bounded
here: abandoning a half-enabled activity API leaves CUPTI buffers in the
target that nothing will ever drain, so it needs a different answer than a
deadline.

A timed-out teardown also retires the instance: `Start()` afterwards fails
fast with its own `CUPTIProfilingFailed` instead of collecting. The state is
not recoverable from inside the library. A thread is still in the driver
holding locks this class cannot see, and the teardown it never finished can
still disable an activity kind that a new window has just enabled, which would
produce a window that is silently empty and still reported as a good one.

`Stop()` has no exception guard, so a throw from `WriteChromeTrace`, which is
a real possibility for a trace larger than available memory, escapes whichever
thread called it, the duration timer or the `atexit` handler, and terminates
the target. That is upstream's behaviour and this patch does not change it,
but it is why the in-flight flag has no recovery path: the process that would
need one is already gone. Making that write failure-tolerant is a separate
change.

Worth being precise about who reaches that second `Start()`, because a
terminal message is new here and an orchestrator may act on it.
CollectionFramework's scheduler answers `CollectFailed` on a signal-triggered
collector with a non-zero duration by starting the next window for that pid
immediately, and before this patch a wedge produced no message at all, so
nothing restarted. It still does not re-enter cuprof:
`CUPTIPluginWrapper::custom_trigger` refuses to trigger a pid whose state is
already past `InitSuccess`. The path the retiring actually guards is cuprof's
own documented one, `docs/embedding.md` step 4 and `CuptiSink::Start()`'s
contract, under which one loaded instance serves consecutive windows and an
embedder calls `cuprof_start()` again. Nothing in the library may assume an
embedder has a state machine that declines the second call, and
`InitializeInjection()`'s return value is discarded by both the CUDA driver
and CollectionFramework's injector, so a silent refusal would leave the window
unreported.

Like 0004 and unlike 0001 and 0003, this is not AIProf-specific policy: any
embedder that calls `Stop()` from a timer thread has the same exposure, so it
should travel upstream if 0002 ever does.

Consumer: `test/native/test_cuprof_bounded_call.cc`, which covers the
primitive the fix rests on. A body that never returns must not hang the
caller, the deadline must be honoured rather than approximated, the body must
run off the caller's thread, and a later bounded call must still work after an
earlier one timed out. It is the only automated coverage of this patch,
because the wedge itself needs a driver that produces one. Run it with `make
test-native`; against a `RunBounded` that calls the body inline the permanent-
block case hangs and takes the suite with it.

## Re-syncing to a newer upstream

```bash
git clone <internal-cuprof-url> /tmp/cuprof-upstream   # URL not published; see the row above
cd /tmp/cuprof-upstream && git checkout <new-base-commit>

# from the AIProf repo root
D=agent/collection_framework/src/plugins/cuprof
rsync -a --delete --exclude='.git' --exclude='patches' --exclude='VENDOR.md' \
      /tmp/cuprof-upstream/ "$D"/
cd "$D" && git apply patches/000*.patch
```

A patch that no longer applies means upstream touched the same lines — resolve
by hand, then re-generate it with `git format-patch` and update this file. Two
caveats on regenerating: the `From <sha>`, `From:` and `Date:` lines of these
five patches are pinned by hand, because they are not commits in any repository a
reader can reach, so only the diff body and the diffstat need to match
`git format-patch` output; and the `Subject: [PATCH n/5]` counters have to stay
in step with the length of the series.

After syncing, always:

1. Update the **Base commit** and **Last synced** rows above.
2. **Run `cargo test` in `agent/collection_framework/`.** Three drift guards in
   `src/unix_handler.rs` fail the build if upstream renamed anything we depend
   on, so this step replaces the by-hand checks that used to be listed here:
   - `drift_guard_vendored_cuprof_emits_every_message_we_match_on` — the four
     `CUPTIProfiling*` socket messages, matched **by string equality** against
     `src/const.rs`. This is the coupling most likely to break silently.
   - `drift_guard_vendored_cuprof_reads_every_config_key_we_write` —
     `CUPROF_OUTPUT` / `DURATION` / `VERBOSE` / `SOCKET`, the four keys
     `write_cuprof_config()` in `cupti_plugin_wrapper.rs` emits, plus
     `CUPROF_CONFIG`, which that function does **not** write: it is the
     environment key `LoadCfgFile()` reads to locate the `.cfg` file, and it is
     part of the same contract. Patch 0005's `CUPROF_TEARDOWN_TIMEOUT_MS` is
     deliberately absent from that list, because CollectionFramework does not
     write it either: cuprof defaults the bound to 30s whatever the window
     length, and the key exists for an embedder whose host or idle budget
     differs.
   - `drift_guard_vendored_cuprof_keeps_the_cfg_filename_convention` — the
     `/tmp/cuprof_<pid>.cfg` path, which `write_cuprof_config()` writes into the
     target's namespace and `LoadCfgFile` must look for by exactly that name.
3. The guards are substring checks and none of them covers
   `src/trace_writer.cc` or `src/bounded_call.cc`, so after a re-sync also run
   `make test-native` — it is the only automated check on patches 0002, 0004 and
   0005.
4. Rebuild the client image — `deploy/docker/Dockerfile.client` stage 1b builds
   this tree with `nvidia/cuda:${CUDA_VERSION}-devel-${BUILDER_UBUNTU}`, which
   defaults to `12.2.0-devel-ubuntu20.04` (20.04 deliberately, so the injected
   library's glibc symbols stay old enough for Anolis 8 / CentOS 8 targets).
   `CUDA_VERSION` has to match the CUDA major of the hosts the client will inject
   into, or injection SIGSEGVs — see `src/third_party/cupti/README.md`.
