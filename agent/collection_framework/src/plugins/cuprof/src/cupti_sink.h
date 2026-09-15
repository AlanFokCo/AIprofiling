// AIProf-local modification (Apache-2.0 4(b)): this file differs from
// upstream cuprof. See VENDOR.md and patches/0005-bound-the-cupti-teardown-in-stop.patch.
#ifndef CUPROF_CUPTI_SINK_H
#define CUPROF_CUPTI_SINK_H

#include "config.h"
#include "trace_writer.h"

#include <pthread.h>

#include <vector>

namespace cuprof {

// Owns the CUPTI activity subscription and the collected events.
//
// Lifetime: created once from cuprof_start(), flushed once at process
// exit (or after Config::duration_sec). Activity callbacks arrive on
// CUPTI-owned threads, so Append() must stay thread-safe. The instance is
// deliberately leaked rather than destroyed at exit, because such a callback can
// still be in flight when the target begins to exit; the destructor below is
// therefore never invoked.
class CuptiSink {
  public:
    static CuptiSink& Instance();

    // Enables activity collection. Calling it while already running is a
    // no-op that returns true. After a Stop(), Start() may be called again
    // to collect a fresh window (same loaded instance, new output file),
    // unless a teardown timed out, which retires the instance; see Stop().
    bool Start(const Config& cfg);

    // Disables collection, drains CUPTI buffers, writes the trace file.
    // Idempotent; concurrent callers are serialized and only the first one
    // performs the stop sequence.
    //
    // The CUPTI teardown is bounded by Config::teardown_timeout_ms. A timeout
    // leaves a thread inside the driver and CUPTI's activity state unknown, so
    // the window is reported failed with whatever was already delivered, and
    // every later Start() in this process fails fast instead of collecting.
    // Start() is also refused while a stop is still in flight, so two windows
    // can never share one sink.
    void Stop();

    void Append(const Event& e);

    const Config& config() const { return cfg_; }
    bool running() const { return running_; }

  private:
    CuptiSink();
    ~CuptiSink();
    CuptiSink(const CuptiSink&);
    CuptiSink& operator=(const CuptiSink&);

    Config cfg_;
    bool running_;
    bool stopped_;
    // Set when a teardown timed out; see Stop(). Guarded by mu_.
    bool wedged_;
    // Set for the whole of Stop(), whose teardown can run for
    // Config::teardown_timeout_ms with mu_ released. Guarded by mu_.
    bool teardown_in_flight_;
    pthread_mutex_t mu_;
    std::vector<Event> events_;
};

}  // namespace cuprof

#endif  // CUPROF_CUPTI_SINK_H
