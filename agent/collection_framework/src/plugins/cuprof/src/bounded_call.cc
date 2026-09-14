// AIProf-local addition (Apache-2.0 4(b)): this file does not exist in
// upstream cuprof. See VENDOR.md and patches/0005-bound-the-cupti-teardown-in-stop.patch.
#include "bounded_call.h"

#include <pthread.h>
#include <time.h>

namespace cuprof {
namespace {

struct BoundedWork {
    std::function<void()> body;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    // The clock `cv`'s deadline is measured against. Recorded rather than
    // assumed, because not every platform lets a condvar use CLOCK_MONOTONIC.
    clockid_t clock;
    bool done;
};

void* RunAndSignal(void* arg) {
    BoundedWork* work = static_cast<BoundedWork*>(arg);
    work->body();
    pthread_mutex_lock(&work->mu);
    work->done = true;
    pthread_cond_signal(&work->cv);
    pthread_mutex_unlock(&work->mu);
    return NULL;
}

// Creates the mutex and the condvar and records which clock the condvar waits
// on. Returns false if either could not be created, leaving nothing initialised
// behind that the caller would have to tear down.
bool InitSync(BoundedWork* work) {
    if (pthread_mutex_init(&work->mu, NULL) != 0) return false;

    // The deadline has to be immune to wall clock steps, which a profiling
    // target performs routinely: an NTP makestep, or a stale RTC corrected
    // after boot. Measured against CLOCK_REALTIME, a forward jump collapses the
    // bound and fails a healthy window, while a backward one stretches it past
    // the caller's budget. Patch 0004 exists because the same kind of step
    // corrupts event durations; this is that hazard on the teardown path.
    //
    // pthread_condattr_setclock is POSIX but not universal: Darwin's libc does
    // not provide it, and its condvars always wait on the wall clock. This
    // library only ever ships into a Linux target, where it does exist, so the
    // fallback is there to keep the GPU-free test in test/native buildable on a
    // developer's macOS host rather than to support one.
    clockid_t clock = CLOCK_REALTIME;
    pthread_condattr_t attr;
    const bool have_attr = pthread_condattr_init(&attr) == 0;
    if (have_attr) {
#if defined(CLOCK_MONOTONIC) && defined(__linux__)
        if (pthread_condattr_setclock(&attr, CLOCK_MONOTONIC) == 0) clock = CLOCK_MONOTONIC;
#endif
    }
    const int rc = pthread_cond_init(&work->cv, have_attr ? &attr : NULL);
    if (have_attr) pthread_condattr_destroy(&attr);
    if (rc != 0) {
        pthread_mutex_destroy(&work->mu);
        return false;
    }
    work->clock = clock;
    return true;
}

// `pthread_cond_timedwait` wants an absolute deadline on the condvar's own
// clock, not a duration. Returns false if the clock could not be read, in which
// case there is no deadline to wait for.
bool DeadlineFromNow(clockid_t clock, unsigned timeout_ms, timespec* out) {
    if (clock_gettime(clock, out) != 0) return false;
    out->tv_sec += timeout_ms / 1000;
    out->tv_nsec += static_cast<long>(timeout_ms % 1000) * 1000000L;
    if (out->tv_nsec >= 1000000000L) {
        out->tv_sec += 1;
        out->tv_nsec -= 1000000000L;
    }
    return true;
}

void DestroyWork(BoundedWork* work) {
    pthread_mutex_destroy(&work->mu);
    pthread_cond_destroy(&work->cv);
    delete work;
}

}  // namespace

bool RunBounded(std::function<void()> body, unsigned timeout_ms) {
    BoundedWork* work = new BoundedWork();
    work->body = body;
    work->done = false;
    if (!InitSync(work)) {
        delete work;
        body();
        return true;
    }

    // Computed before the thread exists, so that a clock failure degrades the
    // same way a thread-creation failure does instead of leaving a thread running
    // against a deadline nobody could read.
    timespec deadline = {0, 0};
    if (!DeadlineFromNow(work->clock, timeout_ms, &deadline)) {
        DestroyWork(work);
        body();
        return true;
    }

    pthread_t tid;
    if (pthread_create(&tid, NULL, RunAndSignal, work) != 0) {
        DestroyWork(work);
        body();
        return true;
    }
    pthread_detach(tid);

    pthread_mutex_lock(&work->mu);
    // Loop for spurious wakeups; `done` is the only thing that ends the wait
    // early, and the deadline is absolute so a wakeup costs nothing. ETIMEDOUT is
    // the deadline. Any other non-zero return means the wait itself is broken,
    // and the honest outcome is the same one: nothing here can claim the body
    // finished, so fall through to the timeout path, which leaks the state and
    // reports failure rather than success.
    int rc = 0;
    while (!work->done && rc == 0) {
        rc = pthread_cond_timedwait(&work->cv, &work->mu, &deadline);
    }
    const bool finished = work->done;
    pthread_mutex_unlock(&work->mu);

    if (finished) {
        DestroyWork(work);
        return true;
    }
    // Timed out. `work`, its mutex, its condvar and the copy of `body` are
    // leaked on purpose: the thread may still lock that mutex and signal at any
    // later point, so none of it may be freed, and a thread stuck inside the
    // CUDA driver cannot be cancelled without risking the target's address
    // space.
    return false;
}

}  // namespace cuprof
