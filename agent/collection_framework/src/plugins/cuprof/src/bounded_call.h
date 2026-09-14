// AIProf-local addition (Apache-2.0 4(b)): this file does not exist in
// upstream cuprof. See VENDOR.md and patches/0005-bound-the-cupti-teardown-in-stop.patch.
#ifndef CUPROF_BOUNDED_CALL_H
#define CUPROF_BOUNDED_CALL_H

#include <functional>

namespace cuprof {

// Runs `body` on a detached thread and waits at most `timeout_ms` for it.
// Returns true if it finished in time.
//
// On timeout the thread and its heap state are leaked deliberately. A thread
// stuck inside the CUDA driver cannot be cancelled without risking the target's
// address space, and it may signal its condvar at any later point, so neither it
// nor the condvar may be freed. `body` is copied into that heap state, so it has
// to own everything it touches: after a timeout it can still run minutes later,
// long after the caller returned. Capturing by reference, or reaching for
// anything with a shorter lifetime than the process, is a use-after-free.
//
// If the synchronisation cannot be initialised, the thread cannot be created, or
// the clock the deadline would be read from cannot be, `body` runs inline and the
// result is true, so a failure of this helper degrades to the unbounded behaviour
// it replaces rather than to doing nothing, and never to reporting a timeout that
// nothing measured.
bool RunBounded(std::function<void()> body, unsigned timeout_ms);

}  // namespace cuprof

#endif  // CUPROF_BOUNDED_CALL_H
