// SPDX-License-Identifier: Apache-2.0
//
// Native regression test for AIProf's local cuprof patch 0005
// (patches/0005-bound-the-cupti-teardown-in-stop.patch), which runs the CUPTI
// teardown in CuptiSink::Stop() under a deadline instead of calling it inline.
//
// The teardown can wedge inside the driver: cuptiActivityDisable takes a
// libcuda write lock that a target thread inside cuLaunchKernel may never
// release, while CUPTI's own worker threads park on semaphores. When that
// happened, cuprof wrote no trace, sent no terminal message, and the
// orchestrator's only recourse was its idle watchdog followed by a successful
// exit on an empty report. Stop() now writes out whatever CUPTI already
// delivered and reports CUPTIProfilingFailed instead.
//
// That behaviour cannot be tested without a wedged driver, but the primitive it
// rests on can be: bounded_call.cc has no CUDA/CUPTI dependency, so this builds
// and runs on any Linux box with a C++11 compiler and pthreads - no GPU, no
// CUDA toolkit, no container:
//
//     make -C test/native && ./test/native/test_cuprof_bounded_call
//
// The point of TestPermanentlyBlockedBodyDoesNotHangTheCaller is that it would
// hang forever, and take the test suite with it, if RunBounded ever went back
// to calling the body inline.

#include <pthread.h>
#include <time.h>

#include <chrono>
#include <functional>
#include <iostream>
#include <string>

// Resolved via -I$(CUPROF)/src in the sibling Makefile.
#include "bounded_call.h"

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
    std::cout << (ok ? "  ok   " : "  FAIL ") << what << "\n";
    if (!ok) ++g_failures;
}

long long ElapsedMs(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - start)
        .count();
}

// Never signalled. A body that waits on this pair stands in for a thread wedged
// inside the driver: it does not fail, it does not return, and it cannot be
// cancelled.
pthread_mutex_t g_never_mu = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t g_never_cv = PTHREAD_COND_INITIALIZER;

void BlockForever() {
    pthread_mutex_lock(&g_never_mu);
    pthread_cond_wait(&g_never_cv, &g_never_mu);
    pthread_mutex_unlock(&g_never_mu);
}

void TestFastBodyCompletesOnAnotherThread() {
    std::cout << "TestFastBodyCompletesOnAnotherThread\n";
    pthread_t caller = pthread_self();
    pthread_t seen;
    bool ran = false;
    bool ok = cuprof::RunBounded(
        [&] {
            seen = pthread_self();
            ran = true;
        },
        5000);
    Check(ok, "a body that finishes well inside the deadline reports success");
    Check(ran, "the body actually ran");
    Check(!pthread_equal(caller, seen),
          "the body ran on its own thread, not inline on the caller's");
}

void TestBodyRunsExactlyOnce() {
    std::cout << "TestBodyRunsExactlyOnce\n";
    int calls = 0;
    cuprof::RunBounded([&] { ++calls; }, 5000);
    Check(calls == 1, "the body is invoked exactly once, not retried");
}

void TestSlowBodyTimesOutWithoutWaitingForIt() {
    std::cout << "TestSlowBodyTimesOutWithoutWaitingForIt\n";
    auto start = std::chrono::steady_clock::now();
    bool ok = cuprof::RunBounded(
        [] {
            timespec nap = {1, 0};  // 1s, against a 50ms deadline
            nanosleep(&nap, NULL);
        },
        50);
    long long took = ElapsedMs(start);
    Check(!ok, "a body slower than the deadline reports failure");
    // The upper bound is deliberately loose: on a loaded CI runner the caller
    // can sit for seconds between the deadline expiring and this line running,
    // and a test that flakes teaches contributors to ignore it. What this
    // asserts is that `took` stays far below the body's own 1s sleep; a genuine
    // hang (RunBounded waiting for the body after all) is caught by the
    // workflow's timeout-minutes, not by this number.
    Check(took < 4000,
          "the caller returned without waiting for the body (took " +
              std::to_string(took) + "ms for a 50ms bound and a 1s body)");
}

void TestPermanentlyBlockedBodyDoesNotHangTheCaller() {
    std::cout << "TestPermanentlyBlockedBodyDoesNotHangTheCaller\n";
    auto start = std::chrono::steady_clock::now();
    // This is the shape of the real failure: the body never returns at all.
    bool ok = cuprof::RunBounded(
        BlockForever, 100);
    long long took = ElapsedMs(start);
    Check(!ok, "a body that never returns still reports failure");
    // `took >= 90` is the assertion that carries the weight: it proves the
    // caller really waited out the 100ms deadline instead of giving up
    // immediately, so it stays tight. Only the upper bound is loose, for the
    // same reason as in TestSlowBodyTimesOutWithoutWaitingForIt above.
    Check(took >= 90 && took < 30000,
          "the deadline was honoured rather than approximated (took " +
              std::to_string(took) + "ms for a 100ms bound)");
}

void TestCallerKeepsRunningAfterATimeout() {
    std::cout << "TestCallerKeepsRunningAfterATimeout\n";
    cuprof::RunBounded(BlockForever, 50);
    // Stop()'s fallback path has to be able to do real work after a timeout:
    // take the mutex, swap the events out, write the trace, notify.
    int reached = 0;
    bool ok = cuprof::RunBounded([&] { reached = 42; }, 5000);
    Check(ok && reached == 42,
          "a later bounded call still works after an earlier one timed out");
}

}  // namespace

int main() {
    TestFastBodyCompletesOnAnotherThread();
    TestBodyRunsExactlyOnce();
    TestSlowBodyTimesOutWithoutWaitingForIt();
    TestPermanentlyBlockedBodyDoesNotHangTheCaller();
    TestCallerKeepsRunningAfterATimeout();
    std::cout << (g_failures == 0 ? "\nALL PASS\n" : "\nFAILURES: ")
              << (g_failures == 0 ? "" : std::to_string(g_failures)) << "\n";
    return g_failures == 0 ? 0 : 1;
}
