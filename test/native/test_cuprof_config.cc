// SPDX-License-Identifier: Apache-2.0
//
// Native regression test for the teardown-timeout resolution that AIProf's local
// cuprof patch 0005 (patches/0005-bound-the-cupti-teardown-in-stop.patch) added
// to cuprof::LoadConfig().
//
// CuptiSink::Stop() now runs the CUPTI teardown under a deadline rather than
// calling it inline (test_cuprof_bounded_call.cc covers the deadline primitive
// itself). That deadline is Config::teardown_timeout_ms, and LoadConfig()
// resolves it to a flat default of 30000 ms whatever the profiling window is.
//
// The bound deliberately does not scale with CUPROF_DURATION: the teardown only
// drains what CUPTI is still holding at Stop(), and CUPTI delivers its activity
// buffers continuously while the window runs instead of accumulating all of them
// until the end, so a longer window does not leave proportionally more work for
// the teardown. The measurement that once justified scaling - "about 7s for a
// 20s window" - turned out to be the trace write that follows the teardown, not
// the teardown itself. The heaviest teardown observed on a 4x A10 host (a 20s
// window that collected 4475153 events into a 1.24 GB trace) spent 82 ms in
// cuptiActivityDisable plus cuptiActivityFlushAll; that same run took 7.8s to
// write the trace, which this bound does not cover at all. 30000 ms is therefore
// orders of magnitude of headroom over the real cost, and staying that low keeps
// an orchestrator that gives up a fixed time after the window ends able to report
// a hung teardown rather than being watchdog-killed by it.
//
// CUPROF_TEARDOWN_TIMEOUT_MS overrides the default, from the environment or from
// the file named by $CUPROF_CONFIG, with the environment winning. An override is
// accepted only if it parses to >= 1000 and is then clamped to <= 600000, so a
// typo can neither disable the bound nor outlive the orchestrator's patience. A
// negative override is rejected before parsing: strtoull() accepts a leading '-'
// and wraps "-1" to ULLONG_MAX, which the clamp would have turned into 600000 -
// the longest bound allowed - so a sign typo would have looked like a request
// for a ten-minute teardown.
//
// config.cc has no CUDA/CUPTI dependency, so this builds and runs on any Linux
// box with a C++11 compiler - no GPU, no CUDA toolkit, no container:
//
//     make -C test/native && ./test/native/test_cuprof_config
//
// Every case starts from a clean slate via ResetEnv(). LoadConfig() re-reads the
// process environment and the cfg file on each call and caches nothing, so a
// CUPROF_* left set by an earlier case would silently change a later one - and
// the environment outranks the file, so the leak would not be obvious from the
// value alone. For the same reason each case pins $CUPROF_CONFIG at a temp file
// of this process: LoadCfgFile() returns early for a non-empty $CUPROF_CONFIG,
// which keeps /tmp/cuprof_<pid>.cfg and /tmp/cuprof.cfg - either of which
// another run on a shared host may have left behind - out of the picture.

// setenv/unsetenv below are POSIX, not ISO C++. libstdc++ defines _GNU_SOURCE
// when compiling C++, so glibc would expose them anyway; this is belt-and-braces
// for a libc that does not, and <features.h> reconciles it with _GNU_SOURCE.
#define _POSIX_C_SOURCE 200809L

#include <time.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Resolved via -I$(CUPROF)/src in the sibling Makefile.
#include "config.h"

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
    std::cout << (ok ? "  ok   " : "  FAIL ") << what << "\n";
    if (!ok) ++g_failures;
}

unsigned ResolvedTimeoutMs() { return cuprof::LoadConfig().teardown_timeout_ms; }

unsigned ResolvedDurationSec() { return cuprof::LoadConfig().duration_sec; }

// The resolved value goes into the message: a CI log has to say what
// LoadConfig() actually returned, not merely that it disagreed with the
// expectation.
void CheckTimeout(unsigned expected, const std::string& scenario) {
    const unsigned got = ResolvedTimeoutMs();
    Check(got == expected,
          scenario + " -> teardown_timeout_ms=" + std::to_string(got) + " (expected " +
              std::to_string(expected) + ")");
}

// Same contract for the window itself. CUPROF_DURATION no longer feeds
// teardown_timeout_ms, so precedence over it has to be asserted directly.
void CheckDuration(unsigned expected, const std::string& scenario) {
    const unsigned got = ResolvedDurationSec();
    Check(got == expected,
          scenario + " -> duration_sec=" + std::to_string(got) + " (expected " +
              std::to_string(expected) + ")");
}

void SetEnv(const char* key, const char* value) { setenv(key, value, 1); }

// CLOCK_REALTIME minus CLOCK_MONOTONIC, widened into the window that reading the
// two clocks itself spans: mono_lo <= monotonic at the realtime read <= mono_hi,
// so the true difference at that instant lies inside [lo_ns, hi_ns]. Reading
// CLOCK_MONOTONIC on both sides is what keeps a second boundary falling between
// the two reads from moving the window by a whole second.
struct ClockWindow {
    long long lo_ns;
    long long hi_ns;
};

long long ToNs(const struct timespec& ts) {
    return static_cast<long long>(ts.tv_sec) * 1000000000LL + static_cast<long long>(ts.tv_nsec);
}

ClockWindow SampleClockWindow() {
    struct timespec mono_lo;
    struct timespec rt;
    struct timespec mono_hi;
    clock_gettime(CLOCK_MONOTONIC, &mono_lo);
    clock_gettime(CLOCK_REALTIME, &rt);
    clock_gettime(CLOCK_MONOTONIC, &mono_hi);
    const ClockWindow w = {ToNs(rt) - ToNs(mono_hi), ToNs(rt) - ToNs(mono_lo)};
    return w;
}

// LoadConfig() samples (CLOCK_REALTIME - CLOCK_MONOTONIC) once, inside the call,
// so a window taken immediately before it and one taken immediately after it
// must between them contain the reported offset. kClockSlackNs absorbs the rate
// difference between the two clocks (NTP slews CLOCK_MONOTONIC by up to ~500
// ppm); at 1 ms it is twelve orders of magnitude below the offset itself, so a
// wrong sign, a unit error (ms or us instead of ns) and a never-computed 0 all
// still fail. This replaces a `!= 0` assertion, which any real clock satisfied
// and which therefore could not fail at all.
const long long kClockSlackNs = 1000000;

void CheckEpochOffset(uint64_t got, const ClockWindow& before, const ClockWindow& after) {
    const long long offset = static_cast<long long>(got);
    const long long lo = before.lo_ns - kClockSlackNs;
    const long long hi = after.hi_ns + kClockSlackNs;
    Check(offset >= lo && offset <= hi,
          "epoch_to_mono_offset_ns=" + std::to_string(offset) +
              " is inside the CLOCK_REALTIME - CLOCK_MONOTONIC window [" + std::to_string(lo) +
              ", " + std::to_string(hi) + "] sampled around LoadConfig()");
}

// Every key LoadConfig() reads from the environment. CUPROF_CONFIG is handled
// separately: it is pinned, never unset.
const char* const kCuprofKeys[] = {"CUPROF_OUTPUT",
                                   "CUPROF_DURATION",
                                   "CUPROF_VERBOSE",
                                   "CUPROF_SOCKET",
                                   "CUPROF_TEARDOWN_TIMEOUT_MS"};

std::vector<std::string> g_cfg_files;
std::string g_empty_cfg;

// getpid() alone is not a unique per-process key: a pid is reused once the
// process exits, so a /tmp/cuprof_config_test_<pid>_*.cfg left behind by an
// earlier run (or by this test being killed before its cleanup) would be read
// back as though this process had just written it. The nonce is derived from two
// clocks and the pid, needs no library beyond libc, and makes a stale path from
// a previous run unreachable.
unsigned long long MakeNonce() {
    struct timespec ts;
    unsigned long long n = 0;
    if (clock_gettime(CLOCK_REALTIME, &ts) == 0) {
        n = static_cast<unsigned long long>(ts.tv_sec) * 1000000000ULL +
            static_cast<unsigned long long>(ts.tv_nsec);
    }
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        n ^= static_cast<unsigned long long>(ts.tv_nsec) << 7;
    }
    return n ^ (static_cast<unsigned long long>(getpid()) * 2654435761ULL);
}

const unsigned long long g_nonce = MakeNonce();

// g_cfg_files.size() doubles as the per-process sequence number, so two calls
// with the same tag cannot collide either.
std::string MakeCfgPath(const std::string& tag) {
    std::ostringstream os;
    os << "/tmp/cuprof_config_test_" << getpid() << "_" << g_nonce << "_" << g_cfg_files.size()
       << "_" << tag << ".cfg";
    const std::string path = os.str();
    g_cfg_files.push_back(path);
    return path;
}

// Removes every temp cfg file this process created, on every exit path. main
// calls it explicitly; main also registers it with atexit so that a case which
// leaves main early still cannot strand a file in /tmp for the next run to read.
// atexit handlers registered from main run before the namespace-scope objects
// above are destroyed, so g_cfg_files is still alive here. Idempotent: clear()
// makes the second call a no-op, and std::remove() on an absent file just
// returns non-zero, which is ignored on purpose.
void RemoveCfgFiles() {
    for (size_t i = 0; i < g_cfg_files.size(); ++i) std::remove(g_cfg_files[i].c_str());
    g_cfg_files.clear();
}

// The cfg file format is KEY=VALUE per line, '#' starts a comment, and both
// sides are whitespace-trimmed.
//
// A failed write is a hard test failure, not a silent one. Without the check,
// an unwritable /tmp (or a full tmpfs) left no file behind, LoadConfig() fell
// back to whatever real /tmp/cuprof.cfg the host happened to have, and every
// assertion in the case then measured the wrong configuration while still
// passing. Both the write and the close are checked: a filesystem can defer
// ENOSPC until close(), and tellp() confirms the byte count that actually left
// the stream buffer.
void WriteCfg(const std::string& path, const std::string& body) {
    std::ofstream f(path.c_str(), std::ios::out | std::ios::trunc);
    f << body;
    const bool streamed = f.good();
    f.flush();
    const bool flushed = f.good();
    const long long written = static_cast<long long>(f.tellp());
    f.close();
    const bool closed = !f.fail();
    const long long expected = static_cast<long long>(body.size());
    Check(streamed && flushed && closed && written == expected,
          "wrote cfg file " + path + ": stream=" + (streamed ? "ok" : "bad") +
              " flush=" + (flushed ? "ok" : "bad") + " close=" + (closed ? "ok" : "bad") +
              " bytes=" + std::to_string(written) + "/" + std::to_string(expected));
}

void UseCfgFile(const std::string& path) { SetEnv("CUPROF_CONFIG", path.c_str()); }

// The documented starting point: no CUPROF_* override in the environment, and
// $CUPROF_CONFIG pointing at an empty file so no ambient cfg file can be read.
void ResetEnv() {
    for (size_t i = 0; i < sizeof(kCuprofKeys) / sizeof(kCuprofKeys[0]); ++i) {
        unsetenv(kCuprofKeys[i]);
    }
    UseCfgFile(g_empty_cfg);
}

// kTeardownDefaultMs in ms. Spelled once so a change of the default in config.cc
// shows up as a single edit here rather than as forty magic numbers.
const unsigned kDefaultMs = 30000;

// CUPROF_DURATION unset means duration_sec == 0. That is the case that matters
// most, because it is what `cuprof run` without a duration and every embedded
// injection that forgets the key actually hit: an unconfigured run still gets a
// bounded teardown, from the flat default rather than from any scaling.
void TestNoDurationGetsTheFlatDefault() {
    std::cout << "TestNoDurationGetsTheFlatDefault\n";
    ResetEnv();
    CheckDuration(0, "CUPROF_DURATION unset");
    CheckTimeout(kDefaultMs, "no CUPROF_DURATION, no override");
}

// Short windows. Every one of them used to be scaled to duration_sec * 500 and
// then floored; now they are simply the default. The values are kept so the
// coverage of the window range survives the removal of the scaling, and so that
// reintroducing a duration-dependent formula fails here rather than in the field.
void TestShortWindowsGetTheFlatDefault() {
    std::cout << "TestShortWindowsGetTheFlatDefault\n";
    ResetEnv();
    SetEnv("CUPROF_DURATION", "1");
    CheckTimeout(kDefaultMs, "CUPROF_DURATION=1 (flat default, not 500ms scaled)");
    ResetEnv();
    SetEnv("CUPROF_DURATION", "5");
    CheckTimeout(kDefaultMs, "CUPROF_DURATION=5 (flat default, not 2500ms scaled)");
    ResetEnv();
    SetEnv("CUPROF_DURATION", "59");
    CheckTimeout(kDefaultMs, "CUPROF_DURATION=59 (flat default, not 29500ms scaled)");
    ResetEnv();
    SetEnv("CUPROF_DURATION", "60");
    CheckTimeout(kDefaultMs, "CUPROF_DURATION=60 (flat default, no longer a boundary)");
}

// Windows past a minute, where the old formula was visibly duration-derived
// (30500 / 40000 / 44500). Nothing scales any more: the teardown drains only
// what CUPTI still holds at Stop(), and CUPTI has been handing its buffers over
// all along, so an 80s window leaves no more work for it than a 61s one.
void TestLongerWindowsGetTheFlatDefault() {
    std::cout << "TestLongerWindowsGetTheFlatDefault\n";
    ResetEnv();
    SetEnv("CUPROF_DURATION", "61");
    CheckTimeout(kDefaultMs, "CUPROF_DURATION=61 (flat default, not 30500ms scaled)");
    ResetEnv();
    SetEnv("CUPROF_DURATION", "80");
    CheckTimeout(kDefaultMs, "CUPROF_DURATION=80 (flat default, not 40000ms scaled)");
    ResetEnv();
    SetEnv("CUPROF_DURATION", "89");
    CheckTimeout(kDefaultMs, "CUPROF_DURATION=89 (flat default, not 44500ms scaled)");
}

// The longest windows covered here. 90s used to land exactly on the 45000
// ceiling and anything longer was capped there; both branches are gone, so 90,
// 255 and an hour all resolve to the same bound as a 1s window.
void TestLongestWindowsGetTheFlatDefault() {
    std::cout << "TestLongestWindowsGetTheFlatDefault\n";
    ResetEnv();
    SetEnv("CUPROF_DURATION", "90");
    CheckTimeout(kDefaultMs, "CUPROF_DURATION=90 (flat default, no ceiling any more)");
    ResetEnv();
    SetEnv("CUPROF_DURATION", "255");
    CheckTimeout(kDefaultMs, "CUPROF_DURATION=255 (flat default, not 127500ms scaled)");
    ResetEnv();
    SetEnv("CUPROF_DURATION", "3600");
    CheckTimeout(kDefaultMs, "CUPROF_DURATION=3600 (flat default, not 1800000ms scaled)");
}

// A malformed duration parses to 0. LoadConfig() never fails, so this is not an
// error path, and with the flat default the unparseable window cannot move the
// bound either way.
void TestMalformedDurationKeepsTheFlatDefault() {
    std::cout << "TestMalformedDurationKeepsTheFlatDefault\n";
    ResetEnv();
    SetEnv("CUPROF_DURATION", "abc");
    CheckDuration(0, "CUPROF_DURATION=abc (strtoul -> 0)");
    CheckTimeout(kDefaultMs, "CUPROF_DURATION=abc (flat default)");
}

// An explicit override inside [1000, 600000] is taken verbatim, and it wins
// over the flat default whatever the window is.
void TestEnvOverrideInsideTheClamps() {
    std::cout << "TestEnvOverrideInsideTheClamps\n";
    ResetEnv();
    SetEnv("CUPROF_TEARDOWN_TIMEOUT_MS", "1000");
    CheckTimeout(1000, "CUPROF_TEARDOWN_TIMEOUT_MS=1000 (the smallest accepted value)");
    ResetEnv();
    SetEnv("CUPROF_DURATION", "255");
    SetEnv("CUPROF_TEARDOWN_TIMEOUT_MS", "12345");
    CheckTimeout(12345, "override 12345 against CUPROF_DURATION=255 (default 30000)");
    ResetEnv();
    SetEnv("CUPROF_DURATION", "1");
    SetEnv("CUPROF_TEARDOWN_TIMEOUT_MS", "5000");
    CheckTimeout(5000, "override 5000 against CUPROF_DURATION=1 (default 30000)");
}

// The clamp above the maximum is unchanged by the flat default: an override
// alone still cannot push the bound past 600000, because the orchestrator gives
// up a fixed time after the window ends and a longer bound would turn a
// reportable failure into a silent watchdog kill.
void TestEnvOverrideAboveTheMaxIsClamped() {
    std::cout << "TestEnvOverrideAboveTheMaxIsClamped\n";
    ResetEnv();
    SetEnv("CUPROF_TEARDOWN_TIMEOUT_MS", "600000");
    CheckTimeout(600000, "CUPROF_TEARDOWN_TIMEOUT_MS=600000 (exactly the max)");
    ResetEnv();
    SetEnv("CUPROF_TEARDOWN_TIMEOUT_MS", "600001");
    CheckTimeout(600000, "CUPROF_TEARDOWN_TIMEOUT_MS=600001");
    ResetEnv();
    SetEnv("CUPROF_TEARDOWN_TIMEOUT_MS", "1000000000");
    CheckTimeout(600000, "CUPROF_TEARDOWN_TIMEOUT_MS=1000000000");
}

// Below 1000 the override is rejected outright and the default survives -
// including 0, which must not be able to disable the bound, and a malformed
// value, which strtoull parses to 0 and which therefore lands on the same path.
//
// The negative cases are here rather than in a block of their own because they
// are the same rejection seen from the other side of zero, and because they
// guard a real hazard: strtoull("-1") does not fail, it wraps to ULLONG_MAX, and
// the <= 600000 clamp would then have turned a sign typo into the longest bound
// the code allows. config.cc checks for a leading '-' before parsing, so a
// negative override must fall back to the default. Each one runs against a
// non-default CUPROF_DURATION so the case cannot pass by accident: if the sign
// check were dropped, the resolved value would be 600000 and not 30000.
// Values that are not a plain number are rejected rather than partially parsed,
// so a typo cannot buy a bound nobody asked for. strtoull would return 5000 for
// "5000abc" and ULLONG_MAX for a value that overflows, which the clamp would then
// turn into the longest bound allowed, so config.cc requires the whole value to
// have been consumed and rejects ERANGE; both fall back to the default. Leading
// and trailing whitespace and a leading plus are still accepted, and the
// whitespace skip has to cover a newline as well, because that is what strtoull
// itself skips and therefore what a sign check has to look past too.
void TestEnvOverrideThatIsNotANumberKeepsTheDefault() {
    std::cout << "TestEnvOverrideThatIsNotANumberKeepsTheDefault\n";
    ResetEnv();
    SetEnv("CUPROF_DURATION", "80");
    SetEnv("CUPROF_TEARDOWN_TIMEOUT_MS", "5000abc");
    CheckTimeout(kDefaultMs,
                 "override 5000abc against CUPROF_DURATION=80 (trailing junk rejected)");
    ResetEnv();
    SetEnv("CUPROF_DURATION", "80");
    SetEnv("CUPROF_TEARDOWN_TIMEOUT_MS", "0x1388");
    CheckTimeout(kDefaultMs,
                 "override 0x1388 against CUPROF_DURATION=80 (hexadecimal rejected, base 10)");
    ResetEnv();
    SetEnv("CUPROF_DURATION", "80");
    SetEnv("CUPROF_TEARDOWN_TIMEOUT_MS", "99999999999999999999999");
    CheckTimeout(kDefaultMs,
                 "override that overflows unsigned long long against CUPROF_DURATION=80 "
                 "(ERANGE rejected rather than clamped to 600000)");
    ResetEnv();
    SetEnv("CUPROF_DURATION", "80");
    SetEnv("CUPROF_TEARDOWN_TIMEOUT_MS", "\n-5000");
    CheckTimeout(kDefaultMs,
                 "negative override behind a newline against CUPROF_DURATION=80 "
                 "(rejected rather than clamped to 600000)");
    ResetEnv();
    SetEnv("CUPROF_TEARDOWN_TIMEOUT_MS", "  45000  ");
    CheckTimeout(45000, "override padded with whitespace is accepted");
    ResetEnv();
    SetEnv("CUPROF_TEARDOWN_TIMEOUT_MS", "+5000");
    CheckTimeout(5000, "override with a leading plus is accepted");
}

void TestEnvOverrideBelowTheMinKeepsTheDefault() {
    std::cout << "TestEnvOverrideBelowTheMinKeepsTheDefault\n";
    ResetEnv();
    SetEnv("CUPROF_DURATION", "80");
    SetEnv("CUPROF_TEARDOWN_TIMEOUT_MS", "999");
    CheckTimeout(kDefaultMs, "override 999 against CUPROF_DURATION=80 (default 30000)");
    ResetEnv();
    SetEnv("CUPROF_DURATION", "80");
    SetEnv("CUPROF_TEARDOWN_TIMEOUT_MS", "0");
    CheckTimeout(kDefaultMs, "override 0 against CUPROF_DURATION=80 (default 30000)");
    ResetEnv();
    SetEnv("CUPROF_DURATION", "80");
    SetEnv("CUPROF_TEARDOWN_TIMEOUT_MS", "abc");
    CheckTimeout(kDefaultMs, "override abc against CUPROF_DURATION=80 (default 30000)");
    ResetEnv();
    SetEnv("CUPROF_DURATION", "80");
    SetEnv("CUPROF_TEARDOWN_TIMEOUT_MS", "");
    CheckTimeout(kDefaultMs, "empty override against CUPROF_DURATION=80 (default 30000)");
    ResetEnv();
    SetEnv("CUPROF_TEARDOWN_TIMEOUT_MS", "0");
    CheckTimeout(kDefaultMs, "override 0 with no CUPROF_DURATION (flat default 30000)");
    ResetEnv();
    SetEnv("CUPROF_DURATION", "80");
    SetEnv("CUPROF_TEARDOWN_TIMEOUT_MS", "-1");
    CheckTimeout(kDefaultMs,
                 "override -1 against CUPROF_DURATION=80 (negative rejected, not the "
                 "600000 clamp)");
    ResetEnv();
    SetEnv("CUPROF_DURATION", "80");
    SetEnv("CUPROF_TEARDOWN_TIMEOUT_MS", "-60000");
    CheckTimeout(kDefaultMs,
                 "override -60000 against CUPROF_DURATION=80 (negative rejected, not the "
                 "600000 clamp)");
}

// The embedded path: an external tool injects libcuprof.so into a process whose
// environment it cannot change, so the override arrives through the file named
// by $CUPROF_CONFIG instead. The same acceptance rules apply to the file value,
// including the clamp.
void TestFileOverrideViaCuprofConfig() {
    std::cout << "TestFileOverrideViaCuprofConfig\n";

    ResetEnv();
    const std::string override_cfg = MakeCfgPath("file_override");
    WriteCfg(override_cfg, "# injected by the launcher\nCUPROF_TEARDOWN_TIMEOUT_MS=25000\n");
    UseCfgFile(override_cfg);
    CheckTimeout(25000, "$CUPROF_CONFIG file override 25000, nothing in the environment");

    ResetEnv();
    const std::string duration_cfg = MakeCfgPath("file_duration");
    WriteCfg(duration_cfg, "CUPROF_DURATION=80\n");
    UseCfgFile(duration_cfg);
    CheckTimeout(kDefaultMs,
                 "$CUPROF_CONFIG file CUPROF_DURATION=80, no override (flat default)");

    // The clamps apply to the file value exactly as they do to the environment.
    ResetEnv();
    const std::string high_cfg = MakeCfgPath("file_high");
    WriteCfg(high_cfg, "CUPROF_TEARDOWN_TIMEOUT_MS=999999\n");
    UseCfgFile(high_cfg);
    CheckTimeout(600000, "$CUPROF_CONFIG file override 999999");

    ResetEnv();
    const std::string low_cfg = MakeCfgPath("file_low");
    WriteCfg(low_cfg, "CUPROF_DURATION=80\nCUPROF_TEARDOWN_TIMEOUT_MS=999\n");
    UseCfgFile(low_cfg);
    CheckTimeout(kDefaultMs,
                 "$CUPROF_CONFIG file override 999 with file duration 80 (rejected, flat "
                 "default 30000)");

    // Whitespace trimming and a trailing comment on the same line.
    ResetEnv();
    const std::string padded_cfg = MakeCfgPath("file_padded");
    WriteCfg(padded_cfg, "  CUPROF_TEARDOWN_TIMEOUT_MS = 33000  # bounded teardown\n");
    UseCfgFile(padded_cfg);
    CheckTimeout(33000, "$CUPROF_CONFIG file override with padding and a comment");
}

// Priority between the two sources: the environment wins, and unsetting the
// environment lets the file through again.
void TestEnvBeatsFile() {
    std::cout << "TestEnvBeatsFile\n";
    ResetEnv();
    const std::string cfg = MakeCfgPath("env_beats_file");
    WriteCfg(cfg, "CUPROF_DURATION=255\nCUPROF_TEARDOWN_TIMEOUT_MS=20000\n");
    UseCfgFile(cfg);
    SetEnv("CUPROF_TEARDOWN_TIMEOUT_MS", "50000");
    CheckTimeout(50000, "env override 50000 vs file override 20000");

    unsetenv("CUPROF_TEARDOWN_TIMEOUT_MS");
    CheckTimeout(20000, "file override 20000 once the env override is unset");

    // The same precedence holds for CUPROF_DURATION, but it can no longer be
    // observed through teardown_timeout_ms: with the flat default every window
    // resolves to 30000, so the precedence is asserted on duration_sec and the
    // timeout is checked alongside to show that no window from either source
    // moves the bound.
    ResetEnv();
    const std::string duration_cfg = MakeCfgPath("env_beats_file_duration");
    WriteCfg(duration_cfg, "CUPROF_DURATION=255\n");
    UseCfgFile(duration_cfg);
    CheckDuration(255, "file CUPROF_DURATION=255 alone");
    CheckTimeout(kDefaultMs, "file CUPROF_DURATION=255 alone (flat default 30000)");
    SetEnv("CUPROF_DURATION", "10");
    CheckDuration(10, "env CUPROF_DURATION=10 vs file 255");
    CheckTimeout(kDefaultMs, "env CUPROF_DURATION=10 vs file 255 (flat default 30000)");
}

// The new key must not disturb the fields that were already resolved, and the
// existing precedence must still hold alongside it.
void TestOverrideLeavesTheOtherFieldsAlone() {
    std::cout << "TestOverrideLeavesTheOtherFieldsAlone\n";
    ResetEnv();
    const std::string cfg = MakeCfgPath("other_fields");
    WriteCfg(cfg, "CUPROF_OUTPUT=/tmp/aiprof-from-file.json\nCUPROF_SOCKET=/tmp/aiprof-from-file.sock\n");
    UseCfgFile(cfg);
    SetEnv("CUPROF_OUTPUT", "/tmp/aiprof-from-env.json");
    SetEnv("CUPROF_DURATION", "80");
    SetEnv("CUPROF_VERBOSE", "1");
    SetEnv("CUPROF_TEARDOWN_TIMEOUT_MS", "41000");

    const ClockWindow clocks_before = SampleClockWindow();
    const cuprof::Config c = cuprof::LoadConfig();
    const ClockWindow clocks_after = SampleClockWindow();
    Check(c.teardown_timeout_ms == 41000,
          "teardown_timeout_ms=" + std::to_string(c.teardown_timeout_ms) + " (expected 41000)");
    Check(c.duration_sec == 80,
          "duration_sec=" + std::to_string(c.duration_sec) + " (expected 80)");
    Check(c.output == "/tmp/aiprof-from-env.json",
          "output=" + c.output + " (expected the env value, not the file's)");
    Check(c.socket_path == "/tmp/aiprof-from-file.sock",
          "socket_path=" + c.socket_path + " (expected the file value: no env override)");
    Check(c.verbose, "verbose resolves true from CUPROF_VERBOSE=1");
    CheckEpochOffset(c.epoch_to_mono_offset_ns, clocks_before, clocks_after);
}

}  // namespace

int main() {
    // Registered before any file is created, so no exit path can strand one.
    std::atexit(RemoveCfgFiles);

    // Created once and pinned by every ResetEnv(), so no case can pick up a
    // /tmp/cuprof*.cfg belonging to another run.
    g_empty_cfg = MakeCfgPath("empty");
    WriteCfg(g_empty_cfg, "# intentionally empty\n");

    TestNoDurationGetsTheFlatDefault();
    TestShortWindowsGetTheFlatDefault();
    TestLongerWindowsGetTheFlatDefault();
    TestLongestWindowsGetTheFlatDefault();
    TestMalformedDurationKeepsTheFlatDefault();
    TestEnvOverrideInsideTheClamps();
    TestEnvOverrideAboveTheMaxIsClamped();
    TestEnvOverrideThatIsNotANumberKeepsTheDefault();
    TestEnvOverrideBelowTheMinKeepsTheDefault();
    TestFileOverrideViaCuprofConfig();
    TestEnvBeatsFile();
    TestOverrideLeavesTheOtherFieldsAlone();

    RemoveCfgFiles();

    std::cout << (g_failures == 0 ? "\nALL PASS\n" : "\nFAILURES: ")
              << (g_failures == 0 ? "" : std::to_string(g_failures)) << "\n";
    return g_failures == 0 ? 0 : 1;
}
