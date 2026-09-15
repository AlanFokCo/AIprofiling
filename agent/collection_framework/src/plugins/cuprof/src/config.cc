// AIProf-local modification (Apache-2.0 4(b)): this file differs from
// upstream cuprof. See VENDOR.md, patches/0002-align-cupti-epoch-to-clock-monotonic.patch
// and patches/0005-bound-the-cupti-teardown-in-stop.patch.
#include "config.h"

#include <time.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

namespace cuprof {
namespace {

std::string DefaultOutput() {
    std::ostringstream os;
    os << "cuprof_" << getpid() << ".json";
    return os.str();
}

// KEY=VALUE per line; '#' starts a comment; whitespace-trimmed.
std::map<std::string, std::string> ReadCfgFile(const std::string& path) {
    std::map<std::string, std::string> kv;
    std::ifstream f(path.c_str());
    std::string line;
    while (std::getline(f, line)) {
        size_t hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = line.substr(0, eq);
        std::string v = line.substr(eq + 1);
        k.erase(0, k.find_first_not_of(" \t"));
        k.erase(k.find_last_not_of(" \t\r") + 1);
        v.erase(0, v.find_first_not_of(" \t"));
        v.erase(v.find_last_not_of(" \t\r") + 1);
        if (!k.empty()) kv[k] = v;
    }
    return kv;
}

std::map<std::string, std::string> LoadCfgFile() {
    const char* p = getenv("CUPROF_CONFIG");
    if (p && *p) return ReadCfgFile(p);

    std::ostringstream os;
    os << "/tmp/cuprof_" << getpid() << ".cfg";
    std::map<std::string, std::string> kv = ReadCfgFile(os.str());
    // An external tool may not know the in-namespace pid; a pid-less path
    // serves the one-process-per-container case.
    if (kv.empty()) kv = ReadCfgFile("/tmp/cuprof.cfg");
    return kv;
}

// Default and clamps for the bound on the CUPTI teardown in CuptiSink::Stop().
//
// The default is a wedge detector, not an allowance for slow teardowns. The
// heaviest one measured on a 4x A10 host with driver 580.126.09 - a 20s window
// that collected 4475153 events into a 1.24 GB trace - took 82 ms of
// cuptiActivityDisable plus cuptiActivityFlushAll, so 30s is about 370x the
// worst healthy case observed. It is flat rather than derived from
// CUPROF_DURATION because the teardown drains whatever CUPTI is still holding at
// Stop(), and buffers are delivered continuously while the window runs, so that
// residual does not grow with the window length.
//
// What does grow with the window is the WriteChromeTrace that follows, which took
// 7.8s for that same trace and is not covered by this bound.
//
// The clamps reject nonsense rather than encode anybody's budget. Zero would
// disable the bound, and the ceiling keeps an absurd magnitude from wrapping.
// Nothing here can know what an orchestrator will wait for - CollectionFramework
// gives up on a window duration + 60s after it started - so raising this key past
// that budget is the embedder's call, and doing it turns a reportable failure
// back into a watchdog kill.
const unsigned kTeardownDefaultMs = 30000;
const unsigned kTeardownMinMs = 1000;
const unsigned kTeardownMaxMs = 600000;

}  // namespace

Config LoadConfig() {
    std::map<std::string, std::string> kv = LoadCfgFile();

    // Environment wins over the file; the file covers processes whose
    // environment the launcher could not touch.
    const char* keys[] = {"CUPROF_OUTPUT",      "CUPROF_DURATION",
                          "CUPROF_VERBOSE",     "CUPROF_SOCKET",
                          "CUPROF_TEARDOWN_TIMEOUT_MS"};
    for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); ++i) {
        const char* v = getenv(keys[i]);
        if (v && *v) kv[keys[i]] = v;
    }

    Config c;
    c.output = kv.count("CUPROF_OUTPUT") ? kv["CUPROF_OUTPUT"] : DefaultOutput();
    c.duration_sec = kv.count("CUPROF_DURATION")
                         ? static_cast<unsigned>(strtoul(kv["CUPROF_DURATION"].c_str(), NULL, 10))
                         : 0;
    c.teardown_timeout_ms = kTeardownDefaultMs;
    if (kv.count("CUPROF_TEARDOWN_TIMEOUT_MS")) {
        // Anything that does not parse cleanly to at least kTeardownMinMs keeps
        // the default above.
        const char* text = kv["CUPROF_TEARDOWN_TIMEOUT_MS"].c_str();
        // strtoull skips leading whitespace on its own, newlines included, and
        // then accepts a '-' by wrapping it into a huge unsigned value that the
        // clamp below would turn into the longest bound allowed. A negative
        // timeout is a typo rather than a request for that, so it is rejected
        // before parsing rather than left to the clamp.
        const char* first = text + strspn(text, " \t\n\v\f\r");
        if (*first != '-') {
            char* end = NULL;
            errno = 0;
            const unsigned long long v = strtoull(first, &end, 10);
            // The whole value has to have been consumed, so "5000abc" is a typo
            // rather than 5000. A value too large to parse at all falls back to
            // the default instead of arriving at the ceiling by saturation; one
            // that parses and merely exceeds the ceiling is clamped to it.
            const bool consumed = end != first && strspn(end, " \t\n\v\f\r") == strlen(end);
            if (consumed && errno != ERANGE && v >= kTeardownMinMs) {
                c.teardown_timeout_ms =
                    v > kTeardownMaxMs ? kTeardownMaxMs : static_cast<unsigned>(v);
            }
        }
    }
    c.verbose = kv.count("CUPROF_VERBOSE") && strtol(kv["CUPROF_VERBOSE"].c_str(), NULL, 10) != 0;
    c.socket_path = kv.count("CUPROF_SOCKET") ? kv["CUPROF_SOCKET"] : "";

    // CUPTI timestamps are epoch (CLOCK_REALTIME) nanoseconds. PyTorch
    // profiler (pyki) uses CLOCK_MONOTONIC. To align them on the same
    // timeline in Perfetto/chrome-trace viewers, subtract the difference.
    struct timespec rt, mt;
    clock_gettime(CLOCK_REALTIME, &rt);
    clock_gettime(CLOCK_MONOTONIC, &mt);
    uint64_t rt_ns = static_cast<uint64_t>(rt.tv_sec) * 1000000000ULL + rt.tv_nsec;
    uint64_t mt_ns = static_cast<uint64_t>(mt.tv_sec) * 1000000000ULL + mt.tv_nsec;
    c.epoch_to_mono_offset_ns = rt_ns - mt_ns;

    return c;
}

}  // namespace cuprof
