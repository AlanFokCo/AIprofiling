# Native (GPU-free) regression tests for the vendored cuprof patch series.
#
# Three cuprof sources are compiled in, trace_writer.cc, bounded_call.cc and
# config.cc, because none of them has a CUDA/CUPTI dependency: that lets three of
# AIProf's local cuprof patches be verified on any Linux host with a C++11
# compiler and pthreads. The rest of cuprof needs a GPU.
#
#   trace_writer.cc  two patches touch this one file, and the test covers both:
#                    patches/0002-align-cupti-epoch-to-clock-monotonic.patch
#                    shifts each event timestamp from CUPTI's CLOCK_REALTIME
#                    epoch onto CLOCK_MONOTONIC by subtracting the offset
#                    LoadConfig() computed, and
#                    patches/0004-duration-from-raw-endpoint-pair.patch takes
#                    each event's duration from the raw, unshifted endpoint pair
#                    so an event that straddles the offset, or whose end_ns was
#                    never set, cannot underflow uint64.
#   bounded_call.cc  patches/0005-bound-the-cupti-teardown-in-stop.patch: the
#                    deadline primitive CuptiSink::Stop() runs the CUPTI teardown
#                    under instead of calling it inline.
#   config.cc        the same patch 0005: the teardown-timeout resolution
#                    LoadConfig() feeds that bound.
#
# Each source gets its own test binary, and `make` builds and runs all three,
# aggregating their exit codes (see the `run` rule below) so a failure in one
# binary cannot stop the other two from running or hide their result.
#
#   make            # build + run all three
#   make clean

CXX ?= g++
CXXFLAGS ?= -O2 -g -std=c++11 -Wall -Wextra

CUPROF := ../../agent/collection_framework/src/plugins/cuprof
BIN := test_cuprof_trace_writer
BIN_BOUNDED := test_cuprof_bounded_call
BIN_CONFIG := test_cuprof_config

.PHONY: all run clean
all: run

$(BIN): test_cuprof_trace_writer.cc $(CUPROF)/src/trace_writer.cc $(CUPROF)/src/trace_writer.h
	$(CXX) $(CXXFLAGS) -I$(CUPROF)/src test_cuprof_trace_writer.cc $(CUPROF)/src/trace_writer.cc -o $@

# -pthread, not just -lpthread: bounded_call.cc creates threads, and the test
# compares pthread_self() across them.
$(BIN_BOUNDED): test_cuprof_bounded_call.cc $(CUPROF)/src/bounded_call.cc $(CUPROF)/src/bounded_call.h
	$(CXX) $(CXXFLAGS) -pthread -I$(CUPROF)/src test_cuprof_bounded_call.cc $(CUPROF)/src/bounded_call.cc -o $@

# No -pthread here: config.cc reads the environment, one cfg file and two
# clocks, and creates no threads.
$(BIN_CONFIG): test_cuprof_config.cc $(CUPROF)/src/config.cc $(CUPROF)/src/config.h
	$(CXX) $(CXXFLAGS) -I$(CUPROF)/src test_cuprof_config.cc $(CUPROF)/src/config.cc -o $@

# All three binaries always run, and their exit codes are aggregated. Separate
# recipe lines (or `&&`) would let a failure in the older trace-writer test abort
# the recipe before the later tests ever start, masking a regression in patch
# 0005 behind an unrelated failure. POSIX sh only: both `make test-native` and
# the CI job run this recipe under /bin/sh. The `for` loop is what preserves that
# as binaries are added: a new one joins the list twice, in the prerequisites
# and in the loop, and still runs after an earlier one has failed.
run: $(BIN) $(BIN_BOUNDED) $(BIN_CONFIG)
	@status=0; \
	for t in $(BIN) $(BIN_BOUNDED) $(BIN_CONFIG); do \
		echo "==== $$t ===="; \
		"./$$t"; rc=$$?; \
		echo "==== $$t exit=$$rc ===="; \
		if [ $$status -eq 0 ] && [ $$rc -ne 0 ]; then status=$$rc; fi; \
	done; \
	if [ $$status -ne 0 ]; then echo "test-native: FAILED (exit $$status)"; \
	else echo "test-native: PASS"; fi; \
	exit $$status

clean:
	rm -f $(BIN) $(BIN_BOUNDED) $(BIN_CONFIG)
