..
    Copyright (C)      2013 Thomas Heller

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

=============
Testing |hpx|
=============

To ensure correctness of |hpx|, we ship a large variety of unit and regression
tests. The tests are driven by the |ctest|_ tool and are executed automatically 
on each commit to the |hpx| |hpx_github|_ repository. In addition, it is encouraged 
to run the test suite manually to ensure proper operation on your target system. 
If a test fails for your platform, we highly recommend submitting an issue on our 
|hpx_github_issues|_ tracker with detailed information about the target system.

Running tests manually
======================

Running the tests manually is as easy as typing ``make tests && make test``.
This will build all tests and run them once the tests are built successfully.
After the tests have been built, you can invoke separate tests with the help of
the ``ctest`` command. You can list all available test targets using ``make help
| grep tests``. Please see the |ctest_doc|_ for further details.

Running performance tests
=========================

For each pull request, Jenkins runs the performance tests on Rostam (see
``.jenkins/lsu-perftests``) and posts a comparison with the reference results
on the pull request. To run a performance test yourself, configure |hpx| with
``-DHPX_WITH_TESTS_RUN_BENCHMARKS=ON`` (and ``-DHPX_WITH_NANOBENCH=ON``, as the
CI does) and build the target ``<name>_perftest``, e.g.
``benchmark_merge_perftest``. It runs the benchmark 15 times (or as often as
the ``N_EXECUTIONS`` argument of ``add_hpx_performance_report_test`` says) and
writes the results to ``tools/perftests_ci/results/<name>.json`` in the build
directory. Then build the target ``perftests_plot``, which compares all results
found there with the reference results, and writes the measurements and their
plots to ``tools/perftests_ci/index.html``.

Adding new performance tests
============================

To add a new performance test, wrap the portion of code to benchmark with
``hpx::util::perftests_report``, passing the test name, the executor name, the
number of repetitions and the function to time (can be a lambda). Add the
command line options of this facility with ``hpx::util::perftests_cfg`` in
``main``, call ``hpx::util::perftests_init(vm, "<name>")`` at the beginning of
``hpx_main``, using the name of the test, and call
``hpx::util::perftests_print_times`` at the end to print the results in the
json format used for the comparison. See ``future_overhead_report.cpp`` for an
example.

To add the test to the CI report, register it with
``add_hpx_performance_report_test`` if ``HPX_WITH_TESTS_RUN_BENCHMARKS`` is
enabled (see ``libs/core/algorithms/tests/performance/CMakeLists.txt``); the CI
builds and runs all registered performance tests.
Finally, add a reference result. The timings differ between the nodes the CI
may run on, so each node has its own reference results in
``tools/perftests_ci/perftest/references/<node>.tar.gz``, and the comparison
uses the ones of the node the benchmarks ran on. Build the ``<name>_perftest``
target on that node, copy ``tools/perftests_ci/results/<name>.json`` from the
build directory to the unpacked reference results of the node, and pack them
again with ``tools/perftests_ci/pack_references.sh`` (see the comment at the top
of the script). If the CI runs on a node without references for all
benchmarks, it uploads the node's references together with the results of the
run, packed, next to the plots of the report, and the report links to them.
The CI comments on a pull request only about measurements that got more than
5% faster or slower; the pull request comment ``retest lsu perf full`` starts a
run that reports all measurements.
Every CI run also keeps all of its results packed as references of its node
(``perftests-reports/references/<node>.tar.gz`` in the Jenkins artifacts).

Issue tracker
=============

If you stumble over a bug or missing feature in |hpx|, please
submit an issue to our |hpx_github_issues|_ page. For more information on how to
submit support requests or other means of getting in contact with the developers,
please see the |support|_ page.

Continuous testing
==================

In addition to manual testing, we run automated tests on various platforms. We also 
run tests on all pull requests using both |github|_. You can see the dashboards
|hpx_cdash|_ .
