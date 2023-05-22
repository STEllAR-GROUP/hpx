..
    Copyright (C) 2007-2023 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_1_9_1:

===========================
|hpx| V1.9.1 (August 4, 2023)
===========================

General changes
===============

This point release fixes a couple of problems reported for the V1.9.0 release.
Most importantly, we fixed various occasional hanging during startup and shutdown
in distributed scenarios. We also added support for zero-copy serialization on
the receiving side to the TCP, MPI, and LCI parcelports. Last but not least, we
have added support for Visual Studio 2019 and gcc using MINGW on Windows, and
also support for gcc V13 and clang V15.

HPX headers are now made consistently named the same as their standard library
counterparts, e.g. `#include <thread>` now corresponds to `#include <hpx/thread.hpp>`.
This significantly simplifies porting existing standards conforming codes to HPX.

A lot of work has been done to improve and optimize our network communication
layers. Primary focus of this work was on the LCI parcelport, but we have also
cleaned up and improved the MPI parcelport.

Additionally, we have continued working on our documentation. The main focus
here was on completing the API documentation of the most important API functions.
We have started adding migration guides for people interested in moving their
codes away from other, commonplace parallelization frameworks like OpenMP.

Breaking changes
================

None

Closed issues
=============

* :hpx-issue:`6155` - hpxcxx and hpxrun.py do not work if HPX_WITH_TESTS=OFF
* :hpx-issue:`6164` - HPX_WITH_DATAPAR_BACKEND=EVE causes compile errors with C++17
* :hpx-issue:`6175` - Make sure all our parallel algorithms accept the predicates by value
* :hpx-issue:`6194` - tests.regressions.threads.threads_all_1422 failed at Perlmutter
* :hpx-issue:`6198` - set_intersection/set_difference fails when run with execution::par
* :hpx-issue:`6214` - Broken Links to the Documentation page in readme.rst
* :hpx-issue:`6217` - hpx::make_heap does not terminate when exPolicy is par (or par_unseq) and size of vector is 4
* :hpx-issue:`6246` - HPX fails to compile under cxx 20 (fresh system)
* :hpx-issue:`6247` - HPX 1.9.0 does not compile with GCC on Windows
* :hpx-issue:`6282` - The "attach-debugger" option is broken on the current master branch.

Closed pull requests
====================

* :hpx-pr:`6219` - Cleaning up #includes in hpx/ folder
* :hpx-pr:`6223` - Move documentation from README.rst to index.rst files under libs directory
* :hpx-pr:`6229` - Adding zero-copy support on the receiving end of the TCP and MPI parcel ports
* :hpx-pr:`6231` - Remove deprecated email from release procedure
* :hpx-pr:`6235` - Modernize more modules (levels 12-16)
* :hpx-pr:`6236` - Attempt to resolve occasional shutdown hangs in distributed operation
* :hpx-pr:`6239` - Fix Optimizing HPX applications page of Manual
* :hpx-pr:`6241` - LCI parcelport: Refactor, add more variants, zero copy receives.
* :hpx-pr:`6242` - updated deprecated headers
* :hpx-pr:`6243` - Adding github action builders using VS2019
* :hpx-pr:`6248` - Fix CUDA/HIP Jenkins pipelines
* :hpx-pr:`6250` - Resolve gcc problems on Windows
* :hpx-pr:`6251` - Attempting to fix problems in barrier causing hangs
* :hpx-pr:`6253` - Modernize set_thread_name on Windows
* :hpx-pr:`6256` - Fix nvcc/gcc-10 (Octo-Tiger) compilation issue
* :hpx-pr:`6257` - Cmake Tests: Delete operator check for size_t arg
* :hpx-pr:`6258` - Rewriting wait_some to circumvent data races causing hangs
* :hpx-pr:`6260` - Add migration guide to manual
* :hpx-pr:`6262` - Fixing wrong command line options in local command line handling
* :hpx-pr:`6266` - Attempt to resolve occasional hang in run_loop
* :hpx-pr:`6267` - Attempting to fix migration tests
* :hpx-pr:`6278` - Making sure the future's shared state doesn't go out of scope prematurely
* :hpx-pr:`6279` - Re-expose error names
* :hpx-pr:`6281` - Creating directory for file copy
* :hpx-pr:`6283` - Consistently #include unistd.h for _POSIX_VERSION
