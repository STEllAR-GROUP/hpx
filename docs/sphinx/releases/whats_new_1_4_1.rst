..
    Copyright (C) 2007-2020 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_1_4_1:

===========================
|hpx| V1.4.1 (Feb 12, 2020)
===========================

General changes
===============

This is a bugfix release. It contains the following changes:

* Fix compilation issues on Windows, macOS, FreeBSD, and with gcc 10
* Install missing ``pdb`` files on Windows
* Allow running tests using an installed version of |hpx|
* Skip MPI finalization if HPX has not initialized MPI
* Give a hard error when attempting to use IO counters on Windows

Closed issues
=============

* :hpx-issue:`4320` - HPX 1.4.0 does not compile with gcc 10
* :hpx-issue:`4336` - Building HPX 1.4.0 with IO Counters breaks (Windows)
* :hpx-issue:`4334` - HPX ``Debug`` and ``RelWithDebinfo`` builds on Windows not
  installing ``.pdb`` files
* :hpx-issue:`4322` - Undefine VT1 and VT2 after boost includes
* :hpx-issue:`4314` - Compile error on 1.4.0
* :hpx-issue:`4307` - ``ld: error: duplicate symbol: freebsd_environ``


Closed pull requests
====================

* :hpx-pr:`4376` - Attempt to fix some test build errors on Windows
* :hpx-pr:`4357` - Adding missing ``#include``\ s to fix gcc V10 linker problems
* :hpx-pr:`4353` - Skip ``MPI_Finalize`` if ``MPI_Init`` is not called from HPX
* :hpx-pr:`4343` - Give a hard error if IO counters are enabled on non-Linux
  systems
* :hpx-pr:`4337` - Installing pdb files on Windows
* :hpx-pr:`4335` - Adding capability to buildsystem to use an installed version
  of HPX
* :hpx-pr:`4315` - Forcing exported symbols from composable_guard to be linked
  into core library
* :hpx-pr:`4310` - Remove environment handling from ``exception.cpp``
