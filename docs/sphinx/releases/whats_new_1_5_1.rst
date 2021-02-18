..
    Copyright (C) 2020 ETH Zurich

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_1_5_1:

===========================
|hpx| V1.5.1 (Sep 30, 2020)
===========================

General changes
===============

This is a patch release. It contains the following changes:

* Remove restriction on suspending runtime with multiple localities, users are
  now responsible for synchronizing work between localities before suspending.
* Fixes several compilation problems and warnings.
* Adds notes in the documentation explaining how to cite HPX.

Closed issues
=============

* :hpx-issue:`4971` - Parallel sort fails to compile with C++20
* :hpx-issue:`4950` - Build with `HPX_WITH_PARCELPORT_ACTION_COUNTERS` `ON` fails
* :hpx-issue:`4940` - Codespell report for "HPX" (on fossies.org)
* :hpx-issue:`4937` - Allow suspension of runtime for multiple localities

Closed pull requests
====================

* :hpx-pr:`4982` - Add page about citing HPX to documentation
* :hpx-pr:`4981` - Adding the missing include
* :hpx-pr:`4974` - Remove leftover format export hack
* :hpx-pr:`4972` - Removing use of get_temporary_buffer and return_temporary_buffer
* :hpx-pr:`4963` - Renaming files to avoid warnings from the vs build system
* :hpx-pr:`4951` - Fixing build if HPX_WITH_PARCELPORT_ACTION_COUNTERS=On
* :hpx-pr:`4946` - Allow suspension on multiple localities
* :hpx-pr:`4944` - Fix typos reported by fossies codespell report
* :hpx-pr:`4941` - Adding some explanation to README about how to cite HPX
* :hpx-pr:`4939` - Small changes
