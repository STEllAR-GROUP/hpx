..
    Copyright (C) 2022      Giannis Gonidelis
    Copyright (C) 2007-2020 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_1_8_0:

===========================
|hpx| V1.8.0 (Feb 15, 2022)
===========================

The new release comes with a renovated documentation! On top of that all
Hpx parallel algorithms have now been adapted in C++20. Much work has been
done towards implementing P2300 ("std::execution") and implementing the
underlying senders and receivers facilities.

General changes
===============

- The new documentation can now be found on our webpage: https://hpx-docs.stellar-group.org.
    This includes a completely new and user-friendly interface environment along with
    restructuring of certain components. The content in the "Quick start", "Manual" and
    "Examples" was improved, while the "Build system" page was adapted to include necessary
    information for newcommers.


- The following algorithms have been adapted to be C++20 conformant:

Breaking changes
================

Closed issues
=============

Closed pull requests
====================

* :hpx-pr:`5776` - Revert 1 deepaksuresh1411 for loop patch
* :hpx-pr:`5755` - Support for data-parallelism for mismatch algorithm
* :hpx-pr:`5730` - fix returning non default constructable objects from action
* :hpx-pr:`5708` - Use `/dev/shm` for Jenkins checkout on Piz Daint
* :hpx-pr:`5705` - Update and add tests for `transfer/schedule_from`
* :hpx-pr:`5703` - Fix `counting_iterator` container tests
* :hpx-pr:`5689` - Enable LTO in CI
* :hpx-pr:`5645` - Add matrix multiplication example
* :hpx-pr:`5625` - Revert "Use advance_and_get_distance where required"
* :hpx-pr:`5578` - Remove distributed functionality