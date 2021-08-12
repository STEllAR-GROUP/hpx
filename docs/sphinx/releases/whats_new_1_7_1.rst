..
    Copyright (C) 2020-2021 ETH Zurich
    Copyright (C) 2007-2020 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_1_7_1:

===========================
|hpx| V1.7.1 (Aug 12, 2021)
===========================

This is a bugfix release with a few minor fixes.

General changes
===============

- Added a CMake option to assume that all types are bitwise serializable by
  default: ``HPX_SERIALIZATION_WITH_ALL_TYPES_ARE_BITWISE_SERIALIZABLE``. The
  default value ``OFF`` corresponds to the old behaviour.
- Added a version check for Asio. The minimum Asio version supported by |hpx| is
  1.12.0.
- Fixed a bug affecting usage of actions, where the internals of |hpx| relied on
  function addresses being unique. This was fixed by relying on variable
  addresses being unique instead.
- Made ``hpx::util::bind`` more strict in checking the validity of placeholders.
- Small performance improvement to spinlocks.
- Adapted the following parallel algorithms to C++20: ``inclusive_scan``,
  ``exclusive_scan``, ``transform_inclusive_scan``,
  ``transform_exclusive_scan``.

Breaking changes
================

- The experimental ``hpx::execution::simdpar`` execution policy (introduced in
  1.7.0) was renamed to ``hpx::execution::par_simd`` for consistency with the
  other parallel policies.

Closed issues
=============

* :hpx-issue:`5494` - Rename `simdpar` execution policy to `par_simd`
* :hpx-issue:`5488` - `hpx::util::bind` doesn't bounds-check placeholders
* :hpx-issue:`5486` - Possible V1.7.1 release

Closed pull requests
====================

* :hpx-pr:`5500` - Minor bug fix in transform exclusive and inclusive scan tests
* :hpx-pr:`5499` - Rename simdpar to par_simd
* :hpx-pr:`5489` - Adding bound-checking for bind placeholders
* :hpx-pr:`5485` - Add Asio version check
* :hpx-pr:`5482` - Change extra archive data to rely on uniqueness of a variable address, not a function address
* :hpx-pr:`5448` - More fixes to enable for all types to be assumed to be bitwise copyable
* :hpx-pr:`5445` - Improve performance of Spinlocks
* :hpx-pr:`5444` - Adapt transform_inclusive_scan to C++ 20
* :hpx-pr:`5440` - Adapt transform_exclusive_scan to C++ 20
* :hpx-pr:`5439` - Adapt inclusive_scan to C++ 20
* :hpx-pr:`5436` - Adapt exclusive_scan to C++20
