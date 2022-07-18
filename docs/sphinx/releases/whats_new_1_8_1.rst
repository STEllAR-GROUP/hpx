..
    Copyright (C) 2007-2022 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_1_8_1:

===========================
|hpx| V1.8.1 (Jul 18, 2022)
===========================

This is a bugfix release with a few minor additions and resolved problems.

General changes
===============

This patch release add a number of small new features and fixes a handful of
problems discovered since the last release, in particular:

- A lot of work has been done to improve vectorization support for our parallel
  algorithms. HPX now supports using |eve| as a vectorization backend.
- Added a simple average power consumption performance counter.
- Added performance counters related to the use of zero-copy chunks in the
  networking layer.
- More work was done towards full compatibility with the sender/receivers
  proposal |p2300|.
- Fixed collective operations to properly avoid overalapping consecutive
  operations on the same communicator.
- Simplified the implementation of our execution policies and added mapping
  functions between those.
- Fixed performance issues with our implementation of `small_vector`.
- Serialization now works with buffers of unsigned characters.
- Fixing dangling reference in serialization of non-default constructible types
- Fixed static linking on Windows.
- Fixed support for M1/MacOS based architectures.
- Fixed support for gentoo/musl.
- Fixed `hpx::counting_semaphore_var`.
- Properly check start and end bounds for `hpx::for_loop`
- A lot of changes and fixes to the documentation (see
  https://hpx-docs.stellar-group.org).

Breaking changes
================

- No breaking changes have been introduced.

Closed issues
=============


Closed pull requests
====================

