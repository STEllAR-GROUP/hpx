..
    Copyright (C) 2007-2024 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_1_11_0:

============================
|hpx| V1.11.0 (TBA)
============================

General changes
===============

- Added synchronous versions of all collective operations. Added global predefined
  communicator objects that are accessible through new APIs:
  ``hpx::collectives::get_world_communicator()`` refers to all localities and
  ``hpx::collectives::get_local_communicator()`` refers to all threads on the
  calling locality.

Breaking changes
================

- We have moved most of the APIs that were defined in the namespace
  ``hpx::parallel::execution`` to the namespace ``hpx::execution::experimental``.
  It was not possible to add compatibility facilities that will allow to continue
  using the old APIs, applications will have to be changed in order to
  continue functioning correctly.
- The CMake configuration parameter ``HPX_WITH_RUN_MAIN_EVERYWHERE`` is now
  deprecated and will be removed in the future. Use the preprocessor macro
  ``HPX_HAVE_RUN_MAIN_EVERYWHERE`` on a target-by-target case instead.

Closed issues
=============

Closed pull requests
====================

