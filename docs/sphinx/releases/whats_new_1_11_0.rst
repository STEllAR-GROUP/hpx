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

Breaking changes
================

- We have moved most of the APIs that were defined in the namespace
  ``hpx::parallel::execution`` to the namespace ``hpx::execution::experimental``.
  It was not possible to add compatibility facilities that will allow to continue
  using the old APIs, applications will have to be changed in order to
  continue functioning correctly.

Closed issues
=============

Closed pull requests
====================

