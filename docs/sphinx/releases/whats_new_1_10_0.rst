..
    Copyright (C) 2007-2023 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_1_10_0:

===========================
|hpx| V1.10.0 (TBD)
===========================

General changes
===============

Breaking changes
================

- The |cmake| configuration keys ``SOMELIB_ROOT`` (e.g., ``BOOST_ROOT``) have been
  renamed to ``Somelib_ROOT`` (e.g., ``Boost_ROOT``) to avoid warnings when using
  newer versions of |cmake|. Please update your scripts accordingly. For now, the
  old variable names are re-assigned to the new names and unset in the |cmake|
  cache.

Closed issues
=============

Closed pull requests
====================

