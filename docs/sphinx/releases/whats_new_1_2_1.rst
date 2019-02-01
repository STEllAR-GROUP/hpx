..
    Copyright (C) 2007-2019 Hartmut Kaiser

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_1_2_1:

=========================
|hpx| V1.2.1 (unreleased)
=========================

General changes
===============

This is a bugfix release. It contains the following changes:

* Fix compilation on ARM.
* Fix compilation on 32-bit architectures.
* Fix compilation with Boost 1.69.0.

Closed issues
=============

* :hpx-issue:`3550` - 1>e:\000work\hpx\src\throw_exception.cpp(54): error C2440: '<function-style-cast>': cannot convert from 'boost::system::error_code' to 'hpx::exception'
* :hpx-issue:`3549` - HPX 1.2.0 does not build on i686, but release candidate did
* :hpx-issue:`3509` - Build on armv7l fails

Closed pull requests
====================

* :hpx-pr:`3542` - Fix numa lookup from pu when using hwloc 2.x
