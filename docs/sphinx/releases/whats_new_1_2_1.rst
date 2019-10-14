..
    Copyright (C) 2007-2019 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_1_2_1:

===========================
|hpx| V1.2.1 (Feb 19, 2019)
===========================

General changes
===============

This is a bugfix release. It contains the following changes:

* Fix compilation on ARM, s390x and 32-bit architectures.
* Fix a critical bug in the ``future`` implementation.
* Fix several problems in the CMake configuration which affects external
  projects.
* Add support for Boost 1.69.0.

Closed issues
=============

* :hpx-issue:`3638` - Build HPX 1.2 with boost 1.69
* :hpx-issue:`3635` - Non-deterministic crashing on Stampede2
* :hpx-issue:`3550` - 1>e:\000work\hpx\src\throw_exception.cpp(54): error C2440:
  '<function-style-cast>': cannot convert from 'boost::system::error_code' to
  'hpx::exception'
* :hpx-issue:`3549` - HPX 1.2.0 does not build on i686, but release candidate
  did
* :hpx-issue:`3511` - Build on s390x fails
* :hpx-issue:`3509` - Build on armv7l fails

Closed pull requests
====================

* :hpx-pr:`3695` - Don't install CMake templates and packaging files
* :hpx-pr:`3666` - Fixing yet another race in future_data
* :hpx-pr:`3663` - Fixing race between setting and getting the value inside
  future_data
* :hpx-pr:`3648` - Adding timestamp option for S390x platform
* :hpx-pr:`3647` - Blind attempt to fix warnings issued by gcc V9
* :hpx-pr:`3611` - Include GNUInstallDirs earlier to have it available for
  subdirectories
* :hpx-pr:`3595` - Use GNUInstallDirs lib path in pkgconfig config file
* :hpx-pr:`3593` - Add include(GNUInstallDirs) to HPXMacros.cmake
* :hpx-pr:`3591` - Fix compilation error on arm7 architecture. Compiles and runs
  on Fedora 29 on Pi 3.
* :hpx-pr:`3558` - Adding constructor `exception(boost::system::error_code
  const&)`
* :hpx-pr:`3555` - cmake: make install locations configurable
* :hpx-pr:`3551` - Fix uint64_t causing compilation fail on i686
