..
    Copyright (C) 2007-2018 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_0_8_1:

===========================
|hpx| V0.8.1 (Apr 21, 2012)
===========================

This is a point release including important bug fixes for :ref:`hpx_0_8_0`.

General changes
===============

* |hpx| does not need to be installed anymore to be functional.

Bug fixes (closed tickets)
==========================

Here is a list of the important tickets we closed for this point release:

* :hpx-issue:`295` - Don't require install path to be known at compile time.
* :hpx-issue:`371` - Add hpx iostreams to standard build.
* :hpx-issue:`384` - Fix compilation with GCC 4.7.
* :hpx-issue:`390` - Remove keep_factory_alive startup call from ShenEOS; add
  shutdown call to H5close.
* :hpx-issue:`393` - Thread affinity control is broken.

Bug fixes (commits)
===================

Here is a list of the important commits included in this point release:

* r7642 - External: Fix backtrace memory violation.
* r7775 - Components: Fix symbol visibility bug with component startup
          providers. This prevents one components providers from overriding
          another components.
* r7778 - Components: Fix startup/shutdown provider shadowing issues.

