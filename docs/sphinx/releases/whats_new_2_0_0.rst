..
    Copyright (C) 2007-2025 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_2_0_0:

============================
|hpx| V2.0.0 (TBD)
============================

General changes
===============

- The ITTNotify backend now emits task lifecycle and work-stealing events when
  ITTNotify is enabled at runtime (``--hpx:ini=hpx.use_itt_notify=1``). This is
  ``ON`` by default for ``HPX_WITH_ITTNOTIFY=ON`` builds; pass
  ``-DHPX_WITH_TRACING_LIFECYCLE_EVENTS=OFF`` and
  ``-DHPX_WITH_TRACING_WORK_STEALING_EVENTS=OFF`` to the |cmake|_ configuration
  to get the earlier behavior.

Breaking changes
================

Closed issues
=============

Closed pull requests
====================

