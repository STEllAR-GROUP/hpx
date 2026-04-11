..
    Copyright (c) 2025 Jiakun Yan

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _using_the_lcw_parcelport:

========================
Using the LCW parcelport
========================

.. _info_lcw:

Basic information
=================

The LCW parcelport is an advanced MPI parcelport that can utilize the MPICH VCI
and Continuation extensions. The exact MPI implementation is wrapped in a thin
wrapper layer (the Lightweight Communication Wrapper, or LCW). This wrapper
provides an active message, send/recv, completion queue, and device abstraction
on top of MPI (with/without extensions), GASNet-EX, and LCI. The GASNet-EX
backend of LCW cannot be used in |hpx| for now, as it lacks send/recv support.

.. _`Lightweight Communication Wrapper`: https://github.com/JiakunYan/lcw

.. _build_lcw_pp:

Build |hpx| with the LCW parcelport
===================================

While building |hpx|, you can specify a set of |cmake|_ variables to enable
and configure the LCW parcelport. Below, there is a set of the most important
and frequently used CMake variables.

.. option:: HPX_WITH_PARCELPORT_LCW

   Enable the LCW parcelport. This enables the use of LCW for networking
   operations in the |hpx| runtime. The default value is ``OFF`` because it's
   not available on all systems and/or requires another dependency.
   You must set this variable to ``ON`` in order to use the LCW parcelport. All
   the following variables only make sense when this variable is set to ``ON``.

.. option:: HPX_WITH_FETCH_LCW

   Use FetchContent to fetch LCW. The default value is ``OFF``.
   If this option is set to ``OFF``. You need to install your own LCW library
   and |hpx| will try to find it using |cmake|_ ``find_package``. You can
   specify the location of the LCW installation by the environmental variable
   ``LCW_ROOT``. Refer to the `LCW README`_ for how to install LCW.
   If this option is set to ``ON``. |hpx| will fetch and build LCW for you. You
   can use the following |cmake|_ variables to configure this behavior for your
   platform.

.. _`LCW README`: https://github.com/JiakunYan/lcw#readme

.. option:: HPX_WITH_LCW_TAG

   This variable only takes effect when ``HPX_WITH_FETCH_LCW`` is set to ``ON``
   and ``FETCHCONTENT_SOURCE_DIR_LCW`` is not set.
   |hpx| will fetch LCW from its github repository. This variable controls the
   branch/tag LCW will be fetched.

.. option:: FETCHCONTENT_SOURCE_DIR_LCW

   This variable only takes effect when ``HPX_WITH_FETCH_LCW`` is set to ``ON``.
   When it is defined, ``HPX_WITH_LCW_TAG`` will be ignored.
   It accepts a path to a local version of LCW source code and |hpx| will fetch
   and build LCW from there.

.. _run_lcw_pp:

Run |hpx| with the LCW parcelport
=================================

You can configure runtime behavior with the following options.

.. option:: --hpx:ini=hpx.parcel.lcw.ndevices=<n>

   The number of LCW devices to use (default: 2). Each LCW device maps to an MPI
   communicator or an LCI device. When the MPICH VCI extension is enabled, each
   MPI communicator will be mapped to a dedicated collection of network resources.
   More devices lead to lower thread contention, but too many devices
   may lead to load imbalance or hardware overhead.

Reference
=========

Yan, Jiakun, Marc Snir, and Yanfei Guo. *Examining MPI and its Extensions for
Asynchronous Multithreaded Communication.* In *European MPI Users' Group Meeting*,
pp. 122-142. Cham: Springer Nature Switzerland, 2025.
