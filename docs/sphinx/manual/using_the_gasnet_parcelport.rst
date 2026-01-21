..
    Copyright (c) 2023 Christopher Taylor

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _using_the_gasnet_parcelport:

===========================
Using the GASNet parcelport
===========================

.. _info_gasnet:

Basic information
=================

GASNet is a networking middleware software layer that provides support for
partitioned global address space (PGAS), remote memory access (RMA), and
active messaging (AM).

.. _`GASNet`: https://gasnet.lbl.gov

.. _build_gasnet_pp:

Build |hpx| with the GASNet parcelport
===================================

While building |hpx|, you can specify a set of |cmake| variables to enable
and configure the GASNet parcelport. Below, there is a set of the most important
and frequently used CMake variables.

.. option:: HPX_WITH_PARCELPORT_GASNET

   Enable the GASNet parcelport. This enables the use of GASNet for networking operations in the |hpx| runtime.
   The default value is ``OFF`` because it's not available on all systems and/or requires another dependency.
   You must set this variable to ``ON`` in order to use the GASNet parcelport. All the following variables only
   make sense when this variable is set to ``ON``.

.. option:: HPX_WITH_PARCELPORT_GASNET_CONDUIT 

   Defines which GASNet to utilize. The options are `smp;udp;mpi;ofi;ucx`. This feature tells cmake how to compile the
   parcelport against a specific implementation of GASNet. The `smp` option is for single-host/rank/PE communciations,
   `udp` enables GASNet udp support, `mpi` enables GASNet MPI support, `ofi` enables GASNet OpenFabrics libfabric support,
   and `ucx` enables GASNet UCX support. 

.. option:: HPX_WITH_FETCH_GASNET

   Use FetchContent to fetch GASNet. The default value is ``OFF``.
   If this option is set to ``OFF``. This feature tells |hpx| to fetch and build GASNet for you.
   The compiled GASNet will use the value provided in `HPX_WITH_PARCELPORT_GASNET_CONDUIT`. This
   feature requires the user to set `CMAKE_C_COMPILER`. GASNet downloaded will be installed into
   `CMAKE_INSTALL_PREFIX`.

.. _run_gasnet_pp:

Run |hpx| with the GASNet parcelport
=================================

We use the same mechanisms as MPI to launch GASNet, so you can use the same way you run MPI parcelport to run GASNet 
parcelport. Typically, it would be ``hpxrun``, ``amudprun``, or ``mpirun`` (for the MPI GASNet conduit).

If you are using ``hpxrun.py``, just pass ``--parcelport gasnet`` to the scripts.

If you are using ``amudprun`` or ``mpirun``, you can just pass
``--hpx:ini=hpx.parcel.gasnet.priority=1000``, ``--hpx:ini=hpx.parcel.gasnet.enable=1``, and
``--hpx:ini=hpx.parcel.bootstrap=gasnet`` to the |hpx| applications.
