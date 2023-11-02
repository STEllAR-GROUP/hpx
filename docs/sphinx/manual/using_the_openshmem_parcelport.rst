..
    Copyright (c) 2023 Christopher Taylor

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _using_the_openshmem_parcelport:

========================
Using the OpenSHMEM parcelport
========================

.. _info_openshmem:

Basic information
=================

OpenSHMEM is an effort to create an API specification standardizing parallel
programming in the Partitioned Global Address Space model. The specification
effort also creates a reference implementation of the API. This implementation
aims to be portable, allowing it to be deployed in multiple environments, and
to be a starting point for implementations targeted to particular hardware
platforms. The reference implementation will serves as a springboard for future
development of the API.

OpenSHMEM provides one-sided communications, atomics, collectives, and again
implements a partitioned global address space (PGAS) distributed memory
computing model. The parcelport uses put_signal (put and atomic_put), wait_until,
and get from the OpenSHMEM API.

.. _`OpenSHMEM`: http://www.openshmem.org/site/ 

.. _build_openshmem_pp:

Build |hpx| with the OpenSHMEM parcelport
===================================

While building |hpx|, you can specify a set of |cmake| variables to enable
and configure the OpenSHMEM parcelport. Below, there is a set of the most important
and frequently used CMake variables.

.. option:: HPX_WITH_PARCELPORT_OPENSHMEM

   Enable the OpenSHMEM parcelport. This enables the use of OpenSHMEM for networking operations in the |hpx| runtime.
   The default value is ``OFF`` because it's not available on all systems and/or requires another dependency.
   You must set this variable to ``ON`` in order to use the OpenSHMEM parcelport. All the following variables only
   make sense when this variable is set to ``ON``.

.. option:: HPX_WITH_PARCELPORT_OPENSHMEM

   Defines which OpenSHMEM to utilize. The options are `sos;ucx;mpi`. This feature tells cmake how to compile the
   parcelport against a specific implementation of OpenSHMEM.

.. option:: HPX_WITH_FETCH_OPENSHMEM

   Use FetchContent to fetch OpenSHMEM. The default value is ``OFF``.
   If this option is set to ``OFF``. This feature tells |hpx| to fetch and build OpenSHMEM for you.
   |hpx| will download either `ucx` or `sos` OpenSHMEM based on the value provided in
   `HPX_WITH_PARCELPORT_OPENSHMEM`. This feature requires the user to set `CMAKE_C_COMPILER`. The
   OpenSHMEM downloaded will be installed into `CMAKE_INSTALL_PREFIX`. PMI support will be compiled
   into the parcelport if it's available. If PMI is not available a barebones implementation for SMP
   systems will be used.

.. _run_openshmem_pp:

Run |hpx| with the OpenSHMEM parcelport
=================================

We use the same mechanisms as MPI to launch OpenSHMEM, so you can use the same way you run MPI parcelport to run OpenSHMEM 
parcelport. Typically, it would be ``hpxrun``, ``mpirun``, ``srun``, or ``oshrun``.

If you are using ``hpxrun.py``, just pass ``--parcelport openshmem`` to the scripts.

If you are using ``mpirun``, ``oshrun``, or ``srun``, you can just pass
``--hpx:ini=hpx.parcel.openshmem.priority=1000``, ``--hpx:ini=hpx.parcel.openshmem.enable=1``, and
``--hpx:ini=hpx.parcel.bootstrap=openshmem`` to the |hpx| applications.

If you are running on a Cray machine, you need to pass `--mpi=pmix` or `--mpi=pmi2` to srun
to enable the PMIx or PMI2 support of SLURM since OpenSHMEM does not support the default Cray PMI.
For example,

.. code-block:: shell-session

   $ srun --mpi=pmix [hpx application]
