..
    Copyright (c) 2023 Jiakun Yan

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _using_the_lci_parcelport:

========================
Using the LCI parcelport
========================

.. _info_lci:

Basic information
=================

The `Lightweight Communication Interface`_ (LCI) is an ongoing research project
aiming to provide efficient support for applications with irregular and asynchronous
communication patterns such as
graph analysis, sparse linear algebra, and task-based runtime on
modern parallel architectures. Its features include (a) support for
more communication primitives such as two-sided send/recv and
one-sided (dynamic or direct) remote put/get (b) better multi-threaded
performance (c) explicit user control of communication resource
(d) flexible signaling mechanisms such as synchronizer, completion queue,
and active message handler.
It is designed to be a low-level communication library used by
high-level libraries and frameworks.

The LCI parcelport is an experimental parcelport.
It aims to provide the best possible communication performance
on high-performance computation platforms.
Compared to the MPI parcelport, it uses much fewer messages
and memory copies to transfer an |hpx| parcel over the network.
Its message transmission path involves minimum synchronization
points and is almost lock-free. It is expected to be much faster
than the MPI parcelport.

.. _`Lightweight Communication Interface`: https://github.com/uiuc-hpc/LC

.. _build_lci_pp:

Build |hpx| with the LCI parcelport
===================================

While building |hpx|, you can specify a set of |cmake| variables to enable
and configure the LCI parcelport. Below, there is a set of the most important
and frequently used CMake variables.

.. option:: HPX_WITH_PARCELPORT_LCI

   Enable the LCI parcelport. This enables the use of LCI for networking operations in the |hpx| runtime.
   The default value is ``OFF`` because it's not available on all systems and/or requires another dependency. However,
   this experimental parcelport may provide better performance than the MPI parcelport.
   You must set this variable to ``ON`` in order to use the LCI parcelport. All the following variables only
   make sense when this variable is set to ``ON``.

.. option:: HPX_WITH_FETCH_LCI

   Use FetchContent to fetch LCI. The default value is ``OFF``.
   If this option is set to ``OFF``. You need to install your own LCI library and |hpx| will try
   to find it using |cmake| ``find_package``. You can specify the location of the LCI installation
   by the environmental variable ``LCI_ROOT``. Refer to the `LCI README`_ for how to install LCI.
   If this option is set to ``ON``. |hpx| will fetch and build LCI for you. You can use the following
   |cmake| variables to configure this behavior for your platform.

.. _`LCI README`: https://github.com/uiuc-hpc/LC#readme

.. option:: HPX_WITH_LCI_TAG

   This variable only takes effect when ``HPX_WITH_FETCH_LCI`` is set to ``ON``
   and ``FETCHCONTENT_SOURCE_DIR_LCI`` is not set.
   |hpx| will fetch LCI from its github repository. This variable controls the branch/tag LCI
   will be fetched.

.. option:: FETCHCONTENT_SOURCE_DIR_LCI

   This variable only takes effect when ``HPX_WITH_FETCH_LCI`` is set to ``ON``.
   When it is defined, ``HPX_WITH_LCI_TAG`` will be ignored.
   It accepts a path to a local version of LCI source code and |hpx| will fetch and build LCI from there.
   The default value is set conservatively for the stability of |hpx|, but users are welcome to set this
   variable to ``master`` for potentially better performance.

.. _run_lci_pp:

Run |hpx| with the LCI parcelport
=================================

We use the same mechanisms as MPI to launch LCI, so you can use the same way you run MPI parcelport to run LCI
parcelport. Typically, it would be ``hpxrun``, ``mpirun``, or ``srun``.

If you are using ``hpxrun.py``, just pass ``--parcelport lci`` to the scripts.

If you are using ``mpirun`` or ``srun``, you can just pass
``--hpx:ini=hpx.parcel.lci.priority=1000``, ``--hpx:ini=hpx.parcel.lci.enable=1``, and
``--hpx:ini=hpx.parcel.bootstrap=lci`` to the |hpx| applications.

If you are running on a Cray machine, you need to pass `--mpi=pmix` or `--mpi=pmi2` to srun
to enable the PMIx or PMI2 support of SLURM since LCI does not support the default Cray PMI.
For example,

.. code-block:: shell-session

   $ srun --mpi=pmix [hpx application]

.. _tune_lci_pp:

Performance tuning of the LCI parcelport
========================================

We encourage users to set the following environmental variables
when using the LCI parcelport to get better performance.

.. code-block:: shell-session

   $ export LCI_SERVER_MAX_SENDS=1024
   $ export LCI_SERVER_MAX_RECVS=4096
   $ export LCI_SERVER_NUM_PKTS=65536
   $ export LCI_SERVER_MAX_CQES=65536
   $ export LCI_PACKET_SIZE=12288

This setting needs roughly 800MB memory per process. The memory consumption mainly
comes from the packets, which can be calculated using `LCI_SERVER_NUM_PKTS x LCI_PACKET_SIZE`.
