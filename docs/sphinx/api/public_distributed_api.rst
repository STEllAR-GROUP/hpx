..
    Copyright (C) 2023 Dimitra Karatza

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _public_distributed_api:

======================
Public distributed API
======================

Our Public Distributed API offers a rich set of tools and functions that enable developers
to harness the full potential of distributed computing. Here, you'll find a comprehensive
list of header files, classes and functions for various distributed computing features
provided by |hpx|.

.. _public_distr_api_header_collectives:

``hpx/collectives.hpp``
=======================

The header :hpx-header:`libs/full/include/include,hpx/collectives.hpp`
contains definitions and implementations related to the collectives operations.

Classes
-------

.. table:: `hpx` classes of header ``hpx/collectives.hpp``

   +-----------------------------------------------------+
   | Function                                            |
   +=====================================================+
   | :cpp:class:`hpx::collectives::barrier`              |
   +-----------------------------------------------------+
   | :cpp:class:`hpx::collectives::channel_communicator` |
   +-----------------------------------------------------+

Functions
---------

.. table:: `hpx` functions of header ``hpx/collectives.hpp``

   +-----------------------------------------------------------+
   | Function                                                  |
   +===========================================================+
   | :cpp:func:`hpx::collectives::all_gather`                  |
   +-----------------------------------------------------------+
   | :cpp:func:`hpx::collectives::all_reduce`                  |
   +-----------------------------------------------------------+
   | :cpp:func:`hpx::collectives::all_to_all`                  |
   +-----------------------------------------------------------+
   | :cpp:func:`hpx::collectives::broadcast_to`                |
   +-----------------------------------------------------------+
   | :cpp:func:`hpx::collectives::broadcast_from`              |
   +-----------------------------------------------------------+
   | :cpp:func:`hpx::collectives::create_channel_communicator` |
   +-----------------------------------------------------------+
   | :cpp:func:`hpx::collectives::set`                         |
   +-----------------------------------------------------------+
   | :cpp:func:`hpx::collectives::get`                         |
   +-----------------------------------------------------------+
   | :cpp:func:`hpx::collectives::create_communication_set`    |
   +-----------------------------------------------------------+
   | :cpp:func:`hpx::collectives::create_communicator`         |
   +-----------------------------------------------------------+
   | :cpp:func:`hpx::collectives::create_local_communicator`   |
   +-----------------------------------------------------------+
   | :cpp:func:`hpx::collectives::exclusive_scan`              |
   +-----------------------------------------------------------+
   | :cpp:func:`hpx::collectives::gather_here`                 |
   +-----------------------------------------------------------+
   | :cpp:func:`hpx::collectives::gather_there`                |
   +-----------------------------------------------------------+
   | :cpp:func:`hpx::collectives::inclusive_scan`              |
   +-----------------------------------------------------------+
   | :cpp:func:`hpx::collectives::reduce_here`                 |
   +-----------------------------------------------------------+
   | :cpp:func:`hpx::collectives::reduce_there`                |
   +-----------------------------------------------------------+
   | :cpp:func:`hpx::collectives::scatter_from`                |
   +-----------------------------------------------------------+
   | :cpp:func:`hpx::collectives::scatter_to`                  |
   +-----------------------------------------------------------+

