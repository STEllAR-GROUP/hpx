..
    Copyright (c) 2019 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _modules_lcos_local:

==========
lcos_local
==========

This module provides the following local :term:`LCO`\ s:

* :cpp:class:`hpx::lcos::local::and_gate`
* :cpp:class:`hpx::lcos::local::channel`
* :cpp:class:`hpx::lcos::local::one_element_channel`
* :cpp:class:`hpx::lcos::local::receive_channel`
* :cpp:class:`hpx::lcos::local::send_channel`
* :cpp:class:`hpx::lcos::local::guard`
* :cpp:class:`hpx::lcos::local::guard_set`
* :cpp:func:`hpx::lcos::local::run_guarded`
* :cpp:class:`hpx::lcos::local::conditional_trigger`
* :cpp:class:`hpx::lcos::local::packaged_task`
* :cpp:class:`hpx::lcos::local::promise`
* :cpp:class:`hpx::lcos::local::receive_buffer`
* :cpp:class:`hpx::lcos::local::trigger`

See :ref:`modules_lcos_distributed` for distributed LCOs. Basic synchronization
primitives for use in |hpx| threads can be found in :ref:`modules_synchronization`.
:ref:`async_combinators` contains useful utility functions for combining
futures.

See the :ref:`API reference <modules_lcos_local_api>` of this module for more
details.

