..
    Copyright (c) 2019-2022 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _modules_synchronization:

===============
synchronization
===============

This module provides synchronization primitives that should be used rather than
the C++ standard ones in |hpx| threads:

* :cpp:class:`hpx::barrier`
* :cpp:class:`hpx::binary_semaphore`
* :cpp:class:`hpx::call_once`
* :cpp:class:`hpx::condition_variable`
* :cpp:class:`hpx::condition_variable_any`
* :cpp:class:`hpx::counting_semaphore`
* :cpp:class:`hpx::lcos::local::event`
* :cpp:class:`hpx::latch`
* :cpp:class:`hpx::mutex`
* :cpp:class:`hpx::no_mutex`
* :cpp:class:`hpx::once_flag`
* :cpp:class:`hpx::recursive_mutex`
* :cpp:class:`hpx::shared_mutex`
* :cpp:class:`hpx::sliding_semaphore`
* :cpp:class:`hpx::spinlock` (`std::mutex` compatible spinlock)
* :cpp:class:`hpx::spinlock_no_backoff` (`boost::mutex` compatible spinlock)
* :cpp:class:`hpx::spinlock_pool`
* :cpp:class:`hpx::stop_callback`
* :cpp:class:`hpx::stop_source`
* :cpp:class:`hpx::stop_token`
* :cpp:class:`hpx::in_place_stop_token`
* :cpp:class:`hpx::timed_mutex`
* :cpp:class:`hpx::upgrade_to_unique_lock`
* :cpp:class:`hpx::upgrade_lock`

See :ref:`modules_lcos_local`, :ref:`modules_async_combinators`, and
:ref:`modules_async_distributed` for higher level synchronization facilities.

See the :ref:`API reference <modules_synchronization_api>` of this module for more
details.

