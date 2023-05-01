..
    Copyright (C) 2023 Dimitra Karatza

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _new_namespaces_1_9_0:

==============================
|hpx| V1.9.0 Namespace changes
==============================

The latest release includes amongst others changes in the namespaces so that |hpx|
facilities correspond to the C++ Standard Library. The old namespaces are
deprecated. Below is a comprehensive list of the namespace changes.

.. table:: Namespace changes in V1.9.0

   ===========================================================  ==============================================================
   Old namespace                                                New namespace
   ===========================================================  ==============================================================
   :cpp:func:`hpx::util::mem_fn`                                :cpp:func:`hpx::mem_fn`
   :cpp:func:`hpx::util::invoke`                                :cpp:func:`hpx::invoke`
   :cpp:func:`hpx::util::invoke_r`                              :cpp:func:`hpx::invoke_r`
   :cpp:func:`hpx::util::invoke_fused`                          :cpp:func:`hpx::invoke_fused`
   :cpp:func:`hpx::util::invoke_fused_r`                        :cpp:func:`hpx::invoke_fused_r`
   :cpp:class:`hpx::util::unlock_guard`                         :cpp:class:`hpx::unlock_guard`
   :cpp:func:`hpx::parallel::v1::reduce_by_key`                 :cpp:func:`hpx::experimental::reduce_by_key`
   :cpp:func:`hpx::parallel::v1::sort_by_key`                   :cpp:func:`hpx::experimental::sort_by_key`
   :cpp:class:`hpx::parallel::task_canceled_exception`          :cpp:class:`hpx::experimental::task_canceled_exception`
   :cpp:class:`hpx::parallel::task_block`                       :cpp:class:`hpx::experimental::task_block`
   :cpp:func:`hpx::parallel::define_task_block`                 :cpp:func:`hpx::experimental::define_task_block`                |
   :cpp:func:`hpx::parallel::define_task_block_restore_thread`  :cpp:func:`hpx::experimental::define_task_block_restore_thread`
   :cpp:class:`hpx::execution::experimental::task_group`        :cpp:class:`hpx::experimental::task_group`
   ===========================================================  ==============================================================
