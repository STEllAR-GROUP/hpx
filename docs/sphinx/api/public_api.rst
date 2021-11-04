..
    Copyright (C) 2020 ETH Zurich

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _public_api:

==========
Public API
==========

All names below are also available in the top-level ``hpx`` namespace unless
otherwise noted. The names in ``hpx`` should be preferred. The names in
sub-namespaces will eventually be removed.

Header ``hpx/algorithm.hpp``
============================

This header includes :ref:`public_api_header_hpx_local_algorithm` and contains
overloads of the algorithms for segmented iterators.

.. _public_api_header_hpx_local_algorithm:

Header ``hpx/local/algorithm.hpp``
==================================

Corresponds to the C++ standard library header :cppreference-header:`algorithm`.
See :ref:`parallel_algorithms` for more information about the parallel
algorithms.

Classes
-------

- :cpp:class:`hpx::parallel::v2::reduction`
- :cpp:class:`hpx::parallel::v2::induction`

Functions
---------

- :cpp:func:`hpx::adjacent_find`
- :cpp:func:`hpx::all_of`
- :cpp:func:`hpx::any_of`
- :cpp:func:`hpx::copy`
- :cpp:func:`hpx::copy_if`
- :cpp:func:`hpx::copy_n`
- :cpp:func:`hpx::count`
- :cpp:func:`hpx::count_if`
- :cpp:func:`hpx::ends_with`
- :cpp:func:`hpx::equal`
- :cpp:func:`hpx::fill`
- :cpp:func:`hpx::fill_n`
- :cpp:func:`hpx::find`
- :cpp:func:`hpx::find_end`
- :cpp:func:`hpx::find_first_of`
- :cpp:func:`hpx::find_if`
- :cpp:func:`hpx::find_if_not`
- :cpp:func:`hpx::for_each`
- :cpp:func:`hpx::for_each_n`
- :cpp:func:`hpx::generate`
- :cpp:func:`hpx::generate_n`
- :cpp:func:`hpx::includes`
- :cpp:func:`hpx::inplace_merge`
- :cpp:func:`hpx::is_heap`
- :cpp:func:`hpx::is_heap_until`
- :cpp:func:`hpx::is_partitioned`
- :cpp:func:`hpx::is_sorted`
- :cpp:func:`hpx::is_sorted_until`
- :cpp:func:`hpx::lexicographical_compare`
- :cpp:func:`hpx::make_heap`
- :cpp:func:`hpx::parallel::v1::max_element`
- :cpp:func:`hpx::merge`
- :cpp:func:`hpx::parallel::v1::min_element`
- :cpp:func:`hpx::parallel::v1::minmax_element`
- :cpp:func:`hpx::parallel::v1::mismatch`
- :cpp:func:`hpx::move`
- :cpp:func:`hpx::none_of`
- :cpp:func:`hpx::nth_element`
- :cpp:func:`hpx::partial_sort`
- :cpp:func:`hpx::partition`
- :cpp:func:`hpx::partition_copy`
- :cpp:func:`hpx::remove`
- :cpp:func:`hpx::remove_copy`
- :cpp:func:`hpx::remove_copy_if`
- :cpp:func:`hpx::remove_if`
- :cpp:func:`hpx::replace`
- :cpp:func:`hpx::replace_copy`
- :cpp:func:`hpx::replace_copy_if`
- :cpp:func:`hpx::replace_if`
- :cpp:func:`hpx::reverse`
- :cpp:func:`hpx::reverse_copy`
- :cpp:func:`hpx::rotate`
- :cpp:func:`hpx::rotate_copy`
- :cpp:func:`hpx::search`
- :cpp:func:`hpx::search_n`
- :cpp:func:`hpx::set_difference`
- :cpp:func:`hpx::set_intersection`
- :cpp:func:`hpx::set_symmetric_difference`
- :cpp:func:`hpx::set_union`
- :cpp:func:`hpx::shift_left`
- :cpp:func:`hpx::shift_right`
- :cpp:func:`hpx::sort`
- :cpp:func:`hpx::stable_partition`
- :cpp:func:`hpx::stable_sort`
- :cpp:func:`hpx::starts_with`
- :cpp:func:`hpx::swap_ranges`
- :cpp:func:`hpx::transform`
- :cpp:func:`hpx::unique`
- :cpp:func:`hpx::unique_copy`
- :cpp:func:`hpx::for_loop`
- :cpp:func:`hpx::for_loop_strided`
- :cpp:func:`hpx::for_loop_n`
- :cpp:func:`hpx::for_loop_n_strided`

- :cpp:func:`hpx::ranges::adjacent_find`
- :cpp:func:`hpx::ranges::all_of`
- :cpp:func:`hpx::ranges::any_of`
- :cpp:func:`hpx::ranges::copy`
- :cpp:func:`hpx::ranges::copy_if`
- :cpp:func:`hpx::ranges::copy_n`
- :cpp:func:`hpx::ranges::count`
- :cpp:func:`hpx::ranges::count_if`
- :cpp:func:`hpx::ranges::ends_with`
- :cpp:func:`hpx::ranges::equal`
- :cpp:func:`hpx::ranges::fill`
- :cpp:func:`hpx::ranges::fill_n`
- :cpp:func:`hpx::ranges::find`
- :cpp:func:`hpx::ranges::find_end`
- :cpp:func:`hpx::ranges::find_first_of`
- :cpp:func:`hpx::ranges::find_if`
- :cpp:func:`hpx::ranges::find_if_not`
- :cpp:func:`hpx::ranges::for_each`
- :cpp:func:`hpx::ranges::for_each_n`
- :cpp:func:`hpx::ranges::generate`
- :cpp:func:`hpx::ranges::generate_n`
- :cpp:func:`hpx::ranges::includes`
- :cpp:func:`hpx::ranges::inplace_merge`
- :cpp:func:`hpx::ranges::is_heap`
- :cpp:func:`hpx::ranges::is_heap_until`
- :cpp:func:`hpx::ranges::is_partitioned`
- :cpp:func:`hpx::ranges::is_sorted`
- :cpp:func:`hpx::ranges::is_sorted_until`
- :cpp:func:`hpx::ranges::make_heap`
- :cpp:func:`hpx::ranges::merge`
- :cpp:func:`hpx::ranges::move`
- :cpp:func:`hpx::ranges::none_of`
- :cpp:func:`hpx::ranges::nth_element`
- :cpp:func:`hpx::ranges::partition`
- :cpp:func:`hpx::ranges::partition_copy`
- :cpp:func:`hpx::ranges::set_difference`
- :cpp:func:`hpx::ranges::set_intersection`
- :cpp:func:`hpx::ranges::set_symmetric_difference`
- :cpp:func:`hpx::ranges::set_union`
- :cpp:func:`hpx::ranges::shift_left`
- :cpp:func:`hpx::ranges::shift_right`
- :cpp:func:`hpx::ranges::sort`
- :cpp:func:`hpx::ranges::stable_partition`
- :cpp:func:`hpx::ranges::stable_sort`
- :cpp:func:`hpx::ranges::starts_with`
- :cpp:func:`hpx::ranges::swap_ranges`
- :cpp:func:`hpx::ranges::unique`
- :cpp:func:`hpx::ranges::unique_copy`
- :cpp:func:`hpx::ranges::for_loop`
- :cpp:func:`hpx::ranges::for_loop_strided`

Header ``hpx/any.hpp``
======================

This header includes :ref:`public_api_header_hpx_local_any`.

.. _public_api_header_hpx_local_any:

Header ``hpx/local/any.hpp``
============================

Corresponds to the C++ standard library header :cppreference-header:`any`.
:cpp:type:`hpx::any` is compatible with ``std::any``.

Classes
-------

- :cpp:type:`hpx::any`
- :cpp:type:`hpx::any_nonser`
- :cpp:type:`hpx::bad_any_cast`
- :cpp:type:`hpx::unique_any_nonser`

Functions
---------

- :cpp:func:`hpx::any_cast`
- :cpp:func:`hpx::make_any`
- :cpp:func:`hpx::make_any_nonser`
- :cpp:func:`hpx::make_unique_any_nonser`

Header ``hpx/assert.hpp``
=========================

Corresponds to the C++ standard library header :cppreference-header:`cassert`.
:c:macro:`HPX_ASSERT` is the |hpx| equivalent to ``assert`` in ``cassert``.
:c:macro:`HPX_ASSERT` can also be used in CUDA device code.

Macros
------

- :c:macro:`HPX_ASSERT`
- :c:macro:`HPX_ASSERT_MSG`

Header ``hpx/barrier.hpp``
==========================

This header includes :ref:`public_api_header_hpx_local_barrier` and contains a
distributed barrier implementation. This functionality is also exposed through
the ``hpx::distributed`` namespace. The name in ``hpx::distributed`` should be
preferred.

Classes
-------

- :cpp:class:`hpx::lcos::barrier`

.. _public_api_header_hpx_local_barrier:

Header ``hpx/local/barrier.hpp``
================================

Corresponds to the C++ standard library header :cppreference-header:`barrier`.

Classes
-------

- :cpp:class:`hpx::lcos::local::cpp20_barrier`

Header ``hpx/channel.hpp``
==========================

This header includes :ref:`public_api_header_hpx_local_channel` and contains a
distributed channel implementation. This functionality is also exposed through
the ``hpx::distributed`` namespace. The name in ``hpx::distributed`` should be
preferred.

Classes
-------

- :cpp:class:`hpx::lcos::channel`

.. _public_api_header_hpx_local_channel:

Header ``hpx/local/channel.hpp``
================================

Contains a local channel implementation.

Classes
-------

- :cpp:class:`hpx::lcos::local::channel`

Header ``hpx/chrono.hpp``
=========================

This header includes :ref:`public_api_header_hpx_local_chrono`.

.. _public_api_header_hpx_local_chrono:

Header ``hpx/local/chrono.hpp``
===============================

Corresponds to the C++ standard library header :cppreference-header:`chrono`.
The following replacements and extensions are provided compared to
:cppreference-header:`chrono`. The classes below are also available in the
``hpx::chrono`` namespace, not in the top-level ``hpx`` namespace.

Classes
-------

- :cpp:class:`hpx::chrono::high_resolution_clock`
- :cpp:class:`hpx::chrono::high_resolution_timer`
- :cpp:class:`hpx::chrono::steady_time_point`

Header ``hpx/condition_variable.hpp``
=====================================

This header includes :ref:`public_api_header_hpx_local_condition_variable`.

.. _public_api_header_hpx_local_condition_variable:

Header ``hpx/local/condition_variable.hpp``
===========================================

Corresponds to the C++ standard library header
:cppreference-header:`condition_variable`.

Classes
-------

- :cpp:class:`hpx::lcos::local::condition_variable`
- :cpp:class:`hpx::lcos::local::condition_variable_any`
- :cpp:class:`hpx::lcos::local::cv_status`

Header ``hpx/exception.hpp``
============================

This header includes :ref:`public_api_header_hpx_local_exception`.

.. _public_api_header_hpx_local_exception:

Header ``hpx/local/exception.hpp``
==================================

Corresponds to the C++ standard library header :cppreference-header:`exception`.
:cpp:class:`hpx::exception` extends ``std::exception`` and is the base class for
all exceptions thrown in |hpx|. :c:macro:`HPX_THROW_EXCEPTION` can be used to
throw |hpx| exceptions with file and line information attached to the exception.

Macros
------

- :c:macro:`HPX_THROW_EXCEPTION`

Classes
-------

- :cpp:class:`hpx::exception`

Header ``hpx/execution.hpp``
============================

This header includes :ref:`public_api_header_hpx_local_execution`.

.. _public_api_header_hpx_local_execution:

Header ``hpx/local/execution.hpp``
==================================

Corresponds to the C++ standard library header :cppreference-header:`execution`.
See :ref:`parallel`, :ref:`parallel_algorithms` and :ref:`executor_parameters`
for more information about execution policies and executor parameters.

.. note::

   These names are only available in the ``hpx::execution`` namespace, not in
   the top-level ``hpx`` namespace.

Constants
---------

- :cpp:var:`hpx::execution::seq`
- :cpp:var:`hpx::execution::par`
- :cpp:var:`hpx::execution::par_unseq`
- :cpp:var:`hpx::execution::task`

Classes
-------

- :cpp:class:`hpx::execution::sequenced_policy`
- :cpp:class:`hpx::execution::parallel_policy`
- :cpp:class:`hpx::execution::parallel_unsequenced_policy`
- :cpp:class:`hpx::execution::sequenced_task_policy`
- :cpp:class:`hpx::execution::parallel_task_policy`
- :cpp:class:`hpx::execution::auto_chunk_size`
- :cpp:class:`hpx::execution::dynamic_chunk_size`
- :cpp:class:`hpx::execution::guided_chunk_size`
- :cpp:class:`hpx::execution::persistent_auto_chunk_size`
- :cpp:class:`hpx::execution::static_chunk_size`

Header ``hpx/functional.hpp``
=============================

This header includes :ref:`public_api_header_hpx_local_functional`.

.. _public_api_header_hpx_local_functional:

Header ``hpx/local/functional.hpp``
===================================

Corresponds to the C++ standard library header
:cppreference-header:`functional`. :cpp:class:`hpx::util::function` is a more
efficient and serializable replacement for ``std::function``.

Constants
---------

The following constants are also available in ``hpx::placeholders``, not the
top-level ``hpx`` namespace.

- :cpp:var:`hpx::util::placeholders::_1`
- :cpp:var:`hpx::util::placeholders::_2`
- ...
- :cpp:var:`hpx::util::placeholders::_9`

Classes
-------

- :cpp:class:`hpx::util::function`
- :cpp:class:`hpx::util::function_nonser`
- :cpp:class:`hpx::util::function_ref`
- :cpp:class:`hpx::util::unique_function`
- :cpp:class:`hpx::util::unique_function_nonser`
- :cpp:struct:`hpx::traits::is_bind_expression`
- :cpp:struct:`hpx::traits::is_placeholder`

Functions
---------

- :cpp:func:`hpx::util::bind`
- :cpp:func:`hpx::util::bind_back`
- :cpp:func:`hpx::util::bind_front`
- :cpp:func:`hpx::util::invoke`
- :cpp:func:`hpx::util::invoke_fused`
- :cpp:func:`hpx::util::mem_fn`

Header ``hpx/future.hpp``
=========================

This header includes :ref:`public_api_header_hpx_local_future` and contains
overloads of :cpp:func:`hpx::async`, :cpp:func:`hpx::apply`,
:cpp:func:`hpx::sync`, and :cpp:func:`hpx::dataflow` that can be used with
actions. See :ref:`action_invocation` for more information about invoking
actions.

.. note::

   The alias from ``hpx::promise`` to :cpp:class:`hpx::lcos::promise` is
   deprecated and will be removed in a future release. The alias
   ``hpx::distributed::promise`` should be used in new applications.

Classes
-------

- :cpp:class:`hpx::lcos::promise`

Functions
---------

- :cpp:func:`hpx::async`
- :cpp:func:`hpx::apply`
- :cpp:func:`hpx::sync`
- :cpp:func:`hpx::dataflow`

.. _public_api_header_hpx_local_future:

Header ``hpx/local/future.hpp``
===============================

Corresponds to the C++ standard library header :cppreference-header:`future`.
See :ref:`extend_futures` for more information about extensions to futures
compared to the C++ standard library.

.. note::

   All names except :cpp:class:`hpx::lcos::local::promise` are also available in
   the top-level ``hpx`` namespace. ``hpx::promise`` refers to
   :cpp:class:`hpx::lcos::promise`, a distributed variant of
   :cpp:class:`hpx::lcos::local::promise`, but will eventually refer to
   :cpp:class:`hpx::lcos::local::promise` after a deprecation period.

Classes
-------

- :cpp:class:`hpx::lcos::future`
- :cpp:class:`hpx::lcos::shared_future`
- :cpp:class:`hpx::lcos::local::promise`
- :cpp:class:`hpx::launch`

Functions
---------

- :cpp:func:`hpx::lcos::make_future`
- :cpp:func:`hpx::lcos::make_shared_future`
- :cpp:func:`hpx::lcos::make_ready_future`
- :cpp:func:`hpx::async`
- :cpp:func:`hpx::apply`
- :cpp:func:`hpx::sync`
- :cpp:func:`hpx::dataflow`
- :cpp:func:`hpx::when_all`
- :cpp:func:`hpx::when_any`
- :cpp:func:`hpx::when_some`
- :cpp:func:`hpx::when_each`
- :cpp:func:`hpx::wait_all`
- :cpp:func:`hpx::wait_any`
- :cpp:func:`hpx::wait_some`
- :cpp:func:`hpx::wait_each`

Examples
--------

.. literalinclude:: ../../libs/full/include/tests/unit/api_future.cpp
   :language: c++
   :lines: 7-

Header ``hpx/init.hpp``
=======================

This header contains functionality for starting, stopping, suspending, and
resuming the |hpx| runtime. This is the main way to explicitly start the |hpx|
runtime. See :ref:`starting_hpx` for more details on starting the |hpx| runtime.

Classes
-------

- :cpp:class:`hpx::init_params`
- :cpp:enum:`hpx::runtime_mode`

Functions
---------

- :cpp:func:`hpx::init`
- :cpp:func:`hpx::start`
- :cpp:func:`hpx::finalize`
- :cpp:func:`hpx::disconnect`
- :cpp:func:`hpx::suspend`
- :cpp:func:`hpx::resume`

Header ``hpx/latch.hpp``
========================

This header includes :ref:`public_api_header_hpx_local_latch` and contains a
distributed latch implementation. This functionality is also exposed through the
``hpx::distributed`` namespace. The name in ``hpx::distributed`` should be
preferred.

Classes
-------

- :cpp:class:`hpx::lcos::latch`

.. _public_api_header_hpx_local_latch:

Header ``hpx/local/latch.hpp``
==============================

Corresponds to the C++ standard library header :cppreference-header:`latch`.

Classes
-------

- :cpp:class:`hpx::lcos::local::cpp20_latch`

Header ``hpx/mutex.hpp``
========================

This header includes :ref:`public_api_header_hpx_local_mutex`.

.. _public_api_header_hpx_local_mutex:

Header ``hpx/local/mutex.hpp``
==============================

Corresponds to the C++ standard library header :cppreference-header:`mutex`.

Classes
-------

- :cpp:class:`hpx::lcos::local::mutex`
- :cpp:class:`hpx::lcos::local::no_mutex`
- :cpp:class:`hpx::lcos::local::once_flag`
- :cpp:class:`hpx::lcos::local::recursive_mutex`
- :cpp:class:`hpx::lcos::local::spinlock`
- :cpp:class:`hpx::lcos::local::timed_mutex`
- :cpp:class:`hpx::lcos::local::unlock_guard`

Functions
---------

- :cpp:func:`hpx::lcos::local::call_once`

Header ``hpx/memory.hpp``
=========================

This header includes :ref:`public_api_header_hpx_local_memory`.

.. _public_api_header_hpx_local_memory:

Header ``hpx/local/memory.hpp``
===============================

Corresponds to the C++ standard library header :cppreference-header:`memory`. It
contains parallel versions of the copy, fill, move, and construct helper
functions in :cppreference-header:`memory`. See :ref:`parallel_algorithms` for
more information about the parallel algorithms.

Functions
---------

- :cpp:func:`hpx::uninitialized_copy`
- :cpp:func:`hpx::uninitialized_copy_n`
- :cpp:func:`hpx::uninitialized_default_construct`
- :cpp:func:`hpx::uninitialized_default_construct_n`
- :cpp:func:`hpx::uninitialized_fill`
- :cpp:func:`hpx::uninitialized_fill_n`
- :cpp:func:`hpx::uninitialized_move`
- :cpp:func:`hpx::uninitialized_move_n`
- :cpp:func:`hpx::uninitialized_value_construct`
- :cpp:func:`hpx::uninitialized_value_construct_n`

- :cpp:func:`hpx::ranges::uninitialized_copy`
- :cpp:func:`hpx::ranges::uninitialized_copy_n`
- :cpp:func:`hpx::ranges::uninitialized_default_construct`
- :cpp:func:`hpx::ranges::uninitialized_default_construct_n`
- :cpp:func:`hpx::ranges::uninitialized_fill`
- :cpp:func:`hpx::ranges::uninitialized_fill_n`
- :cpp:func:`hpx::ranges::uninitialized_move`
- :cpp:func:`hpx::ranges::uninitialized_move_n`
- :cpp:func:`hpx::ranges::uninitialized_value_construct`
- :cpp:func:`hpx::ranges::uninitialized_value_construct_n`

Header ``hpx/numeric.hpp``
==========================

This header includes :ref:`public_api_header_hpx_local_numeric`.

.. _public_api_header_hpx_local_numeric:

Header ``hpx/local/numeric.hpp``
================================

Corresponds to the C++ standard library header :cppreference-header:`numeric`.
See :ref:`parallel_algorithms` for more information about the parallel
algorithms.

Functions
---------

- :cpp:func:`hpx::parallel::v1::adjacent_difference`
- :cpp:func:`hpx::exclusive_scan`
- :cpp:func:`hpx::inclusive_scan`
- :cpp:func:`hpx::reduce`
- :cpp:func:`hpx::transform_exclusive_scan`
- :cpp:func:`hpx::transform_inclusive_scan`
- :cpp:func:`hpx::transform_reduce`

- :cpp:func:`hpx::ranges::exclusive_scan`
- :cpp:func:`hpx::ranges::inclusive_scan`
- :cpp:func:`hpx::ranges::transform_exclusive_scan`
- :cpp:func:`hpx::ranges::transform_inclusive_scan`

Header ``hpx/optional.hpp``
===========================

This header includes :ref:`public_api_header_hpx_local_optional`.

.. _public_api_header_hpx_local_optional:

Header ``hpx/local/optional.hpp``
=================================

Corresponds to the C++ standard library header :cppreference-header:`optional`.
:cpp:type:`hpx::util::optional` is compatible with ``std::optional``.

Constants
---------

- :cpp:var:`hpx::util::nullopt`

Classes
-------

- :cpp:class:`hpx::util::optional`
- :cpp:class:`hpx::util::nullopt_t`
- :cpp:class:`hpx::util::bad_optional_access`

Functions
---------

- :cpp:func:`hpx::util::make_optional`

Header ``hpx/runtime.hpp``
==========================

This header includes :ref:`public_api_header_hpx_local_runtime` and contains
functions for accessing distributed runtime information.

Functions
---------

- :cpp:func:`hpx::find_root_locality`
- :cpp:func:`hpx::find_all_localities`
- :cpp:func:`hpx::find_remote_localities`
- :cpp:func:`hpx::find_locality`
- :cpp:func:`hpx::get_colocation_id`
- :cpp:func:`hpx::get_locality_id`

.. _public_api_header_hpx_local_runtime:

Header ``hpx/local/runtime.hpp``
================================

This header contains functions for accessing local runtime information.

Typedefs
--------

- :cpp:type:`hpx::startup_function_type`
- :cpp:type:`hpx::shutdown_function_type`

Functions
---------

- :cpp:func:`hpx::get_num_worker_threads`
- :cpp:func:`hpx::get_worker_thread_num`
- :cpp:func:`hpx::get_thread_name`
- :cpp:func:`hpx::register_pre_startup_function`
- :cpp:func:`hpx::register_startup_function`
- :cpp:func:`hpx::register_pre_shutdown_function`
- :cpp:func:`hpx::register_shutdown_function`
- :cpp:func:`hpx::get_num_localities`
- :cpp:func:`hpx::get_locality_name`

Header ``hpx/system_error.hpp``
===============================

This header includes :ref:`public_api_header_hpx_local_system_error`.

.. _public_api_header_hpx_local_system_error:

Header ``hpx/local/system_error.hpp``
=====================================

Corresponds to the C++ standard library header
:cppreference-header:`system_error`.

Classes
-------

- :cpp:class:`hpx::error_code`

Header ``hpx/task_block.hpp``
=============================

This header includes :ref:`public_api_header_hpx_local_task_block`.

.. _public_api_header_hpx_local_task_block:

Header ``hpx/local/task_black.hpp``
===================================

Corresponds to the ``task_block`` feature in |cpp11_n4088|_. See
:ref:`using_task_block` for more details on using task blocks.

Classes
-------

- :cpp:class:`hpx::parallel::v2::task_canceled_exception`
- :cpp:class:`hpx::parallel::v2::task_block`

Functions
---------

- :cpp:func:`hpx::parallel::v2::define_task_block`
- :cpp:func:`hpx::parallel::v2::define_task_block_restore_thread`

Header ``hpx/thread.hpp``
=========================

This header includes :ref:`public_api_header_hpx_local_thread`.

.. _public_api_header_hpx_local_thread:

Header ``hpx/local/thread.hpp``
===============================

Corresponds to the C++ standard library header :cppreference-header:`thread`.
The functionality in this header is equivalent to the standard library thread
functionality, with the exception that the |hpx| equivalents are implemented on
top of lightweight threads and the |hpx| runtime.

Classes
-------

- :cpp:class:`hpx::thread`
- :cpp:class:`hpx::jthread`

Functions
---------

- :cpp:func:`hpx::this_thread::yield`
- :cpp:func:`hpx::this_thread::get_id`
- :cpp:func:`hpx::this_thread::sleep_for`
- :cpp:func:`hpx::this_thread::sleep_until`

Header ``hpx/semaphore.hpp``
============================

This header includes :ref:`public_api_header_hpx_local_semaphore`.

.. _public_api_header_hpx_local_semaphore:

Header ``hpx/local/semaphore.hpp``
==================================

Corresponds to the C++ standard library header
:cppreference-header:`semaphore`.

Classes
-------

- :cpp:class:`hpx::lcos::local::cpp20_binary_semaphore`
- :cpp:class:`hpx::lcos::local::cpp20_counting_semaphore`

Header ``hpx/shared_mutex.hpp``
===============================

This header includes :ref:`public_api_header_hpx_local_shared_mutex`.

.. _public_api_header_hpx_local_shared_mutex:

Header ``hpx/local/shared_mutex.hpp``
=====================================

Corresponds to the C++ standard library header
:cppreference-header:`shared_mutex`.

Classes
-------

- :cpp:class:`hpx::lcos::local::shared_mutex`

Header ``hpx/stop_token.hpp``
=============================

This header includes :ref:`public_api_header_hpx_local_stop_token`.

.. _public_api_header_hpx_local_stop_token:

Header ``hpx/local/stop_token.hpp``
===================================

Corresponds to the C++ standard library header
:cppreference-header:`stop_token`.

Constants
---------

- :cpp:var:`hpx::nostopstate`

Classes
-------

- :cpp:class:`hpx::stop_callback`
- :cpp:class:`hpx::stop_source`
- :cpp:class:`hpx::stop_token`
- :cpp:struct:`hpx::nostopstate_t`

Header ``hpx/tuple.hpp``
========================

This header includes :ref:`public_api_header_hpx_local_tuple`.

.. _public_api_header_hpx_local_tuple:

Header ``hpx/local/tuple.hpp``
==============================

Corresponds to the C++ standard library header :cppreference-header:`tuple`.
:cpp:class:`hpx::tuple` can be used in CUDA device code, unlike ``std::tuple``.

Constants
---------

- :cpp:var:`hpx::ignore`

Classes
-------

- :cpp:struct:`hpx::tuple`
- :cpp:struct:`hpx::tuple_size`
- :cpp:struct:`hpx::tuple_element`

Functions
---------

- :cpp:func:`hpx::make_tuple`
- :cpp:func:`hpx::tie`
- :cpp:func:`hpx::forward_as_tuple`
- :cpp:func:`hpx::tuple_cat`
- :cpp:func:`hpx::get`

Header ``hpx/type_traits.hpp``
==============================

This header includes :ref:`public_api_header_hpx_local_type_traits`.

.. _public_api_header_hpx_local_type_traits:

Header ``hpx/local/type_traits.hpp``
====================================

Corresponds to the C++ standard library header
:cppreference-header:`type_traits`.

Classes
-------

- :cpp:struct:`hpx::is_invocable`
- :cpp:struct:`hpx::is_invocable_r`

Header ``hpx/unwrap.hpp``
=========================

This header includes :ref:`public_api_header_hpx_local_unwrap`.

.. _public_api_header_hpx_local_unwrap:

Header ``hpx/local/unwrap.hpp``
===============================

Contains utilities for unwrapping futures.

Classes
-------

- :cpp:struct:`hpx::functional::unwrap`
- :cpp:struct:`hpx::functional::unwrap_n`
- :cpp:struct:`hpx::functional::unwrap_all`

Functions
---------

- :cpp:func:`hpx::unwrap`
- :cpp:func:`hpx::unwrap_n`
- :cpp:func:`hpx::unwrap_all`
- :cpp:func:`hpx::unwrapping`
- :cpp:func:`hpx::unwrapping_n`
- :cpp:func:`hpx::unwrapping_all`

Header ``hpx/version.hpp``
==========================

This header provides version information about |hpx|.

Macros
------

- :c:macro:`HPX_VERSION_MAJOR`
- :c:macro:`HPX_VERSION_MINOR`
- :c:macro:`HPX_VERSION_SUBMINOR`
- :c:macro:`HPX_VERSION_FULL`
- :c:macro:`HPX_VERSION_DATE`
- :c:macro:`HPX_VERSION_TAG`
- :c:macro:`HPX_AGAS_VERSION`

Functions
---------

- :cpp:func:`hpx::major_version`
- :cpp:func:`hpx::minor_version`
- :cpp:func:`hpx::subminor_version`
- :cpp:func:`hpx::full_version`
- :cpp:func:`hpx::full_version_as_string`
- :cpp:func:`hpx::tag`
- :cpp:func:`hpx::agas_version`
- :cpp:func:`hpx::build_type`
- :cpp:func:`hpx::build_date_time`

Header ``hpx/wrap_main.hpp``
============================

This header does not provide any direct functionality but is used for implicitly
using ``main`` as the runtime entry point. See :ref:`minimal` for more details
on implicitly starting the |hpx| runtime.
