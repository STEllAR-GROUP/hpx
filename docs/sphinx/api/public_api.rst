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

Corresponds to the C++ standard library header :cppreference-header:`algorithm`.
See :ref:`parallel_algorithms` for more information about the parallel
algorithms.

Classes
-------

- :cpp:class:`hpx::parallel::v2::reduction`
- :cpp:class:`hpx::parallel::v2::induction`

Functions
---------

- :cpp:func:`hpx::parallel::v1::adjacent_find`
- :cpp:func:`hpx::all_of`
- :cpp:func:`hpx::any_of`
- :cpp:func:`hpx::copy`
- :cpp:func:`hpx::copy_if`
- :cpp:func:`hpx::copy_n`
- :cpp:func:`hpx::count`
- :cpp:func:`hpx::count_if`
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
- :cpp:func:`hpx::parallel::v1::includes`
- :cpp:func:`hpx::parallel::v1::inplace_merge`
- :cpp:func:`hpx::parallel::v1::is_heap`
- :cpp:func:`hpx::parallel::v1::is_heap_until`
- :cpp:func:`hpx::parallel::v1::is_partitioned`
- :cpp:func:`hpx::parallel::v1::is_sorted`
- :cpp:func:`hpx::parallel::v1::is_sorted_until`
- :cpp:func:`hpx::parallel::v1::lexicographical_compare`
- :cpp:func:`hpx::parallel::v1::max_element`
- :cpp:func:`hpx::parallel::v1::merge`
- :cpp:func:`hpx::parallel::v1::min_element`
- :cpp:func:`hpx::parallel::v1::minmax_element`
- :cpp:func:`hpx::parallel::v1::mismatch`
- :cpp:func:`hpx::move`
- :cpp:func:`hpx::none_of`
- :cpp:func:`hpx::parallel::v1::partition`
- :cpp:func:`hpx::parallel::v1::partition_copy`
- :cpp:func:`hpx::parallel::v1::remove`
- :cpp:func:`hpx::parallel::v1::remove_copy`
- :cpp:func:`hpx::parallel::v1::remove_copy_if`
- :cpp:func:`hpx::parallel::v1::remove_if`
- :cpp:func:`hpx::parallel::v1::replace`
- :cpp:func:`hpx::parallel::v1::replace_copy`
- :cpp:func:`hpx::parallel::v1::replace_copy_if`
- :cpp:func:`hpx::parallel::v1::replace_if`
- :cpp:func:`hpx::parallel::v1::reverse`
- :cpp:func:`hpx::parallel::v1::reverse_copy`
- :cpp:func:`hpx::parallel::v1::rotate`
- :cpp:func:`hpx::parallel::v1::rotate_copy`
- :cpp:func:`hpx::parallel::v1::search`
- :cpp:func:`hpx::parallel::v1::search_n`
- :cpp:func:`hpx::parallel::v1::set_difference`
- :cpp:func:`hpx::parallel::v1::set_intersection`
- :cpp:func:`hpx::parallel::v1::set_symmetric_difference`
- :cpp:func:`hpx::parallel::v1::set_union`
- :cpp:func:`hpx::parallel::v1::sort`
- :cpp:func:`hpx::parallel::v1::stable_partition`
- :cpp:func:`hpx::parallel::v1::stable_sort`
- :cpp:func:`hpx::parallel::v1::swap_ranges`
- :cpp:func:`hpx::parallel::v1::unique`
- :cpp:func:`hpx::parallel::v1::unique_copy`
- :cpp:func:`hpx::parallel::v2::for_loop`
- :cpp:func:`hpx::parallel::v2::for_loop_strided`
- :cpp:func:`hpx::parallel::v2::for_loop_n`
- :cpp:func:`hpx::parallel::v2::for_loop_n_strided`

- :cpp:func:`hpx::ranges::all_of`
- :cpp:func:`hpx::ranges::any_of`
- :cpp:func:`hpx::ranges::copy`
- :cpp:func:`hpx::ranges::copy_if`
- :cpp:func:`hpx::ranges::copy_n`
- :cpp:func:`hpx::ranges::count`
- :cpp:func:`hpx::ranges::count_if`
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
- :cpp:func:`hpx::ranges::move`
- :cpp:func:`hpx::ranges::none_of`

Header ``hpx/any.hpp``
======================

Corresponds to the C++ standard library header :cppreference-header:`any`.
:cpp:type:`hpx::util::any` is compatible with ``std::any``.

Classes
-------

- :cpp:type:`hpx::util::any`
- :cpp:type:`hpx::util::any_nonser`
- :cpp:type:`hpx::util::bad_any_cast`
- :cpp:type:`hpx::util::unique_any_nonser`

Functions
---------

- :cpp:func:`hpx::util::make_any`
- :cpp:func:`hpx::util::make_any_nonser`
- :cpp:func:`hpx::util::make_unique_any_nonser`

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

This header includes :ref:`public_api_header_hpx_local_barrier` and
ref:`public_api_header_hpx_distributed_barrier`.

.. _public_api_header_hpx_local_barrier:

Header ``hpx/local/barrier.hpp``
================================

Corresponds to the C++ standard library header :cppreference-header:`barrier`.

Classes
-------

- :cpp:class:`hpx::lcos::local::cpp20_barrier`

.. _public_api_header_hpx_distributed_barrier:

Header ``hpx/distributed/barrier.hpp``
======================================

Contains a distributed barrier implementation. This functionality is also
exposed through the ``hpx::distributed`` namespace. The name in
``hpx::distributed`` should be preferred.

Classes
-------

- :cpp:class:`hpx::lcos::barrier`

Header ``hpx/channel.hpp``
==========================

This header includes :ref:`public_api_header_hpx_local_channel` and
ref:`public_api_header_hpx_distributed_channel`.

.. _public_api_header_hpx_local_channel:

Header ``hpx/local/channel.hpp``
================================

Contains a local channel implementation.

Classes
-------

- :cpp:class:`hpx::lcos::local::channel`

.. _public_api_header_hpx_distributed_channel:

Header ``hpx/distributed/channel.hpp``
======================================

Contains a distributed channel implementation. This functionality is also
exposed through the ``hpx::distributed`` namespace. The name in
``hpx::distributed`` should be preferred.

Classes
-------

- :cpp:class:`hpx::lcos::channel`

Header ``hpx/chrono.hpp``
=========================

Corresponds to the C++ standard library header :cppreference-header:`chrono`.
The following replacements and extensions are provided compared to
:cppreference-header:`chrono`:

Classes
-------

- :cpp:class:`hpx::util::high_resolution_clock`
- :cpp:class:`hpx::util::high_resolution_timer`

Header ``hpx/condition_variable.hpp``
=====================================

Corresponds to the C++ standard library header
:cppreference-header:`condition_variable`.

Classes
-------

- :cpp:class:`hpx::lcos::local::condition_variable`
- :cpp:class:`hpx::lcos::local::condition_variable_any`
- :cpp:class:`hpx::lcos::local::cv_status`

Header ``hpx/exception.hpp``
============================

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

Corresponds to the C++ standard library header :cppreference-header:`execution`.
See :ref:`parallel`, :ref:`parallel_algorithms` and :ref:`executor_parameters`
for more information about execution policies and executor parameters.

.. note::

   These names are also available in the ``hpx::execution`` namespace, but not
   in the top-level ``hpx`` namespace.

Constants
---------

- :cpp:var:`hpx::parallel::execution::seq`
- :cpp:var:`hpx::parallel::execution::par`
- :cpp:var:`hpx::parallel::execution::par_unseq`
- :cpp:var:`hpx::parallel::execution::task`

Classes
-------

- :cpp:class:`hpx::parallel::execution::sequenced_policy`
- :cpp:class:`hpx::parallel::execution::parallel_policy`
- :cpp:class:`hpx::parallel::execution::parallel_unsequenced_policy`
- :cpp:class:`hpx::parallel::execution::sequenced_task_policy`
- :cpp:class:`hpx::parallel::execution::parallel_task_policy`
- :cpp:class:`hpx::parallel::execution::auto_chunk_size`
- :cpp:class:`hpx::parallel::execution::dynamic_chunk_size`
- :cpp:class:`hpx::parallel::execution::guided_chunk_size`
- :cpp:class:`hpx::parallel::execution::persistent_auto_chunk_size`
- :cpp:class:`hpx::parallel::execution::static_chunk_size`

Header ``hpx/functional.hpp``
=============================

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

This header includes :ref:`public_api_header_hpx_local_future` and
ref:`public_api_header_hpx_distributed_future`.

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

.. literalinclude:: ../libs/include/tests/unit/api_future.hpp
   :language: c++
   :lines: 7-

.. _public_api_header_hpx_distributed_future:

Header ``hpx/distributed/future.hpp``
=====================================

Contains overloads of :cpp:func:`hpx::async`, :cpp:func:`hpx::apply`,
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
==========================

This header includes :ref:`public_api_header_hpx_local_latch` and
ref:`public_api_header_hpx_distributed_latch`.

.. _public_api_header_hpx_local_latch:

Header ``hpx/local/latch.hpp``
================================

Corresponds to the C++ standard library header :cppreference-header:`latch`.

Classes
-------

- :cpp:class:`hpx::lcos::local::cpp20_latch`

.. _public_api_header_hpx_distributed_latch:

Header ``hpx/distributed/latch.hpp``
======================================

Contains a distributed latch implementation. This functionality is also exposed
through the ``hpx::distributed`` namespace. The name in ``hpx::distributed``
should be preferred.

Classes
-------

- :cpp:class:`hpx::lcos::latch`

Header ``hpx/mutex.hpp``
========================

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

Corresponds to the C++ standard library header :cppreference-header:`memory`. It
contains parallel versions of the copy, fill, move, and construct helper
functions in :cppreference-header:`memory`. See :ref:`parallel_algorithms` for
more information about the parallel algorithms.

Functions
---------

- :cpp:func:`hpx::parallel::v1::uninitialized_copy`
- :cpp:func:`hpx::parallel::v1::uninitialized_copy_n`
- :cpp:func:`hpx::parallel::v1::uninitialized_default_construct`
- :cpp:func:`hpx::parallel::v1::uninitialized_default_construct_n`
- :cpp:func:`hpx::parallel::v1::uninitialized_fill`
- :cpp:func:`hpx::parallel::v1::uninitialized_fill_n`
- :cpp:func:`hpx::parallel::v1::uninitialized_move`
- :cpp:func:`hpx::parallel::v1::uninitialized_move_n`
- :cpp:func:`hpx::parallel::v1::uninitialized_value_construct`
- :cpp:func:`hpx::parallel::v1::uninitialized_value_construct_n`

Header ``hpx/numeric.hpp``
==========================

Corresponds to the C++ standard library header :cppreference-header:`numeric`.
See :ref:`parallel_algorithms` for more information about the parallel
algorithms.

Functions
---------

- :cpp:func:`hpx::parallel::v1::adjacent_difference`
- :cpp:func:`hpx::parallel::v1::exclusive_scan`
- :cpp:func:`hpx::parallel::v1::inclusive_scan`
- :cpp:func:`hpx::parallel::v1::transform_exclusive_scan`
- :cpp:func:`hpx::parallel::v1::transform_inclusive_scan`
- :cpp:func:`hpx::parallel::v1::transform_reduce`

Header ``hpx/optional.hpp``
===========================

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

This header includes :ref:`public_api_header_hpx_local_runtime` and
ref:`public_api_header_hpx_distributed_runtime`.

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

.. _public_api_header_hpx_distributed_runtime:

Header ``hpx/distributed/runtime.hpp``
======================================

This header contains functions for accessing distributed runtime information.

Functions
---------

- :cpp:func:`hpx::find_root_locality`
- :cpp:func:`hpx::find_all_localities`
- :cpp:func:`hpx::find_remote_localities`
- :cpp:func:`hpx::find_locality`
- :cpp:func:`hpx::get_colocation_id`
- :cpp:func:`hpx::get_locality_id`

Header ``hpx/system_error.hpp``
===============================

Corresponds to the C++ standard library header
:cppreference-header:`system_error`.

Classes
-------

- :cpp:class:`hpx::error_code`

Header ``hpx/task_block.hpp``
=============================

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

Corresponds to the C++ standard library header
:cppreference-header:`semaphore`.

Classes
-------

- :cpp:class:`hpx::lcos::local::cpp20_binary_semaphore`
- :cpp:class:`hpx::lcos::local::cpp20_counting_semaphore`

Header ``hpx/shared_mutex.hpp``
===============================

Corresponds to the C++ standard library header
:cppreference-header:`shared_mutex`.

Classes
-------

- :cpp:class:`hpx::lcos::local::shared_mutex`

Header ``hpx/stop_token.hpp``
=============================

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

Corresponds to the C++ standard library header :cppreference-header:`tuple`.
:cpp:class:`hpx::util::tuple` can be used in CUDA device code, unlike
``std::tuple``.

Constants
---------

- :cpp:var:`hpx::util::ignore`

Classes
-------

- :cpp:struct:`hpx::util::tuple`
- :cpp:struct:`hpx::util::tuple_size`
- :cpp:struct:`hpx::util::tuple_element`

Functions
---------

- :cpp:func:`hpx::util::make_tuple`
- :cpp:func:`hpx::util::tie`
- :cpp:func:`hpx::util::forward_as_tuple`
- :cpp:func:`hpx::util::tuple_cat`
- :cpp:func:`hpx::util::get`

Header ``hpx/type_traits.hpp``
==============================

Corresponds to the C++ standard library header
:cppreference-header:`type_traits`. Provides
:cpp:class:`hpx::util::invoke_result` as a replacement for
``std::invoke_result``.

Classes
-------

- :cpp:struct:`hpx::util::invoke_result`

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
