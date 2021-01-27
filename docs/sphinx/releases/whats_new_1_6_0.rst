..
    Copyright (C) 2020-2021 ETH Zurich
    Copyright (C) 2007-2020 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_1_6_0:

==================
|hpx| V1.6.0 (TBD)
==================

General changes
===============

This release continues the focus on C++20 conformance with multiple new
algorithms adapted to be C++20 conformant and becoming customization point
objects (CPOs). We have also added experimental support for HIP, allowing
previous CUDA features to now be compiled with hipcc and run on AMD GPUs.

* The following algorithms have been adapted to be C++20 conformant:
  ``adjacent_find``, ``includes``, ``inplace_merge``, ``is_heap``,
  ``is_heap_until``, ``is_partitioned``, ``is_sorted``, ``is_sorted_until``,
  ``merge``, ``set_difference``, ``set_intersection``,
  ``set_symmetric_difference``, ``set_union``.
* Experimental HIP support can be used by compiling |hpx| with ``hipcc``. All
  CUDA functionality in |hpx| can now be used with HIP. The HIP functionality is
  for the time being exposed through the same API as the CUDA functionality,
  i.e. no changes are required in user code. The CUDA, and now HIP,
  functionality is in the ``hpx::cuda`` namespace.
* We have added ``partial_sort`` based on Francisco Tapia's implementation.
* ``hpx::init`` and ``hpx::start`` gained new overloads taking an
  ``hpx::init_params`` struct in 1.5.0. All overloads not taking an
  ``hpx::init_params`` are now deprecated.
* We have added an experimental ``fork_join_executor``. This executor can be
  used for OpenMP-style fork-join parallelism, where the latency of a parallel
  region is important for performance.
* The ``parallel_executor`` now uses a hierarchical spawning scheme for bulk
  execution, which improves data locality and performance.
* ``hpx::dataflow`` can now be used with executors that inject additional
  parameters into the call of the user-provided function.
* We have added experimental support for properties as proposed in |p2220|_.
  Currently the only supported property is the scheduling hint on
  ``parallel_executor``.
* In moving functionality to new namespaces, old names have been deprecated.  A
  deprecation warning will be issued if you are using deprecated functionality,
  with instructions on how to correct or ignore the warning.
* We have removed all support for C and Fortran from our build system.
* We have further reduced the use of Boost types within |hpx|
  (``boost::system::error_code`` and ``boost::detail::spinlock``).
* We have enabled more warnings in our CI builds (unused variables and unused
  typedefs).

Breaking changes
================

* hpxMP support has been completely removed.
* The ``verbs`` parcelport has been removed.
* The following compatibility options have been disabled by default:
  ``HPX_WITH_ACTION_BASE_COMPATIBILITY``,
  ``HPX_WITH_REGISTER_THREAD_COMPATIBILITY``,
  ``HPX_WITH_PROMISE_ALIAS_COMPATIBILITY``,
  ``HPX_WITH_UNSCOPED_ENUM_COMPATIBILITY``,
  ``HPX_PROGRAM_OPTIONS_WITH_BOOST_PROGRAM_OPTIONS_COMPATIBILITY``,
  ``HPX_WITH_EMBEDDED_THREAD_POOLS_COMPATIBILITY``,
  ``HPX_WITH_THREAD_POOL_OS_EXECUTOR_COMPATIBILITY``,
  ``HPX_WITH_THREAD_EXECUTORS_COMPATIBILITY``,
  ``HPX_THREAD_AWARE_TIMER_COMPATIBILITY``,
  ``HPX_WITH_POOL_EXECUTOR_COMPATIBILITY``. Unless noted here, the above
  functionalities do not come with replacements. Unscoped enumerations have been
  replaced by scoped enumerations. Previously deprecated unscoped enumerations
  are disabled by ``HPX_WITH_UNSCOPED_ENUM_COMPATIBILITY``. Newly deprecated
  unscoped enumerations have been given deprecation warnings and replaced by
  scoped enumerations. ``hpx::promise`` has been replaced with
  ``hpx::distributed::promise``. ``hpx::program_options`` is a drop-in
  replacement for ``boost::program_options``.
  ``hpx::execution::parallel_executor`` now has constructors which take a thread
  pool, covering the use case of ``hpx::threads::executors::pool_executor``. A
  pool can be supplied with ``hpx::resource::get_thread_pool``.

Closed issues
=============

Closed pull requests
====================
