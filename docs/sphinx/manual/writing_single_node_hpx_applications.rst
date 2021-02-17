..
    Copyright (C) 2012 Bryce Adelstein-Lelbach
    Copyright (C) 2007-2016 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _writing_single_node_hpx_applications:

======================================
Writing single-node |hpx| applications
======================================

|hpx| is a C++ Standard Library for Concurrency and Parallelism. This means that
it implements all of the corresponding facilities as defined by the C++
Standard. Additionally, |hpx| implements functionalities proposed as part
of the ongoing C++ standardization process. This section focuses on the features
available in |hpx| for parallel and concurrent computation on a single node,
although many of the features presented here are also implemented to work in the
distributed case.

.. _lcos:

Using :term:`LCO`\ s
====================

:term:`Lightweight Control Object`\ s (LCOs) provide synchronization for |hpx| applications. Most
of them are familiar from other frameworks, but a few of them work in slightly
different ways adapted to |hpx|. The following synchronization objects are available in |hpx|:

#. ``future``

#. ``queue``

#. ``object_semaphore``

#. ``barrier``

Channels
--------

Channels combine communication (the exchange of a value) with synchronization
(guaranteeing that two calculations (tasks) are in a known state). A channel can
transport any number of values of a given type from a sender to a receiver:

.. literalinclude:: ../../examples/quickstart/local_channel_docs.cpp
   :language: c++
   :start-after: //[local_channel_minimal
   :end-before: //]

Channels can be handed to another thread (or in case of channel components, to
other localities), thus establishing a communication channel between two
independent places in the program:

.. literalinclude:: ../../examples/quickstart/local_channel_docs.cpp
   :language: c++
   :start-after: //[local_channel_send_receive
   :end-before: //]

Note how :cpp:member:`hpx::lcos::local::channel::get` without any arguments
returns a future which is ready when a value has been set on the channel. The
launch policy ``hpx::launch::sync`` can be used to make
:cpp:member:`hpx::lcos::local::channel::get` block until a value is set and
return the value directly.

A channel component is created on one :term:`locality` and can be sent to
another :term:`locality` using an action. This example also demonstrates how a
channel can be used as a range of values:

.. literalinclude:: ../../examples/quickstart/channel_docs.cpp
   :language: c++
   :start-after: //[channel
   :end-before: //]

Composable guards
-----------------

Composable guards operate in a manner similar to locks, but are applied only to
asynchronous functions. The guard (or guards) is automatically locked at the
beginning of a specified task and automatically unlocked at the end. Because
guards are never added to an existing task's execution context, the calling of
guards is freely composable and can never deadlock.

To call an application with a single guard, simply declare the guard and call
run_guarded() with a function (task)::

     hpx::lcos::local::guard gu;
     run_guarded(gu,task);

If a single method needs to run with multiple guards, use a guard set::

     boost::shared<hpx::lcos::local::guard> gu1(new hpx::lcos::local::guard());
     boost::shared<hpx::lcos::local::guard> gu2(new hpx::lcos::local::guard());
     gs.add(*gu1);
     gs.add(*gu2);
     run_guarded(gs,task);

Guards use two atomic operations (which are not called repeatedly) to manage
what they do, so overhead should be extremely low. The following guards are available in |hpx|:

#. ``conditional_trigger``

#. ``counting_semaphore``

#. ``dataflow``

#. ``event``

#. ``mutex``

#. ``once``

#. ``recursive_mutex``

#. ``spinlock``

#. ``spinlock_no_backoff``

#. ``trigger``

.. _extend_futures:

Extended facilities for futures
===============================

Concurrency is about both decomposing and composing the program from the parts
that work well individually and together. It is in the composition of connected
and multicore components where today's C++ libraries are still lacking.

The functionality of ``std::future`` offers a partial solution. It allows for
the separation of the initiation of an operation and the act of waiting for its
result; however, the act of waiting is synchronous. In communication-intensive
code this act of waiting can be unpredictable, inefficient and simply
frustrating. The example below illustrates a possible synchronous wait using
futures::

    #include <future>
    using namespace std;
    int main()
    {
        future<int> f = async([]() { return 123; });
        int result = f.get(); // might block
    }

For this reason, |hpx| implements a set of extensions to ``std::future`` (as
proposed by __cpp11_n4107__). This proposal introduces the following key
asynchronous operations to ``hpx::future``, ``hpx::shared_future`` and
``hpx::async``, which enhance and enrich these facilities.

.. list-table:: Facilities extending ``std::future``

   * * Facility
     * Description
   * * ``hpx::future::then``
     * In asynchronous programming, it is very common for one asynchronous
       operation, on completion, to invoke a second operation and pass data to
       it. The current C++ standard does not allow one to register a
       continuation to a future. With ``then``, instead of waiting for the result,
       a continuation is "attached" to the asynchronous operation, which is
       invoked when the result is ready. Continuations registered using then
       function will help to avoid blocking waits or wasting threads on polling,
       greatly improving the responsiveness and scalability of an application.
   * * unwrapping constructor for ``hpx::future``
     * In some scenarios, you might want to create a future that returns another
       future, resulting in nested futures. Although it is possible to write
       code to unwrap the outer future and retrieve the nested future and its
       result, such code is not easy to write because users must handle exceptions
       and it may cause a blocking call. Unwrapping can allow users to mitigate
       this problem by doing an asynchronous call to unwrap the outermost
       future.
   * * ``hpx::future::is_ready``
     * There are often situations where a ``get()`` call on a future may not be
       a blocking call, or is only a blocking call under certain circumstances.
       This function gives the ability to test for early completion and allows
       us to avoid associating a continuation, which needs to be scheduled with
       some non-trivial overhead and near-certain loss of cache efficiency.
   * * ``hpx::make_ready_future``
     * Some functions may know the value at the point of construction. In these
       cases the value is immediately available, but needs to be returned as a
       future. By using ``hpx::make_ready_future`` a future can be created that
       holds a pre-computed result in its shared state. In the current standard
       it is non-trivial to create a future directly from a value. First a
       promise must be created, then the promise is set, and lastly the future
       is retrieved from the promise. This can now be done with one operation.

The standard also omits the ability to compose multiple futures. This is a
common pattern that is ubiquitous in other asynchronous frameworks and is
absolutely necessary in order to make C++ a powerful asynchronous programming
language. Not including these functions is synonymous to Boolean algebra without
AND/OR.

In addition to the extensions proposed by |cpp11_n4107|_, |hpx| adds functions
allowing users to compose several futures in a more flexible way.

.. list-table:: Facilities for composing ``hpx::future``\ s

   * * Facility
     * Description
     * Comment
   * * :cpp:func:`hpx::when_any`, :cpp:func:`hpx::when_any_n`
     * Asynchronously wait for at least one of multiple future or shared_future
       objects to finish.
     * |cpp11_n4107|_, ``..._n`` versions are |hpx| only
   * * :cpp:func:`hpx::wait_any`, :cpp:func:`hpx::wait_any_n`
     * Synchronously wait for at least one of multiple future or shared_future
       objects to finish.
     * |hpx| only
   * * :cpp:func:`hpx::when_all`, :cpp:func:`hpx::when_all_n`
     * Asynchronously wait for all future and shared_future objects to finish.
     * |cpp11_n4107|_, ``..._n`` versions are |hpx| only
   * * :cpp:func:`hpx::wait_all`, :cpp:func:`hpx::wait_all_n`
     * Synchronously wait for all future and shared_future objects to finish.
     * |hpx| only
   * * :cpp:func:`hpx::when_some`, :cpp:func:`hpx::when_some_n`
     * Asynchronously wait for multiple future and shared_future objects to
       finish.
     * |hpx| only
   * * :cpp:func:`hpx::wait_some`, :cpp:func:`hpx::wait_some_n`
     * Synchronously wait for multiple future and shared_future objects to
       finish.
     * |hpx| only
   * * :cpp:func:`hpx::when_each`
     * Asynchronously wait for multiple future and shared_future objects to
       finish and call a function for each of the future objects as soon as it
       becomes ready.
     * |hpx| only
   * * :cpp:func:`hpx::wait_each`, :cpp:func:`hpx::wait_each_n`
     * Synchronously wait for multiple future and shared_future objects to
       finish and call a function for each of the future objects as soon as it
       becomes ready.
     * |hpx| only

.. _parallel:

High level parallel facilities
==============================

In preparation for the upcoming C++ Standards, there are currently several proposals
targeting different facilities supporting parallel programming. |hpx| implements
(and extends) some of those proposals. This is well aligned with our strategy to
align the APIs exposed from |hpx| with current and future C++ Standards.

At this point, |hpx| implements several of the C++ Standardization working
papers, most notably |cpp11_n4104|_ (Working Draft, Technical Specification for
C++ Extensions for Parallelism), |cpp11_n4088|_ (Task Blocks), and
|cpp11_n4406|_ (Parallel Algorithms Need Executors).

.. _parallel_algorithms:

Using parallel algorithms
-------------------------

.. |sequenced_execution_policy| replace:: :cpp:class:`hpx::execution::sequenced_policy`
.. |sequenced_task_execution_policy| replace:: :cpp:class:`hpx::execution::sequenced_task_policy`
.. |parallel_execution_policy| replace:: :cpp:class:`hpx::execution::parallel_policy`
.. |parallel_unsequenced_execution_policy| replace:: :cpp:class:`hpx::execution::parallel_unsequenced_policy`
.. |parallel_task_execution_policy| replace:: :cpp:class:`hpx::execution::parallel_task_policy`
.. |execution_policy| replace:: :cpp:class:`hpx::parallel::v1::execution_policy`
.. |exception_list| replace:: :cpp:class:`hpx::exception_list`
.. |par_for_each| replace:: :cpp:class:`hpx::parallel::v1::for_each`

A parallel algorithm is a function template described by this document
which is declared in the (inline) namespace ``hpx::parallel::v1``.

.. note::

   For compilers that do not support inline namespaces, all of the ``namespace
   v1`` is imported into the namespace ``hpx::parallel``. The effect is similar
   to what inline namespaces would do, namely all names defined in
   ``hpx::parallel::v1`` are accessible from the namespace ``hpx::parallel`` as
   well.

All parallel algorithms are very similar in semantics to their sequential
counterparts (as defined in the ``namespace std``) with an additional formal
template parameter named ``ExecutionPolicy``. The execution policy is generally
passed as the first argument to any of the parallel algorithms and describes the
manner in which the execution of these algorithms may be parallelized and the
manner in which they apply user-provided function objects.

The applications of function objects in parallel algorithms invoked with an
execution policy object of type |sequenced_execution_policy| or
|sequenced_task_execution_policy| execute in sequential order. For
|sequenced_execution_policy| the execution happens in the calling thread.

The applications of function objects in parallel algorithms invoked with an
execution policy object of type |parallel_execution_policy| or
|parallel_task_execution_policy| are permitted to execute in an unordered
fashion in unspecified threads, and are indeterminately sequenced within each
thread.

.. important::

   It is the caller's responsibility to ensure correctness, such as making sure that the
   invocation does not introduce data races or deadlocks.

The applications of function objects in parallel algorithms invoked with an
execution policy of type |parallel_unsequenced_execution_policy| is, in |hpx|,
equivalent to the use of the execution policy |parallel_execution_policy|.

Algorithms invoked with an execution policy object of type |execution_policy|
execute internally as if invoked with the contained execution policy object. No
exception is thrown when an |execution_policy| contains an execution policy of
type |sequenced_task_execution_policy| or |parallel_task_execution_policy|
(which normally turn the algorithm into its asynchronous version). In this case
the execution is semantically equivalent to the case of passing a
|sequenced_execution_policy| or |parallel_execution_policy| contained in the
|execution_policy| object respectively.

Parallel exceptions
-------------------

During the execution of a standard parallel algorithm, if temporary memory
resources are required by any of the algorithms and no memory is available, the
algorithm throws a ``std::bad_alloc`` exception.

During the execution of any of the parallel algorithms, if the application of a
function object terminates with an uncaught exception, the behavior of the
program is determined by the type of execution policy used to invoke the
algorithm:

* If the execution policy object is of type
  |parallel_unsequenced_execution_policy|, :cpp:func:`hpx::terminate` shall
  be called.
* If the execution policy object is of type |sequenced_execution_policy|,
  |sequenced_task_execution_policy|, |parallel_execution_policy|, or
  |parallel_task_execution_policy|, the execution of the algorithm terminates
  with an |exception_list| exception. All uncaught exceptions thrown during the
  application of user-provided function objects shall be contained in the
  |exception_list|.

For example, the number of invocations of the user-provided function object in
for_each is unspecified. When |par_for_each| is executed sequentially, only one
exception will be contained in the |exception_list| object.

These guarantees imply that, unless the algorithm has failed to allocate memory
and terminated with ``std::bad_alloc``, all exceptions thrown during the
execution of the algorithm are communicated to the caller. It is unspecified
whether an algorithm implementation will "forge ahead" after encountering and
capturing a user exception.

The algorithm may terminate with the ``std::bad_alloc`` exception even if one or
more user-provided function objects have terminated with an exception. For
example, this can happen when an algorithm fails to allocate memory while
creating or adding elements to the |exception_list| object.

Parallel algorithms
-------------------

|hpx| provides implementations of the following parallel algorithms:

.. list-table:: Non-modifying parallel algorithms (in header: ``<hpx/algorithm.hpp>``)

   * * Name
     * Description
     * In header
     * Algorithm page at cppreference.com
   * * :cpp:func:`hpx::adjacent_find`
     * Computes the differences between adjacent elements in a range.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`adjacent_find`
   * * :cpp:func:`hpx::all_of`
     * Checks if a predicate is ``true`` for all of the elements in a range.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`all_any_none_of`
   * * :cpp:func:`hpx::any_of`
     * Checks if a predicate is ``true`` for any of the elements in a range.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`all_any_none_of`
   * * :cpp:func:`hpx::count`
     * Returns the number of elements equal to a given value.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`count`
   * * :cpp:func:`hpx::count_if`
     * Returns the number of elements satisfying a specific criteria.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`count_if`
   * * :cpp:func:`hpx::equal`
     * Determines if two sets of elements are the same.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`equal`
   * * :cpp:func:`hpx::find`
     * Finds the first element equal to a given value.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`find`
   * * :cpp:func:`hpx::find_end`
     * Finds the last sequence of elements in a certain range.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`find_end`
   * * :cpp:func:`hpx::find_first_of`
     * Searches for any one of a set of elements.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`find_first_of`
   * * :cpp:func:`hpx::find_if`
     * Finds the first element satisfying a specific criteria.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`find_if`
   * * :cpp:func:`hpx::find_if_not`
     * Finds the first element not satisfying a specific criteria.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`find_if_not`
   * * :cpp:func:`hpx::for_each`
     * Applies a function to a range of elements.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`for_each`
   * * :cpp:func:`hpx::for_each_n`
     * Applies a function to a number of elements.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`for_each_n`
   * * :cpp:func:`hpx::parallel::v1::lexicographical_compare`
     * Checks if a range of values is lexicographically less than another range of values.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`lexicographical_compare`
   * * :cpp:func:`hpx::parallel::v1::mismatch`
     * Finds the first position where two ranges differ.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`mismatch`
   * * :cpp:func:`hpx::none_of`
     * Checks if a predicate is ``true`` for none of the elements in a range.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`all_any_none_of`
   * * :cpp:func:`hpx::search`
     * Searches for a range of elements.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`search`
   * * :cpp:func:`hpx::search_n`
     * Searches for a number consecutive copies of an element in a range.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`search_n`

.. list-table:: Modifying parallel algorithms (In Header: `<hpx/algorithm.hpp>`)

   * * Name
     * Description
     * In header
     * Algorithm page at cppreference.com
   * * :cpp:func:`hpx::copy`
     * Copies a range of elements to a new location.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`exclusive_scan`
   * * :cpp:func:`hpx::copy_n`
     * Copies a number of elements to a new location.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`copy_n`
   * * :cpp:func:`hpx::copy_if`
     * Copies the elements from a range to a new location for which the given predicate is ``true``
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`copy`
   * * :cpp:func:`hpx::move`
     * Moves a range of elements to a new location.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`move`
   * * :cpp:func:`hpx::fill`
     * Assigns a range of elements a certain value.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`fill`
   * * :cpp:func:`hpx::fill_n`
     * Assigns a value to a number of elements.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`fill_n`
   * * :cpp:func:`hpx::generate`
     * Saves the result of a function in a range.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`generate`
   * * :cpp:func:`hpx::generate_n`
     * Saves the result of N applications of a function.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`generate_n`
   * * :cpp:func:`hpx::remove`
     * Removes the elements from a range that are equal to the given value.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`remove`
   * * :cpp:func:`hpx::remove_if`
     * Removes the elements from a range that are equal to the given predicate is ``false``
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`remove`
   * * :cpp:func:`hpx::remove_copy`
     * Copies the elements from a range to a new location that are not equal to the given value.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`remove_copy`
   * * :cpp:func:`hpx::remove_copy_if`
     * Copies the elements from a range to a new location for which the given predicate is ``false``
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`remove_copy`
   * * :cpp:func:`hpx::parallel::v1::replace`
     * Replaces all values satisfying specific criteria with another value.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`replace`
   * * :cpp:func:`hpx::parallel::v1::replace_if`
     * Replaces all values satisfying specific criteria with another value.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`replace`
   * * :cpp:func:`hpx::parallel::v1::replace_copy`
     * Copies a range, replacing elements satisfying specific criteria with another value.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`replace_copy`
   * * :cpp:func:`hpx::parallel::v1::replace_copy_if`
     * Copies a range, replacing elements satisfying specific criteria with another value.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`replace_copy`
   * * :cpp:func:`hpx::parallel::v1::reverse`
     * Reverses the order elements in a range.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`reverse`
   * * :cpp:func:`hpx::parallel::v1::reverse_copy`
     * Creates a copy of a range that is reversed.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`reverse_copy`
   * * :cpp:func:`hpx::parallel::v1::rotate`
     * Rotates the order of elements in a range.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`rotate`
   * * :cpp:func:`hpx::parallel::v1::rotate_copy`
     * Copies and rotates a range of elements.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`rotate_copy`
   * * :cpp:func:`hpx::parallel::v1::swap_ranges`
     * Swaps two ranges of elements.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`swap_ranges`
   * * :cpp:func:`hpx::transform`
     * Applies a function to a range of elements.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`transform`
   * * :cpp:func:`hpx::parallel::v1::unique_copy`
     * Eliminates all but the first element from every consecutive group of equivalent elements from a range.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`unique`
   * * :cpp:func:`hpx::parallel::v1::unique_copy`
     * Eliminates all but the first element from every consecutive group of equivalent elements from a range.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`unique_copy`

.. list-table:: Set operations on sorted sequences (In Header: `<hpx/algorithm.hpp>`)

   * * Name
     * Description
     * In header
     * Algorithm page at cppreference.com
   * * :cpp:func:`hpx::merge`
     * Merges two sorted ranges.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`merge`
   * * :cpp:func:`hpx::inplace_merge`
     * Merges two ordered ranges in-place.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`inplace_merge`
   * * :cpp:func:`hpx::includes`
     * Returns true if one set is a subset of another.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`includes`
   * * :cpp:func:`hpx::set_difference`
     * Computes the difference between two sets.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`set_difference`
   * * :cpp:func:`hpx::set_intersection`
     * Computes the intersection of two sets.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`set_intersection`
   * * :cpp:func:`hpx::set_symmetric_difference`
     * Computes the symmetric difference between two sets.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`set_symmetric_difference`
   * * :cpp:func:`hpx::set_union`
     * Computes the union of two sets.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`set_union`

.. list-table:: Heap operations (In Header: <hpx/algorithm.hpp>)

   * * Name
     * Description
     * In header
     * Algorithm page at cppreference.com
   * * :cpp:func:`hpx::is_heap`
     * Returns ``true`` if the range is max heap.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`is_heap`
   * * :cpp:func:`hpx::is_heap_until`
     * Returns the first element that breaks a max heap.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`is_heap_until`
   * * :cpp:func:`hpx::make_heap`
     * Constructs a max heap in the range [first, last).
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`make_heap`

.. list-table:: Minimum/maximum operations (In Header: <hpx/algorithm.hpp>)

   * * Name
     * Description
     * In header
     * Algorithm page at cppreference.com
   * * :cpp:func:`hpx::parallel::v1::max_element`
     * Returns the largest element in a range.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`max_element`
   * * :cpp:func:`hpx::parallel::v1::min_element`
     * Returns the smallest element in a range.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`min_element`
   * * :cpp:func:`hpx::parallel::v1::minmax_element`
     * Returns the smallest and the largest element in a range.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`minmax_element`

.. list-table:: Partitioning Operations (In Header: `<hpx/algorithm.hpp>`)

   * * Name
     * Description
     * In header
     * Algorithm page at cppreference.com
   * * :cpp:func:`hpx::is_partitioned`
     * Returns ``true`` if each true element for a predicate precedes the false elements in a range.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`is_partitioned`
   * * :cpp:func:`hpx::parallel::v1::partition`
     * Divides elements into two groups without preserving their relative order.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`partition`
   * * :cpp:func:`hpx::parallel::v1::partition_copy`
     * Copies a range dividing the elements into two groups.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`partition_copy`
   * * :cpp:func:`hpx::parallel::v1::stable_partition`
     * Divides elements into two groups while preserving their relative order.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`stable_partition`

.. list-table:: Sorting Operations (In Header: `<hpx/algorithm.hpp>`)

   * * Name
     * Description
     * In header
     * Algorithm page at cppreference.com
   * * :cpp:func:`hpx::is_sorted`
     * Returns ``true`` if each element in a range is sorted.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`is_sorted`
   * * :cpp:func:`hpx::is_sorted_until`
     * Returns the first unsorted element.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`is_sorted_until`
   * * :cpp:func:`hpx::parallel::v1::sort`
     * Sorts the elements in a range.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`sort`
   * * :cpp:func:`hpx::parallel::v1::stable_sort`
     * Sorts the elements in a range, maintain sequence of equal elements.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`stable_sort`
   * * :cpp:func:`hpx::partial_sort`
     * Sorts the first elements in a range.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`partial_sort`
   * * :cpp:func:`hpx::parallel::v1::sort_by_key`
     * Sorts one range of data using keys supplied in another range.
     * ``<hpx/algorithm.hpp>``
     *


.. list-table:: Numeric Parallel Algorithms (In Header: `<hpx/numeric.hpp>`)

   * * Name
     * Description
     * In header
     * Algorithm page at cppreference.com
   * * :cpp:func:`hpx::parallel::v1::adjacent_difference`
     * Calculates the difference between each element in an input range and the preceding element.
     * ``<hpx/numeric.hpp>``
     * :cppreference-algorithm:`adjacent_difference`
   * * :cpp:func:`hpx::parallel::v1::exclusive_scan`
     * Does an exclusive parallel scan over a range of elements.
     * ``<hpx/numeric.hpp>``
     * :cppreference-algorithm:`exclusive_scan`
   * * :cpp:func:`hpx::reduce`
     * Sums up a range of elements.
     * ``<hpx/numeric.hpp>``
     * :cppreference-algorithm:`reduce`
   * * :cpp:func:`hpx::parallel::v1::inclusive_scan`
     * Does an inclusive parallel scan over a range of elements.
     * ``<hpx/algorithm.hpp>``
     * :cppreference-algorithm:`inclusive_scan`
   * * :cpp:func:`hpx::parallel::v1::reduce_by_key`
     * Performs an inclusive scan on consecutive elements with matching keys,
       with a reduction to output only the final sum for each key. The key
       sequence ``{1,1,1,2,3,3,3,3,1}`` and value sequence
       ``{2,3,4,5,6,7,8,9,10}`` would be reduced to ``keys={1,2,3,1}``,
       ``values={9,5,30,10}``.
     * ``<hpx/numeric.hpp>``
     *
   * * :cpp:func:`hpx::transform_reduce`
     * Sums up a range of elements after applying a function. Also, accumulates the inner products of two input ranges.
     * ``<hpx/numeric.hpp>``
     * :cppreference-algorithm:`transform_reduce`
   * * :cpp:func:`hpx::parallel::v1::transform_inclusive_scan`
     * Does an inclusive parallel scan over a range of elements after applying a function.
     * ``<hpx/numeric.hpp>``
     * :cppreference-algorithm:`transform_inclusive_scan`
   * * :cpp:func:`hpx::parallel::v1::transform_exclusive_scan`
     * Does an exclusive parallel scan over a range of elements after applying a function.
     * ``<hpx/numeric.hpp>``
     * :cppreference-algorithm:`transform_exclusive_scan`

.. list-table:: Dynamic Memory Management (In Header: `<hpx/memory.hpp>`)

   * * Name
     * Description
     * In header
     * Algorithm page at cppreference.com
   * * :cpp:func:`hpx::destroy`
     * Destroys a range of objects.
     * ``<hpx/memory.hpp>``
     * :cppreference-memory:`destroy`
   * * :cpp:func:`hpx::destroy_n`
     * Destroys a range of objects.
     * ``<hpx/memory.hpp>``
     * :cppreference-memory:`destroy_n`
   * * :cpp:func:`hpx::parallel::v1::uninitialized_copy`
     * Copies a range of objects to an uninitialized area of memory.
     * ``<hpx/memory.hpp>``
     * :cppreference-memory:`uninitialized_copy`
   * * :cpp:func:`hpx::parallel::v1::uninitialized_copy_n`
     * Copies a number of objects to an uninitialized area of memory.
     * ``<hpx/memory.hpp>``
     * :cppreference-memory:`uninitialized_copy_n`
   * * :cpp:func:`hpx::parallel::v1::uninitialized_default_construct`
     * Copies a range of objects to an uninitialized area of memory.
     * ``<hpx/memory.hpp>``
     * :cppreference-memory:`uninitialized_default_construct`
   * * :cpp:func:`hpx::parallel::v1::uninitialized_default_construct_n`
     * Copies a number of objects to an uninitialized area of memory.
     * ``<hpx/memory.hpp>``
     * :cppreference-memory:`uninitialized_default_construct_n`
   * * :cpp:func:`hpx::parallel::v1::uninitialized_fill`
     * Copies an object to an uninitialized area of memory.
     * ``<hpx/memory.hpp>``
     * :cppreference-memory:`uninitialized_fill`
   * * :cpp:func:`hpx::parallel::v1::uninitialized_fill_n`
     * Copies an object to an uninitialized area of memory.
     * ``<hpx/memory.hpp>``
     * :cppreference-memory:`uninitialized_fill_n`
   * * :cpp:func:`hpx::parallel::v1::uninitialized_move`
     * Moves a range of objects to an uninitialized area of memory.
     * ``<hpx/memory.hpp>``
     * :cppreference-memory:`uninitialized_move`
   * * :cpp:func:`hpx::parallel::v1::uninitialized_move_n`
     * Moves a number of objects to an uninitialized area of memory.
     * ``<hpx/memory.hpp>``
     * :cppreference-memory:`uninitialized_move_n`
   * * :cpp:func:`hpx::parallel::v1::uninitialized_value_construct`
     * Constructs objects in an uninitialized area of memory.
     * ``<hpx/memory.hpp>``
     * :cppreference-memory:`uninitialized_value_construct`
   * * :cpp:func:`hpx::parallel::v1::uninitialized_value_construct_n`
     * Constructs objects in an uninitialized area of memory.
     * ``<hpx/memory.hpp>``
     * :cppreference-memory:`uninitialized_value_construct_n`

.. list-table:: Index-based for-loops (In Header: `<hpx/algorithm.hpp>`)

   * * Name
     * Description
     * In header
   * * :cpp:func:`hpx::for_loop`
     * Implements loop functionality over a range specified by integral or iterator bounds.
     * ``<hpx/algorithm.hpp>``
   * * :cpp:func:`hpx::for_loop_strided`
     * Implements loop functionality over a range specified by integral or iterator bounds.
     * ``<hpx/algorithm.hpp>``
   * * :cpp:func:`hpx::for_loop_n`
     * Implements loop functionality over a range specified by integral or iterator bounds.
     * ``<hpx/algorithm.hpp>``
   * * :cpp:func:`hpx::for_loop_n_strided`
     * Implements loop functionality over a range specified by integral or iterator bounds.
     * ``<hpx/algorithm.hpp>``

.. _executor_parameters:

Executor parameters and executor parameter traits
-------------------------------------------------

|hpx| introduces the notion of execution parameters and execution parameter
traits. At this point, the only parameter that can be customized is the size of
the chunks of work executed on a single |hpx| thread (such as the number of loop
iterations combined to run as a single task).

An executor parameter object is responsible for exposing the calculation of the
size of the chunks scheduled. It abstracts the (potentially platform-specific)
algorithms of determining those chunk sizes.

The way executor parameters are implemented is aligned with the way executors
are implemented. All functionalities of concrete executor parameter types are
exposed and accessible through a corresponding
:cpp:class:`hpx::parallel::executor_parameter_traits` type.

With ``executor_parameter_traits``, clients access all types of executor
parameters uniformly::

    std::size_t chunk_size =
        executor_parameter_traits<my_parameter_t>::get_chunk_size(my_parameter,
            my_executor, [](){ return 0; }, num_tasks);

This call synchronously retrieves the size of a single chunk of loop iterations
(or similar) to combine for execution on a single |hpx| thread if the overall
number of tasks to schedule is given by ``num_tasks``. The lambda function
exposes a means of test-probing the execution of a single iteration for
performance measurement purposes. The execution parameter type might dynamically
determine the execution time of one or more tasks in order to calculate the
chunk size; see :cpp:class:`hpx::execution::auto_chunk_size` for an example of
this executor parameter type.

Other functions in the interface exist to discover whether an executor parameter
type should be invoked once (i.e., it returns a static chunk size; see
:cpp:class:`hpx::execution::static_chunk_size`) or whether it should be invoked
for each scheduled chunk of work (i.e., it returns a variable chunk size; for an
example, see :cpp:class:`hpx::execution::guided_chunk_size`).

Although this interface appears to require executor parameter type authors to
implement all different basic operations, none are required. In
practice, all operations have sensible defaults. However, some executor
parameter types will naturally specialize all operations for maximum efficiency.

|hpx|  implements the following executor parameter types:

* :cpp:class:`hpx::execution::auto_chunk_size`: Loop iterations are divided into
  pieces and then assigned to threads. The number of loop iterations combined is
  determined based on measurements of how long the execution of 1% of the
  overall number of iterations takes. This executor parameter type makes sure
  that as many loop iterations are combined as necessary to run for the amount
  of time specified.
* :cpp:class:`hpx::execution::static_chunk_size`: Loop iterations are divided
  into pieces of a given size and then assigned to threads. If the size is not
  specified, the iterations are, if possible, evenly divided contiguously among
  the threads. This executor parameters type is equivalent to OpenMP's STATIC
  scheduling directive.
* :cpp:class:`hpx::execution::dynamic_chunk_size`: Loop iterations are divided
  into pieces of a given size and then dynamically scheduled among the cores;
  when a core finishes one chunk, it is dynamically assigned another. If the
  size is not specified, the default chunk size is 1. This executor parameter
  type is equivalent to OpenMP's DYNAMIC scheduling directive.
* :cpp:class:`hpx::execution::guided_chunk_size`: Iterations are dynamically
  assigned to cores in blocks as cores request them until no blocks remain to be
  assigned. This is similar to ``dynamic_chunk_size`` except that the block size
  decreases each time a number of loop iterations is given to a thread. The size
  of the initial block is proportional to ``number_of_iterations /
  number_of_cores``. Subsequent blocks are proportional to
  ``number_of_iterations_remaining / number_of_cores``. The optional chunk size
  parameter defines the minimum block size. The default minimal chunk size is 1.
  This executor parameter type is equivalent to OpenMP's GUIDED scheduling
  directive.

.. _using_task_block:

Using task blocks
=================

The ``define_task_block``, ``run`` and the ``wait`` functions implemented based
on |cpp11_n4088|_ are based on the ``task_block`` concept that is a part of the
common subset of the |ppl|_ and the |tbb|_ libraries.

These implementations adopt a simpler syntax than exposed by those libraries---
one that is influenced by language-based concepts, such as spawn and sync from
|cilk_pp|_ and async and finish from |x10|_. They improve on existing practice in
the following ways:

* The exception handling model is simplified and more consistent with normal C++
  exceptions.
* Most violations of strict fork-join parallelism can be enforced at compile
  time (with compiler assistance, in some cases).
* The syntax allows scheduling approaches other than child stealing.

Consider an example of a parallel traversal of a tree, where a user-provided
function compute is applied to each node of the tree, returning the sum of the
results::

    template <typename Func>
    int traverse(node& n, Func && compute)
    {
        int left = 0, right = 0;
        define_task_block(
            [&](task_block<>& tr) {
                if (n.left)
                    tr.run([&] { left = traverse(*n.left, compute); });
                if (n.right)
                    tr.run([&] { right = traverse(*n.right, compute); });
            });

        return compute(n) + left + right;
    }

The example above demonstrates the use of two of the functions,
:cpp:func:`hpx::parallel::define_task_block` and the
:cpp:member:`hpx::parallel::task_block::run` member function of a
:cpp:class:`hpx::parallel::task_block`.

The ``task_block`` function delineates a region in a program code potentially
containing invocations of threads spawned by the ``run`` member function of the
``task_block`` class. The ``run`` function spawns an |hpx| thread, a unit of
work that is allowed to execute in parallel with respect to the caller. Any
parallel tasks spawned by ``run`` within the task block are joined back to a
single thread of execution at the end of the ``define_task_block``. ``run``
takes a user-provided function object ``f`` and starts it asynchronously---i.e.,
it may return before the execution of ``f`` completes. The |hpx| scheduler may
choose to run ``f`` immediately or delay running ``f`` until compute resources
become available.

A ``task_block`` can be constructed only by ``define_task_block`` because it has
no public constructors. Thus, ``run`` can be invoked directly or indirectly
only from a user-provided function passed to ``define_task_block``::

    void g();

    void f(task_block<>& tr)
    {
        tr.run(g);          // OK, invoked from within task_block in h
    }

    void h()
    {
        define_task_block(f);
    }

    int main()
    {
        task_block<> tr;    // Error: no public constructor
        tr.run(g);          // No way to call run outside of a define_task_block
        return 0;
    }

.. _task_block_extensions:

Extensions for task blocks
--------------------------

Using execution policies with task blocks
.........................................

|hpx| implements some extensions for ``task_block`` beyond the actual
standards proposal |cpp11_n4088|_. The main addition is that a ``task_block``
can be invoked with an execution policy as its first argument, very similar to
the parallel algorithms.

An execution policy is an object that expresses the requirements on the
ordering of functions invoked as a consequence of the invocation of a
task block. Enabling passing an execution policy to ``define_task_block``
gives the user control over the amount of parallelism employed by the
created ``task_block``. In the following example the use of an explicit
``par`` execution policy makes the user's intent explicit::

    template <typename Func>
    int traverse(node *n, Func&& compute)
    {
        int left = 0, right = 0;

        define_task_block(
            execution::par,                // execution::parallel_policy
            [&](task_block<>& tb) {
                if (n->left)
                    tb.run([&] { left = traverse(n->left, compute); });
                if (n->right)
                    tb.run([&] { right = traverse(n->right, compute); });
            });

        return compute(n) + left + right;
    }

This also causes the :cpp:class:`hpx::parallel::v2::task_block` object to be a
template in our implementation. The template argument is the type of the
execution policy used to create the task block. The template argument defaults
to :cpp:class:`hpx::execution::parallel_policy`.

|hpx| still supports calling :cpp:func:`hpx::parallel::v2::define_task_block`
without an explicit execution policy. In this case the task block will run using
the :cpp:class:`hpx::execution::parallel_policy`.

|hpx| also adds the ability to access the execution policy that was used to
create a given ``task_block``.

Using executors to run tasks
............................

Often, users want to be able to not only define an execution policy to use by
default for all spawned tasks inside the task block, but also to
customize the execution context for one of the tasks executed by
``task_block::run``. Adding an optionally passed executor instance to that
function enables this use case::

    template <typename Func>
    int traverse(node *n, Func&& compute)
    {
        int left = 0, right = 0;

        define_task_block(
            execution::par,                // execution::parallel_policy
            [&](auto& tb) {
                if (n->left)
                {
                    // use explicitly specified executor to run this task
                    tb.run(my_executor(), [&] { left = traverse(n->left, compute); });
                }
                if (n->right)
                {
                    // use the executor associated with the par execution policy
                    tb.run([&] { right = traverse(n->right, compute); });
                }
            });

        return compute(n) + left + right;
    }

|hpx| still supports calling :cpp:func:`hpx::parallel::v2::task_block::run`
without an explicit executor object. In this case the task will be run using the
executor associated with the execution policy that was used to call
:cpp:func:`hpx::parallel::v2::define_task_block`.

