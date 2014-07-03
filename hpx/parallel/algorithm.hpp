//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file algorithm.hpp

#if !defined(HPX_PARALLEL_ALGORITHM_MAY_28_2014_0522PM)
#define HPX_PARALLEL_ALGORITHM_MAY_28_2014_0522PM

#include <hpx/hpx_fwd.hpp>

#include <hpx/parallel/exception_list.hpp>
#include <hpx/parallel/detail/dispatch.hpp>

/// If temporary memory resources are required by any of the algorithms and
/// none are available, the algorithm throws a \a std::bad_alloc exception.
///
/// During the execution of any of the parallel algorithms, if the application
/// of a function object terminates with an uncaught exception, the behavior
/// of the program is determined by the type of execution policy used to invoke
/// the algorithm:
///
/// * If the execution policy object is of type \a vector_execution_policy,
///   \a hpx::terminate shall be called.
/// * If the execution policy object is of type \a sequential_execution_policy,
///   \a parallel_execution_policy, or task_execution_policy the execution of
///   the algorithm terminates with an \a exception_list exception. All
///   uncaught exceptions thrown during the application of user-provided
///   function objects shall be contained in the \a exception_list.
///
///   \note For example, the number of invocations of the user-provided
///         function object in for_each is unspecified. When for_each is
///         executed sequentially, only one exception will be contained in
///         the \a exception_list object.
///   \note These guarantees imply that, unless the algorithm has failed to
///         allocate memory and terminated with \a std::bad_alloc, all
///         exceptions thrown during the execution of the algorithm are
///         communicated to the caller. It is unspecified whether an algorithm
///         implementation will "forge ahead" after encountering and capturing
///         a user exception.
///   \note The algorithm may terminate with the \a std::bad_alloc exception
///         even if one or more user-provided function objects have terminated
///         with an exception. For example, this can happen when an algorithm
///         fails to allocate memory while creating or adding elements to the
///         exception_list object.
///
/// * If the execution policy object is of any other type, the behavior is
///   implementation-defined.

#include <hpx/parallel/detail/copy.hpp>
#include <hpx/parallel/detail/count.hpp>
#include <hpx/parallel/detail/fill.hpp>
#include <hpx/parallel/detail/for_each.hpp>
#include <hpx/parallel/detail/move.hpp>
#include <hpx/parallel/detail/reduce.hpp>
#include <hpx/parallel/detail/swap_ranges.hpp>
#include <hpx/parallel/detail/transform.hpp>

#undef HPX_PARALLEL_DISPATCH

#endif

