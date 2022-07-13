//  Copyright (c) 2017 Bruno Pitrus
//  Copyright (c) 2020 Hartmut Kaiser
//  Copyright (c) 2022 Dimitra Karatza
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/all_any_none.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx { namespace ranges {
    // clang-format off

    ///  Checks if unary predicate \a f returns true for no elements in the
    ///  range \a rng.
    ///
    /// \note   Complexity: At most std::distance(begin(rng), end(rng))
    ///         applications of the predicate \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a none_of requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed
    ///                     to it. The type \a Type must be such that an object
    ///                     of type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a none_of algorithm returns a \a hpx::future<bool> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a bool
    ///           otherwise.
    ///           The \a none_of algorithm returns true if the unary predicate
    ///           \a f returns true for no elements in the range, false
    ///           otherwise. It returns true if the range is empty.
    ///
    template <typename ExPolicy, typename Rng, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy, bool>::type
    none_of(ExPolicy&& policy, Rng&& rng, F&& f, Proj&& proj = Proj());

    ///  Checks if unary predicate \a f returns true for no elements in the
    ///  range [first, last).
    ///
    /// \note   Complexity: At most \a last - \a first applications of the
    ///         predicate \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam Iter        The type of the source iterators used for the
    ///                     range (deduced).
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a none_of requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the range the algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed
    ///                     to it. The type \a Type must be such that an object
    ///                     of type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a none_of algorithm returns a \a hpx::future<bool> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a bool
    ///           otherwise.
    ///           The \a none_of algorithm returns true if the unary predicate
    ///           \a f returns true for no elements in the range, false
    ///           otherwise. It returns true if the range is empty.
    ///
    template <typename ExPolicy, typename Iter, typename Sent, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy, bool>::type
    none_of(ExPolicy&& policy, Iter first, Sent last, F&& f, Proj&& proj = Proj());

    ///  Checks if unary predicate \a f returns true for no elements in the
    ///  range \a rng.
    ///
    /// \note   Complexity: At most std::distance(begin(rng), end(rng))
    ///         applications of the predicate \a f
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a none_of requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed
    ///                     to it. The type \a Type must be such that an object
    ///                     of type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// \returns  The \a none_of algorithm returns true if the unary predicate
    ///           \a f returns true for no elements in the range, false
    ///           otherwise. It returns true if the range is empty.
    ///
    template <typename Rng, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    bool none_of(Rng&& rng, F&& f, Proj&& proj = Proj());

    ///  Checks if unary predicate \a f returns true for no elements in the
    ///  range [first, last).
    ///
    /// \note   Complexity: At most \a last - \a first applications of the
    ///         predicate \a f
    ///
    /// \tparam Iter        The type of the source iterators used for the
    ///                     range (deduced).
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a none_of requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the range the algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed
    ///                     to it. The type \a Type must be such that an object
    ///                     of type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// \returns  The \a none_of algorithm returns true if the unary predicate
    ///           \a f returns true for no elements in the range, false
    ///           otherwise. It returns true if the range is empty.
    ///
    template <typename Iter, typename Sent, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    bool none_of(Iter first, Sent last, F&& f, Proj&& proj = Proj());
    
    /// Checks if unary predicate \a f returns true for at least one element
    /// in the range \a rng.
    ///
    /// \note   Complexity: At most std::distance(begin(rng), end(rng))
    ///         applications of the predicate \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a none_of requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed
    ///                     to it. The type \a Type must be such that an object
    ///                     of type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a any_of algorithm returns a \a hpx::future<bool> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a bool otherwise.
    ///           The \a any_of algorithm returns true if the unary predicate
    ///           \a f returns true for at least one element in the range,
    ///           false otherwise. It returns false if the range is empty.
    ///
    template <typename ExPolicy, typename Rng, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy, bool>::type
    any_of(ExPolicy&& policy, Rng&& rng, F&& f, Proj&& proj = Proj());

    /// Checks if unary predicate \a f returns true for at least one element
    /// in the range \a rng.
    ///
    /// \note   Complexity: At most std::distance(begin(rng), end(rng))
    ///         applications of the predicate \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam Iter        The type of the source iterators used for the
    ///                     range (deduced).
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a none_of requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the range the algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed
    ///                     to it. The type \a Type must be such that an object
    ///                     of type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a any_of algorithm returns a \a hpx::future<bool> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a bool otherwise.
    ///           The \a any_of algorithm returns true if the unary predicate
    ///           \a f returns true for at least one element in the range,
    ///           false otherwise. It returns false if the range is empty.
    ///
    template <typename ExPolicy, typename Iter, typename Sent, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy, bool>::type
    any_of(ExPolicy&& policy, Iter first, Sent last, F&& f, Proj&& proj = Proj());
    
    /// Checks if unary predicate \a f returns true for at least one element
    /// in the range \a rng.
    ///
    /// \note   Complexity: At most std::distance(begin(rng), end(rng))
    ///         applications of the predicate \a f
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a none_of requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed
    ///                     to it. The type \a Type must be such that an object
    ///                     of type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// \returns  The \a any_of algorithm returns true if the unary predicate
    ///           \a f returns true for at least one element in the range,
    ///           false otherwise. It returns false if the range is empty.
    ///
    template <typename Rng, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    bool any_of(Rng&& rng, F&& f, Proj&& proj = Proj());
    
    /// Checks if unary predicate \a f returns true for at least one element
    /// in the range \a rng.
    ///
    /// \note   Complexity: At most std::distance(begin(rng), end(rng))
    ///         applications of the predicate \a f
    ///
    /// \tparam Iter        The type of the source iterators used for the
    ///                     range (deduced).
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a none_of requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the range the algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed
    ///                     to it. The type \a Type must be such that an object
    ///                     of type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// \returns  The \a any_of algorithm returns true if the unary predicate
    ///           \a f returns true for at least one element in the range,
    ///           false otherwise. It returns false if the range is empty.
    ///
    template <typename Iter, typename Sent, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    bool any_of(Iter first, Sent last, F&& f, Proj&& proj = Proj());
    
    /// Checks if unary predicate \a f returns true for all elements in the
    /// range \a rng.
    ///
    /// \note   Complexity: At most std::distance(begin(rng), end(rng))
    ///         applications of the predicate \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a none_of requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed
    ///                     to it. The type \a Type must be such that an object
    ///                     of type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a all_of algorithm returns a \a hpx::future<bool> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a bool otherwise.
    ///           The \a all_of algorithm returns true if the unary predicate
    ///           \a f returns true for all elements in the range, false
    ///           otherwise. It returns true if the range is empty.
    ///
    template <typename ExPolicy, typename Rng, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy, bool>::type
    all_of(ExPolicy&& policy, Rng&& rng, F&& f, Proj&& proj = Proj());

    /// Checks if unary predicate \a f returns true for all elements in the
    /// range \a rng.
    ///
    /// \note   Complexity: At most std::distance(begin(rng), end(rng))
    ///         applications of the predicate \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam Iter        The type of the source iterators used for the
    ///                     range (deduced).
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a none_of requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the range the algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed
    ///                     to it. The type \a Type must be such that an object
    ///                     of type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a all_of algorithm returns a \a hpx::future<bool> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a bool otherwise.
    ///           The \a all_of algorithm returns true if the unary predicate
    ///           \a f returns true for all elements in the range, false
    ///           otherwise. It returns true if the range is empty.
    ///
    template <typename ExPolicy, typename Iter, typename Sent, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy, bool>::type
    all_of(ExPolicy&& policy, Iter first, Sent last, F&& f, Proj&& proj = Proj());

    /// Checks if unary predicate \a f returns true for all elements in the
    /// range \a rng.
    ///
    /// \note   Complexity: At most std::distance(begin(rng), end(rng))
    ///         applications of the predicate \a f
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a none_of requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed
    ///                     to it. The type \a Type must be such that an object
    ///                     of type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// \returns  The \a all_of algorithm returns true if the unary predicate
    ///           \a f returns true for all elements in the range, false
    ///           otherwise. It returns true if the range is empty.
    ///
    template <typename Rng, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    bool all_of(Rng&& rng, F&& f, Proj&& proj = Proj());

    /// Checks if unary predicate \a f returns true for all elements in the
    /// range \a rng.
    ///
    /// \note   Complexity: At most std::distance(begin(rng), end(rng))
    ///         applications of the predicate \a f
    ///
    /// \tparam Iter        The type of the source iterators used for the
    ///                     range (deduced).
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a none_of requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the range the algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed
    ///                     to it. The type \a Type must be such that an object
    ///                     of type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// \returns  The \a all_of algorithm returns true if the unary predicate
    ///           \a f returns true for all elements in the range, false
    ///           otherwise. It returns true if the range is empty.
    ///
    template <typename Iter, typename Sent, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    bool all_of(Iter first, Sent last, F&& f, Proj&& proj = Proj());

    // clang-format on
}}    // namespace hpx::ranges

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>

#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/all_any_none.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace ranges {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::none_of
    inline constexpr struct none_of_t final
      : hpx::detail::tag_parallel_algorithm<none_of_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Rng, typename F,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy, F,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_invoke(none_of_t, ExPolicy&& policy, Rng&& rng, F&& f,
            Proj&& proj = Proj())
        {
            using iterator_type =
                typename hpx::traits::range_iterator<Rng>::type;

            static_assert(
                hpx::traits::is_forward_iterator<iterator_type>::value,
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::none_of().call(
                HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
                hpx::util::end(rng), HPX_FORWARD(F, f),
                HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent, typename F,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<Iter>::value &&
                hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                hpx::parallel::traits::is_projected<Proj, Iter>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy, F,
                    hpx::parallel::traits::projected<Proj, Iter>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_invoke(none_of_t, ExPolicy&& policy, Iter first, Sent last,
            F&& f, Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_forward_iterator<Iter>::value,
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::none_of().call(
                HPX_FORWARD(ExPolicy, policy), first, last, HPX_FORWARD(F, f),
                HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename Rng, typename F,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, F,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend bool tag_fallback_invoke(
            none_of_t, Rng&& rng, F&& f, Proj&& proj = Proj())
        {
            using iterator_type =
                typename hpx::traits::range_iterator<Rng>::type;

            static_assert(hpx::traits::is_input_iterator<iterator_type>::value,
                "Required at least input iterator.");

            return hpx::parallel::v1::detail::none_of().call(
                hpx::execution::seq, hpx::util::begin(rng), hpx::util::end(rng),
                HPX_FORWARD(F, f), HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename Iter, typename Sent, typename F,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<Iter>::value &&
                hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                hpx::parallel::traits::is_projected<Proj, Iter>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, F,
                    hpx::parallel::traits::projected<Proj, Iter>
                >::value
            )>
        // clang-format on
        friend bool tag_fallback_invoke(
            none_of_t, Iter first, Sent last, F&& f, Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_input_iterator<Iter>::value,
                "Required at least input iterator.");

            return hpx::parallel::v1::detail::none_of().call(
                hpx::execution::seq, first, last, HPX_FORWARD(F, f),
                HPX_FORWARD(Proj, proj));
        }
    } none_of{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::any_of
    inline constexpr struct any_of_t final
      : hpx::detail::tag_parallel_algorithm<any_of_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Rng, typename F,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy, F,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_invoke(
            any_of_t, ExPolicy&& policy, Rng&& rng, F&& f, Proj&& proj = Proj())
        {
            using iterator_type =
                typename hpx::traits::range_iterator<Rng>::type;

            static_assert(
                hpx::traits::is_forward_iterator<iterator_type>::value,
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::any_of().call(
                HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
                hpx::util::end(rng), HPX_FORWARD(F, f),
                HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent, typename F,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<Iter>::value &&
                hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                hpx::parallel::traits::is_projected<Proj, Iter>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy, F,
                    hpx::parallel::traits::projected<Proj, Iter>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_invoke(any_of_t, ExPolicy&& policy, Iter first, Sent last,
            F&& f, Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_forward_iterator<Iter>::value,
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::any_of().call(
                HPX_FORWARD(ExPolicy, policy), first, last, HPX_FORWARD(F, f),
                HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename Rng, typename F,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value&&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value&&
                hpx::parallel::traits::is_indirect_callable<
                hpx::execution::sequenced_policy, F,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend bool tag_fallback_invoke(
            any_of_t, Rng&& rng, F&& f, Proj&& proj = Proj())
        {
            using iterator_type =
                typename hpx::traits::range_iterator<Rng>::type;

            static_assert(hpx::traits::is_input_iterator<iterator_type>::value,
                "Required at least input iterator.");

            return hpx::parallel::v1::detail::any_of().call(hpx::execution::seq,
                hpx::util::begin(rng), hpx::util::end(rng), HPX_FORWARD(F, f),
                HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename Iter, typename Sent, typename F,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<Iter>::value &&
                hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                hpx::parallel::traits::is_projected<Proj, Iter>::value &&
                hpx::parallel::traits::is_indirect_callable<
                hpx::execution::sequenced_policy, F,
                    hpx::parallel::traits::projected<Proj, Iter>
                >::value
            )>
        // clang-format on
        friend bool tag_fallback_invoke(
            any_of_t, Iter first, Sent last, F&& f, Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_input_iterator<Iter>::value,
                "Required at least input iterator.");

            return hpx::parallel::v1::detail::any_of().call(hpx::execution::seq,
                first, last, HPX_FORWARD(F, f), HPX_FORWARD(Proj, proj));
        }
    } any_of{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::all_of
    inline constexpr struct all_of_t final
      : hpx::detail::tag_parallel_algorithm<all_of_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Rng, typename F,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy, F,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_invoke(
            all_of_t, ExPolicy&& policy, Rng&& rng, F&& f, Proj&& proj = Proj())
        {
            using iterator_type =
                typename hpx::traits::range_iterator<Rng>::type;

            static_assert(
                hpx::traits::is_forward_iterator<iterator_type>::value,
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::all_of().call(
                HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
                hpx::util::end(rng), HPX_FORWARD(F, f),
                HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent, typename F,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<Iter>::value &&
                hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                hpx::parallel::traits::is_projected<Proj, Iter>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy, F,
                    hpx::parallel::traits::projected<Proj, Iter>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_invoke(all_of_t, ExPolicy&& policy, Iter first, Sent last,
            F&& f, Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_forward_iterator<Iter>::value,
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::all_of().call(
                HPX_FORWARD(ExPolicy, policy), first, last, HPX_FORWARD(F, f),
                HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename Rng, typename F,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
                hpx::parallel::traits::is_indirect_callable<
                hpx::execution::sequenced_policy, F,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend bool tag_fallback_invoke(
            all_of_t, Rng&& rng, F&& f, Proj&& proj = Proj())
        {
            using iterator_type =
                typename hpx::traits::range_iterator<Rng>::type;

            static_assert(hpx::traits::is_input_iterator<iterator_type>::value,
                "Required at least input iterator.");

            return hpx::parallel::v1::detail::all_of().call(hpx::execution::seq,
                hpx::util::begin(rng), hpx::util::end(rng), HPX_FORWARD(F, f),
                HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename Iter, typename Sent, typename F,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<Iter>::value &&
                hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                hpx::parallel::traits::is_projected<Proj, Iter>::value &&
                hpx::parallel::traits::is_indirect_callable<
                hpx::execution::sequenced_policy, F,
                    hpx::parallel::traits::projected<Proj, Iter>
                >::value
            )>
        // clang-format on
        friend bool tag_fallback_invoke(
            all_of_t, Iter first, Sent last, F&& f, Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_input_iterator<Iter>::value,
                "Required at least input iterator.");

            return hpx::parallel::v1::detail::all_of().call(hpx::execution::seq,
                first, last, HPX_FORWARD(F, f), HPX_FORWARD(Proj, proj));
        }
    } all_of{};

}}    // namespace hpx::ranges

#endif    // DOXYGEN
