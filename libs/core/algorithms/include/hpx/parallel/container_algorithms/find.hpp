//  Copyright (c) 2018 Bruno Pitrus
//  Copyright (c) 2020-2023 Hartmut Kaiser
//  Copyright (c) 2022 Dimitra Karatza
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/find.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx { namespace ranges {
    // clang-format off

    /// Returns the first element in the range [first, last) that is equal
    /// to value
    ///
    /// \note   Complexity: At most last - first
    ///         applications of the operator==().
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Iter        The type of the begin source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the end source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter.
    /// \tparam T           The type of the value to find (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param val          the value to compare the elements to
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparison operations in the parallel \a find algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a find algorithm returns the first element in the range
    ///           [first,last) that is equal to \a val.
    ///           If no such element in the range of [first,last) is equal to
    ///           \a val, then the algorithm returns \a last.
    ///
    template <typename ExPolicy, typename Iter, typename Sent,
        typename Proj = hpx::identity,
        typename T = typename hpx::parallel::traits::projected<Iter,
            Proj>::value_type>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, Iter>
    find(ExPolicy&& policy, Iter first, Sent last, T const& val,
        Proj&& proj = Proj());

    /// Returns the first element in the range [first, last) that is equal
    /// to value
    ///
    /// \note   Complexity: At most last - first
    ///         applications of the operator==().
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam T           The type of the value to find (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param val          the value to compare the elements to
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparison operations in the parallel \a find algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a find algorithm returns the first element in the range
    ///           [first,last) that is equal to \a val.
    ///           If no such element in the range of [first,last) is equal to
    ///           \a val, then the algorithm returns \a last.
    ///
    template <typename ExPolicy, typename Rng,
        typename Proj = hpx::identity,
        typename T = typename hpx::parallel::traits::projected<
            hpx::traits::range_iterator_t<Rng>, Proj>::value_type>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        hpx::traits::range_iterator_t<Rng>>
    find(ExPolicy&& policy, Rng&& rng, T const& val, Proj&& proj = Proj());

    /// Returns the first element in the range [first, last) that is equal
    /// to value
    ///
    /// \note   Complexity: At most last - first
    ///         applications of the operator==().
    ///
    /// \tparam Iter        The type of the begin source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the end source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter.
    /// \tparam T           The type of the value to find (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param val          the value to compare the elements to
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// \returns  The \a find algorithm returns the first element in the range
    ///           [first,last) that is equal to \a val.
    ///           If no such element in the range of [first,last) is equal to
    ///           \a val, then the algorithm returns \a last.
    ///
    template <typename Iter, typename Sent,
        typename Proj = hpx::identity,
        typename T = typename hpx::parallel::traits::projected<Iter,
            Proj>::value_type>
    Iter find(Iter first, Sent last, T const& val, Proj&& proj = Proj());

    /// Returns the first element in the range [first, last) that is equal
    /// to value
    ///
    /// \note   Complexity: At most last - first
    ///         applications of the operator==().
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam T           The type of the value to find (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param val          the value to compare the elements to
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparison operations in the parallel \a find algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a find algorithm returns the first element in the range
    ///           [first,last) that is equal to \a val.
    ///           If no such element in the range of [first,last) is equal to
    ///           \a val, then the algorithm returns \a last.
    ///
    template <typename Rng,
        typename Proj = hpx::identity,
        typename T = typename hpx::parallel::traits::projected<
            hpx::traits::range_iterator_t<Rng>, Proj>::value_type>
    hpx::traits::range_iterator_t<Rng>
    find(Rng&& rng, T const& val, Proj&& proj = Proj());

    /// Returns the first element in the range [first, last) for which
    /// predicate \a pred returns true
    ///
    /// \note   Complexity: At most last - first
    ///         applications of the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Iter        The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Sent        The type of the end source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param pred         The unary predicate which returns true for the
    ///                     required element. The signature of the predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such
    ///                     that objects of type \a FwdIter can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparison operations in the parallel \a find_if algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find_if algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find_if algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a find_if algorithm returns the first element in the range
    ///           [first,last) that satisfies the predicate \a f.
    ///           If no such element exists that satisfies the predicate f, the
    ///           algorithm returns \a last.
    ///
    template <typename ExPolicy, typename Iter, typename Sent, typename Pred,
        typename Proj = hpx::identity>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, Iter>
    find_if(ExPolicy&& policy, Iter first, Sent last,
        Pred&& pred, Proj&& proj = Proj());

    /// Returns the first element in the range \a rng for which
    /// predicate \a pred returns true
    ///
    /// \note   Complexity: At most last - first
    ///         applications of the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param pred         The unary predicate which returns true for the
    ///                     required element. The signature of the predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such
    ///                     that objects of type \a FwdIter can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparison operations in the parallel \a find_if algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find_if algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find_if algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a find_if algorithm returns the first element in the range
    ///           [first,last) that satisfies the predicate \a f.
    ///           If no such element exists that satisfies the predicate f, the
    ///           algorithm returns \a last.
    ///
    template <typename ExPolicy, typename Rng, typename Pred,
        typename Proj = hpx::identity>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        hpx::traits::range_iterator_t<Rng>>
    find_if(ExPolicy&& policy, Rng&& rng, Pred&& pred, Proj&& proj = Proj());

    /// Returns the first element in the range [first, last) for which
    /// predicate \a pred returns true
    ///
    /// \note   Complexity: At most last - first
    ///         applications of the predicate.
    ///
    /// \tparam Iter        The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Sent        The type of the end source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param pred         The unary predicate which returns true for the
    ///                     required element. The signature of the predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such
    ///                     that objects of type \a FwdIter can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// \returns  The \a find_if algorithm returns the first element in the range
    ///           [first,last) that satisfies the predicate \a f.
    ///           If no such element exists that satisfies the predicate f, the
    ///           algorithm returns \a last.
    ///
    template <typename Iter, typename Sent, typename Pred,
        typename Proj = hpx::identity>
    Iter find_if(Iter first, Sent last, Pred&& pred, Proj&& proj = Proj());

    /// Returns the first element in the range \a rng for which
    /// predicate \a pred returns true
    ///
    /// \note   Complexity: At most last - first
    ///         applications of the predicate.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param pred         The unary predicate which returns true for the
    ///                     required element. The signature of the predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such
    ///                     that objects of type \a FwdIter can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// \returns  The \a find_if algorithm returns the first element in the range
    ///           [first,last) that satisfies the predicate \a f.
    ///           If no such element exists that satisfies the predicate f, the
    ///           algorithm returns \a last.
    ///
    template <typename Rng, typename Pred,
        typename Proj = hpx::identity>
    hpx::traits::range_iterator_t<Rng>
    find_if(Rng&& rng, Pred&& pred, Proj&& proj = Proj());

    /// Returns the first element in the range [first, last) for which
    /// predicate \a f returns false
    ///
    /// \note   Complexity: At most last - first
    ///         applications of the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Iter        The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Sent        The type of the end source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param pred         The unary predicate which returns false for the
    ///                     required element. The signature of the predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such
    ///                     that objects of type \a FwdIter can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparison operations in the parallel \a find_if_not algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find_if_not algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find_if_not algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a find_if_not algorithm returns the first element in the range
    ///           [first, last) that does \b not satisfy the predicate \a f.
    ///           If no such element exists that does not satisfy the predicate f, the
    ///           algorithm returns \a last.
    ///
    template <typename ExPolicy, typename Iter, typename Sent, typename Pred,
        typename Proj = hpx::identity>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        Iter>::type
    find_if_not(ExPolicy&& policy, Iter first, Sent last, Pred&& pred,
        Proj&& proj = Proj());

    /// Returns the first element in the range \a rng for which
    /// predicate \a f returns false
    ///
    /// \note   Complexity: At most last - first
    ///         applications of the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param pred         The unary predicate which returns false for the
    ///                     required element. The signature of the predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such
    ///                     that objects of type \a FwdIter can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparison operations in the parallel \a find_if_not algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find_if_not algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find_if_not algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a find_if_not algorithm returns the first element in the range
    ///           [first, last) that does \b not satisfy the predicate \a f.
    ///           If no such element exists that does not satisfy the predicate f, the
    ///           algorithm returns \a last.
    ///
    template <typename ExPolicy, typename Rng, typename Pred,
        typename Proj = hpx::identity>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        hpx::traits::range_iterator_t<Rng>>
    find_if_not(ExPolicy&& policy, Rng&& rng, Pred&& pred, Proj&& proj = Proj());

    /// Returns the first element in the range [first, last) for which
    /// predicate \a f returns false
    ///
    /// \note   Complexity: At most last - first
    ///         applications of the predicate.
    ///
    /// \tparam Iter        The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Sent        The type of the end source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param pred         The unary predicate which returns false for the
    ///                     required element. The signature of the predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such
    ///                     that objects of type \a FwdIter can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// \returns  The \a find_if_not algorithm returns the first element in the range
    ///           [first, last) that does \b not satisfy the predicate \a f.
    ///           If no such element exists that does not satisfy the predicate f, the
    ///           algorithm returns \a last.
    ///
    template <typename Iter, typename Sent, typename Pred,
        typename Proj = hpx::identity>
    Iter find_if_not(Iter first, Sent last, Pred&& pred, Proj&& proj = Proj());

    /// Returns the first element in the range \a rng for which
    /// predicate \a f returns false
    ///
    /// \note   Complexity: At most last - first
    ///         applications of the predicate.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param pred         The unary predicate which returns false for the
    ///                     required element. The signature of the predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such
    ///                     that objects of type \a FwdIter can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// \returns  The \a find_if_not algorithm returns the first element in the range
    ///           [first, last) that does \b not satisfy the predicate \a f.
    ///           If no such element exists that does not satisfy the predicate f, the
    ///           algorithm returns \a last.
    ///
    template <typename Rng, typename Pred,
        typename Proj = hpx::identity>
    hpx::traits::range_iterator_t<Rng>
    find_if_not(Rng&& rng, Pred&& pred, Proj&& proj = Proj());

    /// Returns the last subsequence of elements \a rng2 found in the range
    /// \a rng using the given predicate \a f to compare elements.
    ///
    /// \note   Complexity: at most S*(N-S+1) comparisons where
    ///         \a S = distance(begin(rng2), end(rng2)) and
    ///         \a N = distance(begin(rng), end(rng)).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng1        The type of the first source range (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam Rng2        The type of the second source range (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a replace requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj1       The type of an optional projection function applied
    ///                     to the first sequence. This
    ///                     defaults to \a hpx::identity
    /// \tparam Proj2       The type of an optional projection function applied
    ///                     to the second sequence. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng1         Refers to the first sequence of elements
    ///                     the algorithm will be applied to.
    /// \param rng2         Refers to the second sequence of elements
    ///                     the algorithm will be applied to.
    /// \param op           The binary predicate which returns \a true
    ///                     if the elements should be treated as equal. The signature
    ///                     should be equivalent to the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such
    ///                     that objects of types \a iterator_t<Rng> and \a iterator_t<Rng2>
    ///                     can be dereferenced and then implicitly converted
    ///                     to \a Type1 and \a Type2 respectively.
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the first
    ///                     range of type dereferenced \a iterator_t<Rng1>
    ///                     as a projection operation before the function \a op
    ///                     is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the second
    ///                     range of type dereferenced \a iterator_t<Rng2>
    ///                     as a projection operation before the function \a op
    ///                     is invoked.
    ///
    /// The comparison operations in the parallel \a find_end algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find_end algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find_end algorithm returns a \a hpx::future<iterator_t<Rng> >
    ///           if the execution policy is of type \a sequenced_task_policy
    ///           or \a parallel_task_policy and returns \a iterator_t<Rng> otherwise.
    ///           The \a find_end algorithm returns an iterator to the beginning of
    ///           the last subsequence \a rng2 in range \a rng.
    ///           If the length of the subsequence \a rng2 is greater
    ///           than the length of the range \a rng, \a end(rng) is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, \a end(rng) is also returned.
    ///
    /// This overload of \a find_end is available if the user decides to provide the
    /// algorithm their own predicate \a op.
    ///
    template <typename ExPolicy, typename Rng1, typename Rng2,
        typename Pred = equal_to,
        typename Proj1 = hpx::identity,
        typename Proj2 = hpx::identity>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        hpx::traits::range_iterator_t<Rng1>>
    find_end(ExPolicy&& policy, Rng1&& rng1, Rng2&& rng2, Pred&& op = Pred(),
        Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2());

    /// Returns the last subsequence of elements \a [first2, last2) found in
    /// the range \a [first1, last1) using the given predicate \a f to
    /// compare elements.
    ///
    /// \note   Complexity: at most S*(N-S+1) comparisons where
    ///         \a S = distance(first2, last2) and
    ///         \a N = distance(first1, last1).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Iter1       The type of the begin source iterators for the first
    ///                     sequence used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent1       The type of the end source iterators for the first
    ///                     sequence used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter1.
    /// \tparam Iter2       The type of the begin source iterators for the second
    ///                     sequence used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent2       The type of the end source iterators for the second
    ///                     sequence used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter2.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a replace requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj1       The type of an optional projection function applied
    ///                     to the first sequence. This
    ///                     defaults to \a hpx::identity
    /// \tparam Proj2       The type of an optional projection function applied
    ///                     to the second sequence. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the first sequence of
    ///                     elements the algorithm will be applied to.
    /// \param last1        Refers to the end of the first sequence of elements
    ///                     the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the second sequence of
    ///                     elements the algorithm will be applied to.
    /// \param last2        Refers to the end of the second sequence of elements
    ///                     the algorithm will be applied to.
    /// \param op           The binary predicate which returns \a true
    ///                     if the elements should be treated as equal. The signature
    ///                     should be equivalent to the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such
    ///                     that objects of types \a iterator_t<Rng> and \a iterator_t<Rng2>
    ///                     can be dereferenced and then implicitly converted
    ///                     to \a Type1 and \a Type2 respectively.
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the first
    ///                     range of type dereferenced \a iterator_t<Rng1>
    ///                     as a projection operation before the function \a op
    ///                     is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the second
    ///                     range of type dereferenced \a iterator_t<Rng2>
    ///                     as a projection operation before the function \a op
    ///                     is invoked.
    ///
    /// The comparison operations in the parallel \a find_end algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find_end algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find_end algorithm returns a \a hpx::future<iterator_t<Rng> >
    ///           if the execution policy is of type \a sequenced_task_policy
    ///           or \a parallel_task_policy and returns \a iterator_t<Rng> otherwise.
    ///           The \a find_end algorithm returns an iterator to the beginning of
    ///           the last subsequence \a rng2 in range \a rng.
    ///           If the length of the subsequence \a rng2 is greater
    ///           than the length of the range \a rng, \a end(rng) is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, \a end(rng) is also returned.
    ///
    /// This overload of \a find_end is available if the user decides to provide the
    /// algorithm their own predicate \a op.
    ///
    template <typename ExPolicy, typename Iter1, typename Sent1,
        typename Iter2, typename Sent2, typename Pred = equal_to,
        typename Proj1 = hpx::identity,
        typename Proj2 = hpx::identity>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, Iter1>
    find_end(ExPolicy&& policy, Iter1 first1, Sent1 last1, Iter2 first2,
        Sent2 last2, Pred&& op = Pred(), Proj1&& proj1 = Proj1(),
        Proj2&& proj2 = Proj2());

    /// Returns the last subsequence of elements \a rng2 found in the range
    /// \a rng using the given predicate \a f to compare elements.
    ///
    /// \note   Complexity: at most S*(N-S+1) comparisons where
    ///         \a S = distance(begin(rng2), end(rng2)) and
    ///         \a N = distance(begin(rng), end(rng)).
    ///
    /// \tparam Rng1        The type of the first source range (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam Rng2        The type of the second source range (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a replace requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj1       The type of an optional projection function applied
    ///                     to the first sequence. This
    ///                     defaults to \a hpx::identity
    /// \tparam Proj2       The type of an optional projection function applied
    ///                     to the second sequence. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param rng1         Refers to the first sequence of elements
    ///                     the algorithm will be applied to.
    /// \param rng2         Refers to the second sequence of elements
    ///                     the algorithm will be applied to.
    /// \param op           The binary predicate which returns \a true
    ///                     if the elements should be treated as equal. The signature
    ///                     should be equivalent to the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such
    ///                     that objects of types \a iterator_t<Rng> and \a iterator_t<Rng2>
    ///                     can be dereferenced and then implicitly converted
    ///                     to \a Type1 and \a Type2 respectively.
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the first
    ///                     range of type dereferenced \a iterator_t<Rng1>
    ///                     as a projection operation before the function \a op
    ///                     is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the second
    ///                     range of type dereferenced \a iterator_t<Rng2>
    ///                     as a projection operation before the function \a op
    ///                     is invoked.
    ///
    /// \returns  The \a find_end algorithm returns an iterator to the beginning of
    ///           the last subsequence \a rng2 in range \a rng.
    ///           If the length of the subsequence \a rng2 is greater
    ///           than the length of the range \a rng, \a end(rng) is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, \a end(rng) is also returned.
    ///
    /// This overload of \a find_end is available if the user decides to provide the
    /// algorithm their own predicate \a op.
    ///
    template <typename Rng1, typename Rng2, typename Pred = equal_to,
        typename Proj1 = hpx::identity,
        typename Proj2 = hpx::identity>
    hpx::traits::range_iterator_t<Rng1>
    find_end(Rng1&& rng1, Rng2&& rng2, Pred&& op = Pred(),
        Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2());

    /// Returns the last subsequence of elements \a [first2, last2) found in
    /// the range \a [first1, last1) using the given predicate \a f to
    /// compare elements.
    ///
    /// \note   Complexity: at most S*(N-S+1) comparisons where
    ///         \a S = distance(first2, last2) and
    ///         \a N = distance(first1, last1).
    ///
    /// \tparam Iter1       The type of the begin source iterators for the first
    ///                     sequence used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent1       The type of the end source iterators for the first
    ///                     sequence used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter1.
    /// \tparam Iter2       The type of the begin source iterators for the second
    ///                     sequence used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent2       The type of the end source iterators for the second
    ///                     sequence used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter2.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a replace requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj1       The type of an optional projection function applied
    ///                     to the first sequence. This
    ///                     defaults to \a hpx::identity
    /// \tparam Proj2       The type of an optional projection function applied
    ///                     to the second sequence. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param first1       Refers to the beginning of the first sequence of
    ///                     elements the algorithm will be applied to.
    /// \param last1        Refers to the end of the first sequence of elements
    ///                     the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the second sequence of
    ///                     elements the algorithm will be applied to.
    /// \param last2        Refers to the end of the second sequence of elements
    ///                     the algorithm will be applied to.
    /// \param op           The binary predicate which returns \a true
    ///                     if the elements should be treated as equal. The signature
    ///                     should be equivalent to the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such
    ///                     that objects of types \a iterator_t<Rng> and \a iterator_t<Rng2>
    ///                     can be dereferenced and then implicitly converted
    ///                     to \a Type1 and \a Type2 respectively.
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the first
    ///                     range of type dereferenced \a iterator_t<Rng1>
    ///                     as a projection operation before the function \a op
    ///                     is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the second
    ///                     range of type dereferenced \a iterator_t<Rng2>
    ///                     as a projection operation before the function \a op
    ///                     is invoked.
    ///
    /// \returns  The \a find_end algorithm returns an iterator to the beginning of
    ///           the last subsequence \a rng2 in range \a rng.
    ///           If the length of the subsequence \a rng2 is greater
    ///           than the length of the range \a rng, \a end(rng) is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, \a end(rng) is also returned.
    ///
    /// This overload of \a find_end is available if the user decides to provide the
    /// algorithm their own predicate \a op.
    ///
    template <typename Iter1, typename Sent1,
        typename Iter2, typename Sent2, typename Pred = equal_to,
        typename Proj1 = hpx::identity,
        typename Proj2 = hpx::identity>
    Iter1 find_end(Iter1 first1, Sent1 last1, Iter2 first2,
        Sent2 last2, Pred&& op = Pred(), Proj1&& proj1 = Proj1(),
        Proj2&& proj2 = Proj2());

    /// Searches the range \a rng1 for any elements in the range \a rng2.
    /// Uses binary predicate \a p to compare elements
    ///
    /// \note   Complexity: at most (S*N) comparisons where
    ///         \a S = distance(begin(rng2), end(rng2)) and
    ///         \a N = distance(begin(rng1), end(rng1)).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng1        The type of the first source range (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam Rng2        The type of the second source range (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a replace requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj1       The type of an optional projection function. This
    ///                     defaults to \a hpx::identity and is
    ///                     applied to the elements in \a rng1.
    /// \tparam Proj2       The type of an optional projection function. This
    ///                     defaults to \a hpx::identity and is
    ///                     applied to the elements in \a rng2.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng1         Refers to the first sequence of elements
    ///                     the algorithm will be applied to.
    /// \param rng2         Refers to the second sequence of elements
    ///                     the algorithm will be applied to.
    /// \param op           The binary predicate which returns \a true
    ///                     if the elements should be treated as equal. The signature
    ///                     should be equivalent to the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such
    ///                     that objects of types \a iterator_t<Rng1>
    ///                     and \a iterator_t<Rng2>
    ///                     can be dereferenced and then implicitly converted
    ///                     to \a Type1 and \a Type2 respectively.
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of type
    ///                     dereferenced \a iterator_t<Rng1> before the function
    ///                     \a op is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of type
    ///                     dereferenced \a iterator_t<Rng2> before the function
    ///                     \a op is invoked.
    ///
    /// The comparison operations in the parallel \a find_first_of algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find_first_of algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find_end algorithm returns a \a hpx::future<iterator_t<Rng1> >
    ///           if the execution policy is of type \a sequenced_task_policy
    ///           or \a parallel_task_policy and returns \a iterator_t<Rng1> otherwise.
    ///           The \a find_first_of algorithm returns an iterator to the first element
    ///           in the range \a rng1 that is equal to an element from the range
    ///           \a rng2.
    ///           If the length of the subsequence \a rng2 is
    ///           greater than the length of the range \a rng1,
    ///           \a end(rng1) is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, \a end(rng1) is also returned.
    ///
    /// This overload of \a find_first_of is available if the user decides to provide the
    /// algorithm their own predicate \a op.
    ///
    template <typename ExPolicy, typename Rng1, typename Rng2,
        typename Pred = equal_to,
        typename Proj1 = hpx::identity,
        typename Proj2 = hpx::identity>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        hpx::traits::range_iterator_t<Rng1>>
    find_first_of(ExPolicy&& policy, Rng1&& rng1, Rng2&& rng2,
        Pred&& op = Pred(), Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2());

    /// Searches the range \a [first1, last1) for any elements in the
    /// range \a [first2, last2).
    /// Uses binary predicate \a p to compare elements
    ///
    /// \note   Complexity: at most (S*N) comparisons where
    ///         \a S = distance(first2, last2) and
    ///         \a N = distance(first1, last1).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Iter1       The type of the begin source iterators for the first
    ///                     sequence used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent1       The type of the end source iterators for the first
    ///                     sequence used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter1.
    /// \tparam Iter2       The type of the begin source iterators for the second
    ///                     sequence used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent2       The type of the end source iterators for the second
    ///                     sequence used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter2.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a replace requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj1       The type of an optional projection function. This
    ///                     defaults to \a hpx::identity and is
    ///                     applied to the elements in \a rng1.
    /// \tparam Proj2       The type of an optional projection function. This
    ///                     defaults to \a hpx::identity and is
    ///                     applied to the elements in \a rng2.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the first sequence of
    ///                     elements the algorithm will be applied to.
    /// \param last1        Refers to the end of the first sequence of elements
    ///                     the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the second sequence of
    ///                     elements the algorithm will be applied to.
    /// \param last2        Refers to the end of the second sequence of elements
    ///                     the algorithm will be applied to.
    /// \param op           The binary predicate which returns \a true
    ///                     if the elements should be treated as equal. The signature
    ///                     should be equivalent to the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such
    ///                     that objects of types \a iterator_t<Rng1>
    ///                     and \a iterator_t<Rng2>
    ///                     can be dereferenced and then implicitly converted
    ///                     to \a Type1 and \a Type2 respectively.
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of type
    ///                     dereferenced \a iterator_t<Rng1> before the function
    ///                     \a op is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of type
    ///                     dereferenced \a iterator_t<Rng2> before the function
    ///                     \a op is invoked.
    ///
    /// The comparison operations in the parallel \a find_first_of algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find_first_of algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find_end algorithm returns a \a hpx::future<iterator_t<Rng1> >
    ///           if the execution policy is of type \a sequenced_task_policy
    ///           or \a parallel_task_policy and returns \a iterator_t<Rng1> otherwise.
    ///           The \a find_first_of algorithm returns an iterator to the first element
    ///           in the range \a rng1 that is equal to an element from the range
    ///           \a rng2.
    ///           If the length of the subsequence \a rng2 is
    ///           greater than the length of the range \a rng1,
    ///           \a end(rng1) is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, \a end(rng1) is also returned.
    ///
    /// This overload of \a find_first_of is available if the user decides to provide the
    /// algorithm their own predicate \a op.
    ///
    template <typename ExPolicy, typename Iter1, typename Sent1,
        typename Iter2, typename Sent2, typename Pred = equal_to,
        typename Proj1 = hpx::identity,
        typename Proj2 = hpx::identity>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, Iter1>
    find_first_of(ExPolicy&& policy, Iter1 first1, Sent1 last1, Iter2 first2,
        Sent2 last2, Pred&& op = Pred(), Proj1&& proj1 = Proj1(),
        Proj2&& proj2 = Proj2());

    /// Searches the range \a rng1 for any elements in the range \a rng2.
    /// Uses binary predicate \a p to compare elements
    ///
    /// \note   Complexity: at most (S*N) comparisons where
    ///         \a S = distance(begin(rng2), end(rng2)) and
    ///         \a N = distance(begin(rng1), end(rng1)).
    ///
    /// \tparam Rng1        The type of the first source range (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam Rng2        The type of the second source range (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a replace requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj1       The type of an optional projection function. This
    ///                     defaults to \a hpx::identity and is
    ///                     applied to the elements in \a rng1.
    /// \tparam Proj2       The type of an optional projection function. This
    ///                     defaults to \a hpx::identity and is
    ///                     applied to the elements in \a rng2.
    ///
    /// \param rng1         Refers to the first sequence of elements
    ///                     the algorithm will be applied to.
    /// \param rng2         Refers to the second sequence of elements
    ///                     the algorithm will be applied to.
    /// \param op           The binary predicate which returns \a true
    ///                     if the elements should be treated as equal. The signature
    ///                     should be equivalent to the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such
    ///                     that objects of types \a iterator_t<Rng1>
    ///                     and \a iterator_t<Rng2>
    ///                     can be dereferenced and then implicitly converted
    ///                     to \a Type1 and \a Type2 respectively.
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of type
    ///                     dereferenced \a iterator_t<Rng1> before the function
    ///                     \a op is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of type
    ///                     dereferenced \a iterator_t<Rng2> before the function
    ///                     \a op is invoked.
    ///
    /// \returns  The \a find_first_of algorithm returns an iterator to the first element
    ///           in the range \a rng1 that is equal to an element from the range
    ///           \a rng2.
    ///           If the length of the subsequence \a rng2 is
    ///           greater than the length of the range \a rng1,
    ///           \a end(rng1) is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, \a end(rng1) is also returned.
    ///
    /// This overload of \a find_first_of is available if the user decides to provide the
    /// algorithm their own predicate \a op.
    ///
    template <typename Rng1, typename Rng2, typename Pred = equal_to,
        typename Proj1 = hpx::identity,
        typename Proj2 = hpx::identity>
    hpx::traits::range_iterator_t<Rng1>
    find_first_of(Rng1&& rng1, Rng2&& rng2,
        Pred&& op = Pred(), Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2());

    /// Searches the range \a [first1, last1) for any elements in the
    /// range \a [first2, last2).
    /// Uses binary predicate \a p to compare elements
    ///
    /// \note   Complexity: at most (S*N) comparisons where
    ///         \a S = distance(first2, last2) and
    ///         \a N = distance(first1, last1).
    ///
    /// \tparam Iter1       The type of the begin source iterators for the first
    ///                     sequence used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent1       The type of the end source iterators for the first
    ///                     sequence used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter1.
    /// \tparam Iter2       The type of the begin source iterators for the second
    ///                     sequence used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent2       The type of the end source iterators for the second
    ///                     sequence used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter2.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a replace requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj1       The type of an optional projection function. This
    ///                     defaults to \a hpx::identity and is
    ///                     applied to the elements in \a rng1.
    /// \tparam Proj2       The type of an optional projection function. This
    ///                     defaults to \a hpx::identity and is
    ///                     applied to the elements in \a rng2.
    ///
    /// \param first1       Refers to the beginning of the first sequence of
    ///                     elements the algorithm will be applied to.
    /// \param last1        Refers to the end of the first sequence of elements
    ///                     the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the second sequence of
    ///                     elements the algorithm will be applied to.
    /// \param last2        Refers to the end of the second sequence of elements
    ///                     the algorithm will be applied to.
    /// \param op           The binary predicate which returns \a true
    ///                     if the elements should be treated as equal. The signature
    ///                     should be equivalent to the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such
    ///                     that objects of types \a iterator_t<Rng1>
    ///                     and \a iterator_t<Rng2>
    ///                     can be dereferenced and then implicitly converted
    ///                     to \a Type1 and \a Type2 respectively.
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of type
    ///                     dereferenced \a iterator_t<Rng1> before the function
    ///                     \a op is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of type
    ///                     dereferenced \a iterator_t<Rng2> before the function
    ///                     \a op is invoked.
    ///
    /// \returns  The \a find_first_of algorithm returns an iterator to the first element
    ///           in the range \a rng1 that is equal to an element from the range
    ///           \a rng2.
    ///           If the length of the subsequence \a rng2 is
    ///           greater than the length of the range \a rng1,
    ///           \a end(rng1) is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, \a end(rng1) is also returned.
    ///
    /// This overload of \a find_first_of is available if the user decides to provide the
    /// algorithm their own predicate \a op.
    ///
    template <typename Iter1, typename Sent1, typename Iter2, typename Sent2,
        typename Pred = equal_to,
        typename Proj1 = hpx::identity,
        typename Proj2 = hpx::identity>
    Iter1 find_first_of(Iter1 first1, Sent1 last1, Iter2 first2,
        Sent2 last2, Pred&& op = Pred(), Proj1&& proj1 = Proj1(),
        Proj2&& proj2 = Proj2());

    // clang-format on
}}    // namespace hpx::ranges

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/parallel/algorithms/find.hpp>
#include <hpx/type_support/identity.hpp>

#include <type_traits>
#include <utility>

namespace hpx::ranges {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::find
    inline constexpr struct find_t final
      : hpx::detail::tag_parallel_algorithm<find_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent,
            typename Proj = hpx::identity,
            typename T = typename hpx::parallel::traits::projected<Iter,
                Proj>::value_type,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_sentinel_for_v<Sent, Iter> &&
                hpx::parallel::traits::is_projected_v<Proj, Iter>
            )>
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy, Iter>
        tag_fallback_invoke(find_t, ExPolicy&& policy, Iter first, Sent last,
            T const& val, Proj proj = Proj())
        {
            static_assert(hpx::traits::is_forward_iterator_v<Iter>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::find<Iter>().call(
                HPX_FORWARD(ExPolicy, policy), first, last, val,
                HPX_MOVE(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,
            typename Proj = hpx::identity,
            typename T = typename hpx::parallel::traits::projected<
                hpx::traits::range_iterator_t<Rng>, Proj>::value_type,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_range_v<Rng> &&
                hpx::parallel::traits::is_projected_range_v<Proj, Rng>
            )>
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy,
            hpx::traits::range_iterator_t<Rng>>
        tag_fallback_invoke(find_t, ExPolicy&& policy, Rng&& rng, T const& val,
            Proj proj = Proj())
        {
            using iterator_type = hpx::traits::range_iterator_t<Rng>;

            static_assert(hpx::traits::is_forward_iterator_v<iterator_type>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::find<iterator_type>().call(
                HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
                hpx::util::end(rng), val, HPX_MOVE(proj));
        }

        // clang-format off
        template <typename Iter, typename Sent,
            typename Proj = hpx::identity,
            typename T = typename hpx::parallel::traits::projected<Iter,
                Proj>::value_type,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_sentinel_for_v<Sent, Iter> &&
                hpx::parallel::traits::is_projected_v<Proj, Iter>
            )>
        // clang-format on
        friend Iter tag_fallback_invoke(
            find_t, Iter first, Sent last, T const& val, Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_input_iterator_v<Iter>,
                "Requires at least input iterator.");

            return hpx::parallel::detail::find<Iter>().call(
                hpx::execution::seq, first, last, val, HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename Rng,
            typename Proj = hpx::identity,
            typename T = typename hpx::parallel::traits::projected<
                hpx::traits::range_iterator_t<Rng>, Proj>::value_type,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range_v<Rng> &&
                hpx::parallel::traits::is_projected_range_v<Proj, Rng>
            )>
        // clang-format on
        friend hpx::traits::range_iterator_t<Rng> tag_fallback_invoke(
            find_t, Rng&& rng, T const& val, Proj&& proj = Proj())
        {
            using iterator_type = hpx::traits::range_iterator_t<Rng>;

            static_assert(hpx::traits::is_input_iterator_v<iterator_type>,
                "Requires at least input iterator.");

            return hpx::parallel::detail::find<iterator_type>().call(
                hpx::execution::seq, hpx::util::begin(rng), hpx::util::end(rng),
                val, HPX_FORWARD(Proj, proj));
        }
    } find{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::find_if
    inline constexpr struct find_if_t final
      : hpx::detail::tag_parallel_algorithm<find_if_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent, typename Pred,
            typename Proj = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_sentinel_for_v<Sent, Iter> &&
                hpx::parallel::traits::is_projected_v<Proj, Iter> &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<Iter>::value_type
                >
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            Iter>::type
        tag_fallback_invoke(find_if_t, ExPolicy&& policy, Iter first, Sent last,
            Pred&& pred, Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_forward_iterator_v<Iter>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::find_if<Iter>().call(
                HPX_FORWARD(ExPolicy, policy), first, last,
                HPX_FORWARD(Pred, pred), HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename Pred,
            typename Proj = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_range_v<Rng> &&
                hpx::parallel::traits::is_projected_range_v<Proj, Rng> &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<
                        hpx::traits::range_iterator_t<Rng>
                    >::value_type
                >
            )>
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy,
            hpx::traits::range_iterator_t<Rng>>
        tag_fallback_invoke(find_if_t, ExPolicy&& policy, Rng&& rng, Pred pred,
            Proj proj = Proj())
        {
            using iterator_type = hpx::traits::range_iterator_t<Rng>;

            static_assert(hpx::traits::is_forward_iterator_v<iterator_type>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::find_if<iterator_type>().call(
                HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
                hpx::util::end(rng), HPX_MOVE(pred), HPX_MOVE(proj));
        }

        // clang-format off
        template <typename Iter, typename Sent, typename Pred,
            typename Proj = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_sentinel_for_v<Sent, Iter> &&
                hpx::parallel::traits::is_projected_v<Proj, Iter> &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<Iter>::value_type
                >
            )>
        // clang-format on
        friend Iter tag_fallback_invoke(
            find_if_t, Iter first, Sent last, Pred pred, Proj proj = Proj())
        {
            static_assert(hpx::traits::is_input_iterator_v<Iter>,
                "Requires at least input iterator.");

            return hpx::parallel::detail::find_if<Iter>().call(
                hpx::execution::seq, first, last, HPX_MOVE(pred),
                HPX_MOVE(proj));
        }

        // clang-format off
        template <typename Rng, typename Pred,
            typename Proj = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range_v<Rng> &&
                hpx::parallel::traits::is_projected_range_v<Proj, Rng> &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<
                        hpx::traits::range_iterator_t<Rng>
                    >::value_type
                >
            )>
        // clang-format on
        friend hpx::traits::range_iterator_t<Rng> tag_fallback_invoke(
            find_if_t, Rng&& rng, Pred pred, Proj proj = Proj())
        {
            using iterator_type = hpx::traits::range_iterator_t<Rng>;

            static_assert(hpx::traits::is_input_iterator_v<iterator_type>,
                "Requires at least input iterator.");

            return hpx::parallel::detail::find_if<iterator_type>().call(
                hpx::execution::seq, hpx::util::begin(rng), hpx::util::end(rng),
                HPX_MOVE(pred), HPX_MOVE(proj));
        }
    } find_if{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::find_if_not
    inline constexpr struct find_if_not_t final
      : hpx::detail::tag_parallel_algorithm<find_if_not_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent, typename Pred,
            typename Proj = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_sentinel_for_v<Sent, Iter> &&
                hpx::parallel::traits::is_projected_v<Proj, Iter> &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<Iter>::value_type
                >
            )>
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy, Iter>
        tag_fallback_invoke(find_if_not_t, ExPolicy&& policy, Iter first,
            Sent last, Pred pred, Proj proj = Proj())
        {
            static_assert(hpx::traits::is_forward_iterator_v<Iter>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::find_if_not<Iter>().call(
                HPX_FORWARD(ExPolicy, policy), first, last, HPX_MOVE(pred),
                HPX_MOVE(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename Pred,
            typename Proj = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_range_v<Rng> &&
                hpx::parallel::traits::is_projected_range_v<Proj, Rng> &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<
                        hpx::traits::range_iterator_t<Rng>
                    >::value_type
                >
            )>
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy,
            hpx::traits::range_iterator_t<Rng>>
        tag_fallback_invoke(find_if_not_t, ExPolicy&& policy, Rng&& rng,
            Pred pred, Proj proj = Proj())
        {
            using iterator_type = hpx::traits::range_iterator_t<Rng>;

            static_assert(hpx::traits::is_forward_iterator_v<iterator_type>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::find_if_not<iterator_type>().call(
                HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
                hpx::util::end(rng), HPX_MOVE(pred), HPX_MOVE(proj));
        }

        // clang-format off
        template <typename Iter, typename Sent, typename Pred,
            typename Proj = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_sentinel_for_v<Sent, Iter> &&
                hpx::parallel::traits::is_projected_v<Proj, Iter> &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<Iter>::value_type
                >
            )>
        // clang-format on
        friend Iter tag_fallback_invoke(
            find_if_not_t, Iter first, Sent last, Pred pred, Proj proj = Proj())
        {
            static_assert(hpx::traits::is_input_iterator_v<Iter>,
                "Requires at least input iterator.");

            return hpx::parallel::detail::find_if_not<Iter>().call(
                hpx::execution::seq, first, last, HPX_MOVE(pred),
                HPX_MOVE(proj));
        }

        // clang-format off
        template <typename Rng, typename Pred,
            typename Proj = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range_v<Rng> &&
                hpx::parallel::traits::is_projected_range_v<Proj, Rng> &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<
                        hpx::traits::range_iterator_t<Rng>
                    >::value_type
                >
            )>
        // clang-format on
        friend hpx::traits::range_iterator_t<Rng> tag_fallback_invoke(
            find_if_not_t, Rng&& rng, Pred pred, Proj proj = Proj())
        {
            using iterator_type = hpx::traits::range_iterator_t<Rng>;

            static_assert(hpx::traits::is_input_iterator_v<iterator_type>,
                "Requires at least input iterator.");

            return hpx::parallel::detail::find_if_not<iterator_type>().call(
                hpx::execution::seq, hpx::util::begin(rng), hpx::util::end(rng),
                HPX_MOVE(pred), HPX_MOVE(proj));
        }
    } find_if_not{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::find_end
    inline constexpr struct find_end_t final
      : hpx::detail::tag_parallel_algorithm<find_end_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Rng1, typename Rng2,
            typename Pred = equal_to,
            typename Proj1 = hpx::identity,
            typename Proj2 = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::parallel::traits::is_projected_range_v<Proj1, Rng1> &&
                hpx::parallel::traits::is_projected_range_v<Proj2, Rng2> &&
                hpx::parallel::traits::is_indirect_callable_v<ExPolicy, Pred,
                    hpx::parallel::traits::projected_range<Proj1, Rng1>,
                    hpx::parallel::traits::projected_range<Proj2, Rng2>
                >
            )>
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy,
            hpx::traits::range_iterator_t<Rng1>>
        tag_fallback_invoke(find_end_t, ExPolicy&& policy, Rng1&& rng1,
            Rng2&& rng2, Pred op = Pred(), Proj1 proj1 = Proj1(),
            Proj2 proj2 = Proj2())
        {
            using iterator_type = hpx::traits::range_iterator_t<Rng1>;

            static_assert(hpx::traits::is_forward_iterator_v<iterator_type>,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator<
                              hpx::traits::range_iterator_t<Rng2>>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::find_end<iterator_type>().call(
                HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng1),
                hpx::util::end(rng1), hpx::util::begin(rng2),
                hpx::util::end(rng2), HPX_MOVE(op), HPX_MOVE(proj1),
                HPX_MOVE(proj2));
        }

        // clang-format off
        template <typename ExPolicy, typename Iter1, typename Sent1,
            typename Iter2, typename Sent2, typename Pred = equal_to,
            typename Proj1 = hpx::identity,
            typename Proj2 = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_sentinel_for_v<Sent1, Iter1> &&
                hpx::traits::is_sentinel_for_v<Sent2, Iter2> &&
                hpx::parallel::traits::is_indirect_callable_v<ExPolicy, Pred,
                    hpx::parallel::traits::projected<Proj1, Iter1>,
                    hpx::parallel::traits::projected<Proj2, Iter2>
                >
            )>
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy, Iter1>
        tag_fallback_invoke(find_end_t, ExPolicy&& policy, Iter1 first1,
            Sent1 last1, Iter2 first2, Sent2 last2, Pred op = Pred(),
            Proj1 proj1 = Proj1(), Proj2 proj2 = Proj2())
        {
            static_assert(hpx::traits::is_forward_iterator_v<Iter1>,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator_v<Iter2>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::find_end<Iter1>().call(
                HPX_FORWARD(ExPolicy, policy), first1, last1, first2, last2,
                HPX_MOVE(op), HPX_MOVE(proj1), HPX_MOVE(proj2));
        }

        // clang-format off
        template <typename Rng1, typename Rng2, typename Pred = equal_to,
            typename Proj1 = hpx::identity,
            typename Proj2 = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::traits::is_projected_range_v<Proj1, Rng1> &&
                hpx::parallel::traits::is_projected_range_v<Proj2, Rng2> &&
                hpx::parallel::traits::is_indirect_callable_v<
                    hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected_range<Proj1, Rng1>,
                    hpx::parallel::traits::projected_range<Proj2, Rng2>
                >
            )>
        // clang-format on
        friend hpx::traits::range_iterator_t<Rng1> tag_fallback_invoke(
            find_end_t, Rng1&& rng1, Rng2&& rng2, Pred op = Pred(),
            Proj1 proj1 = Proj1(), Proj2 proj2 = Proj2())
        {
            using iterator_type = hpx::traits::range_iterator_t<Rng1>;

            static_assert(hpx::traits::is_forward_iterator_v<iterator_type>,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator<
                              hpx::traits::range_iterator_t<Rng2>>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::find_end<iterator_type>().call(
                hpx::execution::seq, hpx::util::begin(rng1),
                hpx::util::end(rng1), hpx::util::begin(rng2),
                hpx::util::end(rng2), HPX_MOVE(op), HPX_MOVE(proj1),
                HPX_MOVE(proj2));
        }

        // clang-format off
        template <typename Iter1, typename Sent1,
            typename Iter2, typename Sent2, typename Pred = equal_to,
            typename Proj1 = hpx::identity,
            typename Proj2 = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_sentinel_for_v<Sent1, Iter1> &&
                hpx::traits::is_sentinel_for_v<Sent2, Iter2> &&
                hpx::parallel::traits::is_indirect_callable_v<
                    hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected<Proj1, Iter1>,
                    hpx::parallel::traits::projected<Proj2, Iter2>
                >
            )>
        // clang-format on
        friend Iter1 tag_fallback_invoke(find_end_t, Iter1 first1, Sent1 last1,
            Iter2 first2, Sent2 last2, Pred op = Pred(), Proj1 proj1 = Proj1(),
            Proj2 proj2 = Proj2())
        {
            static_assert(hpx::traits::is_forward_iterator_v<Iter1>,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator_v<Iter2>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::find_end<Iter1>().call(
                hpx::execution::seq, first1, last1, first2, last2, HPX_MOVE(op),
                HPX_MOVE(proj1), HPX_MOVE(proj2));
        }
    } find_end{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::find_first_of
    inline constexpr struct find_first_of_t final
      : hpx::detail::tag_parallel_algorithm<find_first_of_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Rng1, typename Rng2,
            typename Pred = equal_to,
            typename Proj1 = hpx::identity,
            typename Proj2 = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::parallel::traits::is_projected_range_v<Proj1, Rng1> &&
                hpx::parallel::traits::is_projected_range_v<Proj2, Rng2> &&
                hpx::parallel::traits::is_indirect_callable_v<ExPolicy, Pred,
                    hpx::parallel::traits::projected_range<Proj1, Rng1>,
                    hpx::parallel::traits::projected_range<Proj2, Rng2>
                >
            )>
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy,
            hpx::traits::range_iterator_t<Rng1>>
        tag_fallback_invoke(find_first_of_t, ExPolicy&& policy, Rng1&& rng1,
            Rng2&& rng2, Pred op = Pred(), Proj1 proj1 = Proj1(),
            Proj2 proj2 = Proj2())
        {
            using iterator_type = hpx::traits::range_iterator_t<Rng1>;

            static_assert(hpx::traits::is_forward_iterator_v<iterator_type>,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator<
                              hpx::traits::range_iterator_t<Rng2>>::value,
                "Subsequence requires at least forward iterator.");

            return hpx::parallel::detail::find_first_of<iterator_type>().call(
                HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng1),
                hpx::util::end(rng1), hpx::util::begin(rng2),
                hpx::util::end(rng2), HPX_MOVE(op), HPX_MOVE(proj1),
                HPX_MOVE(proj2));
        }

        // clang-format off
        template <typename ExPolicy, typename Iter1, typename Sent1,
            typename Iter2, typename Sent2, typename Pred = equal_to,
            typename Proj1 = hpx::identity,
            typename Proj2 = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_sentinel_for_v<Sent1, Iter1> &&
                hpx::traits::is_sentinel_for_v<Sent2, Iter2> &&
                hpx::parallel::traits::is_indirect_callable_v<ExPolicy, Pred,
                    hpx::parallel::traits::projected<Proj1, Iter1>,
                    hpx::parallel::traits::projected<Proj2, Iter2>
                >
            )>
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy, Iter1>
        tag_fallback_invoke(find_first_of_t, ExPolicy&& policy, Iter1 first1,
            Sent1 last1, Iter2 first2, Sent2 last2, Pred op = Pred(),
            Proj1 proj1 = Proj1(), Proj2 proj2 = Proj2())
        {
            static_assert(hpx::traits::is_forward_iterator_v<Iter1>,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator_v<Iter2>,
                "Subsequence requires at least forward iterator.");

            return hpx::parallel::detail::find_first_of<Iter1>().call(
                HPX_FORWARD(ExPolicy, policy), first1, last1, first2, last2,
                HPX_MOVE(op), HPX_MOVE(proj1), HPX_MOVE(proj2));
        }

        // clang-format off
        template <typename Rng1, typename Rng2, typename Pred = equal_to,
            typename Proj1 = hpx::identity,
            typename Proj2 = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::traits::is_projected_range_v<Proj1, Rng1> &&
                hpx::parallel::traits::is_projected_range_v<Proj2, Rng2> &&
                hpx::parallel::traits::is_indirect_callable_v<
                    hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected_range<Proj1, Rng1>,
                    hpx::parallel::traits::projected_range<Proj2, Rng2>
                >
            )>
        // clang-format on
        friend hpx::traits::range_iterator_t<Rng1> tag_fallback_invoke(
            find_first_of_t, Rng1&& rng1, Rng2&& rng2, Pred op = Pred(),
            Proj1 proj1 = Proj1(), Proj2 proj2 = Proj2())
        {
            using iterator_type = hpx::traits::range_iterator_t<Rng1>;

            static_assert(hpx::traits::is_forward_iterator_v<iterator_type>,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator<
                              hpx::traits::range_iterator_t<Rng2>>::value,
                "Subsequence requires at least forward iterator.");

            return hpx::parallel::detail::find_first_of<iterator_type>().call(
                hpx::execution::seq, hpx::util::begin(rng1),
                hpx::util::end(rng1), hpx::util::begin(rng2),
                hpx::util::end(rng2), HPX_MOVE(op), HPX_MOVE(proj1),
                HPX_MOVE(proj2));
        }

        // clang-format off
        template <typename Iter1, typename Sent1, typename Iter2, typename Sent2,
            typename Pred = equal_to,
            typename Proj1 = hpx::identity,
            typename Proj2 = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_sentinel_for_v<Sent1, Iter1> &&
                hpx::traits::is_sentinel_for_v<Sent2, Iter2> &&
                hpx::parallel::traits::is_indirect_callable_v<
                hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected<Proj1, Iter1>,
                    hpx::parallel::traits::projected<Proj2, Iter2>
                >
            )>
        // clang-format on
        friend Iter1 tag_fallback_invoke(find_first_of_t, Iter1 first1,
            Sent1 last1, Iter2 first2, Sent2 last2, Pred op = Pred(),
            Proj1 proj1 = Proj1(), Proj2 proj2 = Proj2())
        {
            static_assert(hpx::traits::is_forward_iterator_v<Iter1>,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator_v<Iter2>,
                "Subsequence requires at least forward iterator.");

            return hpx::parallel::detail::find_first_of<Iter1>().call(
                hpx::execution::seq, first1, last1, first2, last2, HPX_MOVE(op),
                HPX_MOVE(proj1), HPX_MOVE(proj2));
        }
    } find_first_of{};
}    // namespace hpx::ranges

#endif    // DOXYGEN
