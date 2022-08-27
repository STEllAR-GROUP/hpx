//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// make inspect happy: hpxinspect:nominmax

/// \file parallel/container_algorithms/minmax.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    /// Finds the smallest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: Exactly \a max(N-1, 0) comparisons, where
    ///                     N = std::distance(first, last).
    ///
    /// \tparam FwdIter     The type of the source iterator used (deduced).
    ///                     The iterator type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Sent        The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     the left argument is less than the right element.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such that objects of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type1.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparisons in the parallel \a min_element algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a min_element algorithm  returns \a FwdIter.
    ///           The \a min_element algorithm returns the iterator to the
    ///           smallest element in the range [first, last). If several
    ///           elements in the range are equivalent to the smallest element,
    ///           returns the iterator to the first such element. Returns last
    ///           if the range is empty.
    ///
    template <typename FwdIter, typename Sent, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    FwdIter min_element(
        FwdIter first, Sent last, F&& f = F(), Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Finds the smallest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: Exactly \a max(N-1, 0) comparisons, where
    ///                     N = std::distance(first, last).
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     the left argument is less than the right element.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such that objects of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type1.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparisons in the parallel \a min_element algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a min_element algorithm returns a \a hpx::traits::range_iterator<Rng>::type otherwise.
    ///           The \a min_element algorithm returns the iterator to the
    ///           smallest element in the range [first, last). If several
    ///           elements in the range are equivalent to the smallest element,
    ///           returns the iterator to the first such element. Returns last
    ///           if the range is empty.
    ///
    template <typename Rng, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    hpx::traits::range_iterator_t<Rng> min_element(
        Rng&& rng, F&& f = F(), Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Finds the smallest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: Exactly \a max(N-1, 0) comparisons, where
    ///                     N = std::distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterator used (deduced).
    ///                     The iterator type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Sent        The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a min_element requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     the left argument is less than the right element.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such that objects of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type1.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparisons in the parallel \a min_element algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a min_element algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a min_element algorithm returns a \a hpx::future<FwdIter>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy
    ///           and returns \a FwdIter otherwise.
    ///           The \a min_element algorithm returns the iterator to the
    ///           smallest element in the range [first, last). If several
    ///           elements in the range are equivalent to the smallest element,
    ///           returns the iterator to the first such element. Returns last
    ///           if the range is empty.
    ///
    template <typename ExPolicy, typename FwdIter, typename Sent, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    typename parallel::util::detail::algorithm_result<ExPolicy, FwdIter>::type
    min_element(ExPolicy&& policy, FwdIter first, Sent sent, F&& f = F(),
        Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Finds the smallest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: Exactly \a max(N-1, 0) comparisons, where
    ///                     N = std::distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a min_element requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     the left argument is less than the right element.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such that objects of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type1.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparisons in the parallel \a min_element algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a min_element algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a min_element algorithm returns a \a hpx::future<hpx::traits::range_iterator<Rng>::type>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy
    ///           and returns \a FwdIter otherwise.
    ///           The \a min_element algorithm returns the iterator to the
    ///           smallest element in the range [first, last). If several
    ///           elements in the range are equivalent to the smallest element,
    ///           returns the iterator to the first such element. Returns last
    ///           if the range is empty.
    ///
    template <typename ExPolicy, typename Rng, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy,
        hpx::traits::range_iterator_t<Rng>>::type
    min_element(
        ExPolicy&& policy, Rng&& rng, F&& f = F(), Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Finds the greatest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: Exactly \a max(N-1, 0) comparisons, where
    ///                     N = std::distance(first, last).
    ///
    /// \tparam FwdIter     The type of the source iterator used (deduced).
    ///                     The iterator type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Sent        The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     This argument is optional and defaults to std::less.
    ///                     the left argument is less than the right element.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such that objects of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type1.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparisons in the parallel \a max_element algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a max_element algorithm returns a \a FwdIter.
    ///           The \a max_element algorithm returns the iterator to the
    ///           smallest element in the range [first, last). If several
    ///           elements in the range are equivalent to the smallest element,
    ///           returns the iterator to the first such element. Returns last
    ///           if the range is empty.
    ///
    template <typename FwdIter, typename Sent, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    FwdIter max_element(
        FwdIter first, Sent sent, F&& f = F(), Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Finds the greatest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: Exactly \a max(N-1, 0) comparisons, where
    ///                     N = std::distance(first, last).
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     This argument is optional and defaults to std::less.
    ///                     the left argument is less than the right element.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such that objects of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type1.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparisons in the parallel \a max_element algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a max_element algorithm returns a \a hpx::traits::range_iterator<Rng>::type otherwise.
    ///           The \a max_element algorithm returns the iterator to the
    ///           smallest element in the range [first, last). If several
    ///           elements in the range are equivalent to the smallest element,
    ///           returns the iterator to the first such element. Returns last
    ///           if the range is empty.
    ///
    template <typename Rng, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    hpx::traits::range_iterator_t<Rng> max_element(
        Rng&& rng, F&& f = F(), Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Finds the greatest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: Exactly \a max(N-1, 0) comparisons, where
    ///                     N = std::distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterator used (deduced).
    ///                     The iterator type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Sent        The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a max_element requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     This argument is optional and defaults to std::less.
    ///                     the left argument is less than the right element.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such that objects of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type1.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparisons in the parallel \a max_element algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a max_element algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a max_element algorithm returns a \a hpx::future<FwdIter>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy
    ///           and returns \a FwdIter otherwise.
    ///           The \a max_element algorithm returns the iterator to the
    ///           smallest element in the range [first, last). If several
    ///           elements in the range are equivalent to the smallest element,
    ///           returns the iterator to the first such element. Returns last
    ///           if the range is empty.
    ///
    template <typename ExPolicy, typename FwdIter, typename Sent, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    typename parallel::util::detail::algorithm_result<ExPolicy, FwdIter>::type
    max_element(ExPolicy&& policy, FwdIter first, Sent sent, F&& f = F(),
        Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Finds the greatest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: Exactly \a max(N-1, 0) comparisons, where
    ///                     N = std::distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a max_element requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     This argument is optional and defaults to std::less.
    ///                     the left argument is less than the right element.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such that objects of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type1.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparisons in the parallel \a max_element algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a max_element algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a max_element algorithm returns a \a hpx::future<hpx::traits::range_iterator<Rng>::type>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy
    ///           and returns \a FwdIter otherwise.
    ///           The \a max_element algorithm returns the iterator to the
    ///           smallest element in the range [first, last). If several
    ///           elements in the range are equivalent to the smallest element,
    ///           returns the iterator to the first such element. Returns last
    ///           if the range is empty.
    ///
    template <typename ExPolicy, typename Rng, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy,
        hpx::traits::range_iterator_t<Rng>>::type
    max_element(
        ExPolicy&& policy, Rng&& rng, F&& f = F(), Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Finds the greatest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: At most \a max(floor(3/2*(N-1)), 0) applications of
    ///                     the predicate, where N = std::distance(first, last).
    ///
    /// \tparam FwdIter     The type of the source iterator used (deduced).
    ///                     The iterator type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Sent        The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     the left argument is less than the right element.
    ///                     This argument is optional and defaults to std::less.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such that objects of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type1.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a minmax_element algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a minmax_element algorithm returns a
    ///           \a minmax_element_result<FwdIter, FwdIter>
    ///           The \a minmax_element algorithm returns a min_max_result consisting of
    ///           an iterator to the smallest element as the min element and
    ///           an iterator to the greatest element as the max element. Returns
    ///           minmax_element_result{first, first} if the range is empty. If
    ///           several elements are equivalent to the smallest element, the
    ///           iterator to the first such element is returned. If several
    ///           elements are equivalent to the largest element, the iterator
    ///           to the last such element is returned.
    ///
    template <typename FwdIter, typename Sent, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    minmax_element_result<FwdIter, FwdIter> minmax_element(
        FwdIter first, Sent last, F&& f = F(), Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Finds the greatest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: At most \a max(floor(3/2*(N-1)), 0) applications of
    ///                     the predicate, where N = std::distance(first, last).
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     the left argument is less than the right element.
    ///                     This argument is optional and defaults to std::less.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such that objects of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type1.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a minmax_element algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a minmax_element algorithm returns a
    /// \a minmax_element_result<hpx::traits::range_iterator<Rng>::type
    /// , hpx::traits::range_iterator<Rng>::type>
    ///           The \a minmax_element algorithm returns a min_max_result consisting of
    ///           an range iterator to the smallest element as the min element and
    ///           an range iterator to the greatest element as the max element. If
    ///           several elements are equivalent to the smallest element, the
    ///           iterator to the first such element is returned. If several
    ///           elements are equivalent to the largest element, the iterator
    ///           to the last such element is returned.
    ///
    template <typename Rng, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    minmax_element_result<hpx::traits::range_iterator_t<Rng>,
        hpx::traits::range_iterator_t<Rng>>
    minmax_element(Rng&& rng, F&& f = F(), Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Finds the greatest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: At most \a max(floor(3/2*(N-1)), 0) applications of
    ///                     the predicate, where N = std::distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterator used (deduced).
    ///                     The iterator type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Sent        The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a minmax_element requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     the left argument is less than the right element.
    ///                     This argument is optional and defaults to std::less.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such that objects of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type1.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparisons in the parallel \a minmax_element algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a minmax_element algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a minmax_element algorithm returns a \a minmax_element_result<FwdIter, FwdIter>
    ///           The \a minmax_element algorithm returns a min_max_result consisting of
    ///           an iterator to the smallest element as the min element and
    ///           an iterator to the greatest element as the max element. Returns
    ///           minmax_element_result{first, first} if the range is empty. If
    ///           several elements are equivalent to the smallest element, the
    ///           iterator to the first such element is returned. If several
    ///           elements are equivalent to the largest element, the iterator
    ///           to the last such element is returned.
    ///
    template <typename ExPolicy, typename FwdIter, typename Sent, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        minmax_element_result<FwdIter, FwdIter>>::type
    minmax_element(ExPolicy&& policy, FwdIter first, Sent last, F&& f = F(),
        Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Finds the greatest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: At most \a max(floor(3/2*(N-1)), 0) applications of
    ///                     the predicate, where N = std::distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a minmax_element requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     the left argument is less than the right element.
    ///                     This argument is optional and defaults to std::less.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such that objects of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type1.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparisons in the parallel \a minmax_element algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a minmax_element algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a minmax_element algorithm returns a
    ///           \a minmax_element_result<hpx::traits::range_iterator<Rng>::type,
    ///           hpx::traits::range_iterator<Rng>::type>
    ///           The \a minmax_element algorithm returns a min_max_result consisting of
    ///           an range iterator to the smallest element as the min element and
    ///           an range iterator to the greatest element as the max element. If
    ///           several elements are equivalent to the smallest element, the
    ///           iterator to the first such element is returned. If several
    ///           elements are equivalent to the largest element, the iterator
    ///           to the last such element is returned.
    ///
    template <typename ExPolicy, typename Rng, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy,
        minmax_element_result<hpx::traits::range_iterator_t<Rng>,
            hpx::traits::range_iterator_t<Rng>>>::type
    minmax_element(
        ExPolicy&& policy, Rng&& rng, F&& f = F(), Proj&& proj = Proj());

}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/parallel/algorithms/minmax.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {

    template <typename T>
    using minmax_element_result = hpx::parallel::util::min_max_result<T>;

    // clang-format off
    template <typename ExPolicy, typename Rng,
        typename Proj = util::projection_identity,
        typename F = detail::less,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_range_v<Rng> &&
            hpx::parallel::traits::is_projected_range_v<Proj, Rng> &&
            hpx::parallel::traits::is_indirect_callable_v<
                ExPolicy, F,
                hpx::parallel::traits::projected_range<Proj, Rng>,
                hpx::parallel::traits::projected_range<Proj, Rng>
            >
        )>
    HPX_DEPRECATED_V(1, 7,
        "hpx::parallel::min_element is deprecated, use "
        "hpx::ranges::min_element instead")
        hpx::parallel::util::detail::algorithm_result_t<ExPolicy,
            typename hpx::traits::range_traits<Rng>::iterator_type>
    min_element(ExPolicy&& policy, Rng&& rng, F&& f = F(), Proj&& proj = Proj())
    // clang-format on
    {
        return hpx::parallel::v1::detail::min_element<
            hpx::traits::range_iterator_t<Rng>>()
            .call(HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
                hpx::util::end(rng), HPX_FORWARD(F, f),
                HPX_FORWARD(Proj, proj));
    }

    // clang-format off
    template <typename ExPolicy, typename Rng,
        typename Proj = util::projection_identity,
        typename F = detail::less,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_range_v<Rng> &&
            hpx::parallel::traits::is_projected_range_v<Proj, Rng> &&
            hpx::parallel::traits::is_indirect_callable_v<
                ExPolicy, F,
                hpx::parallel::traits::projected_range<Proj, Rng>,
                hpx::parallel::traits::projected_range<Proj, Rng>
            >
        )>
    HPX_DEPRECATED_V(1, 7,
        "hpx::parallel::max_element is deprecated, use "
        "hpx::ranges::max_element instead")
        hpx::parallel::util::detail::algorithm_result_t<ExPolicy,
            typename hpx::traits::range_traits<Rng>::iterator_type>
    max_element(ExPolicy&& policy, Rng&& rng, F&& f = F(), Proj&& proj = Proj())
    // clang-format on
    {
        return hpx::parallel::v1::detail::max_element<
            hpx::traits::range_iterator_t<Rng>>()
            .call(HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
                hpx::util::end(rng), HPX_FORWARD(F, f),
                HPX_FORWARD(Proj, proj));
    }

#if defined(HPX_MSVC)
#pragma push_macro("min")
#pragma push_macro("max")
#undef min
#undef max
#endif

    // clang-format off
    template <typename ExPolicy, typename Rng,
        typename Proj = util::projection_identity, typename F = detail::less,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_range_v<Rng> &&
            hpx::parallel::traits::is_projected_range_v<Proj, Rng> &&
            hpx::parallel::traits::is_indirect_callable_v<
                ExPolicy, F,
                hpx::parallel::traits::projected_range<Proj, Rng>,
                hpx::parallel::traits::projected_range<Proj, Rng>
            >
        )>
    HPX_DEPRECATED_V(1, 7,
        "hpx::parallel::minmax_element is deprecated, use "
        "hpx::ranges::minmax_element instead")
        hpx::parallel::util::detail::algorithm_result_t<ExPolicy,
            minmax_element_result<hpx::traits::range_iterator_t<Rng>>>
    minmax_element(
        ExPolicy&& policy, Rng&& rng, F&& f = F(), Proj&& proj = Proj())
    // clang-format on
    {
        return hpx::parallel::v1::detail::minmax_element<
            hpx::traits::range_iterator_t<Rng>>()
            .call(HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
                hpx::util::end(rng), HPX_FORWARD(F, f),
                HPX_FORWARD(Proj, proj));
    }

#if defined(HPX_MSVC)
#pragma pop_macro("min")
#pragma pop_macro("max")
#endif
}}}    // namespace hpx::parallel::v1

namespace hpx { namespace ranges {

    /// `minmax_element_result` is equivalent to
    /// `hpx::parallel::util::min_max_result`
    template <typename T>
    using minmax_element_result = hpx::parallel::util::min_max_result<T>;

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::min_element
    inline constexpr struct min_element_t final
      : hpx::detail::tag_parallel_algorithm<min_element_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Sent,
            typename F = hpx::parallel::v1::detail::less,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<FwdIter> &&
                hpx::parallel::traits::is_projected_v<Proj, FwdIter> &&
                hpx::traits::is_sentinel_for_v<Sent, FwdIter> &&
                hpx::parallel::traits::is_indirect_callable_v<
                    hpx::execution::sequenced_policy, F,
                    hpx::parallel::traits::projected<Proj, FwdIter>,
                    hpx::parallel::traits::projected<Proj, FwdIter>
                >
            )>
        // clang-format on
        friend FwdIter tag_fallback_invoke(hpx::ranges::min_element_t,
            FwdIter first, Sent last, F&& f = F(), Proj&& proj = Proj())
        {
            static_assert((hpx::traits::is_forward_iterator_v<FwdIter>),
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::min_element<FwdIter>().call(
                hpx::execution::seq, first, last, HPX_FORWARD(F, f),
                HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename Rng,
            typename F = hpx::parallel::v1::detail::less,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range_v<Rng> &&
                hpx::parallel::traits::is_projected_range_v<Proj, Rng> &&
                hpx::parallel::traits::is_indirect_callable_v<
                    hpx::execution::sequenced_policy, F,
                    hpx::parallel::traits::projected_range<Proj, Rng>,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >
            )>
        // clang-format on
        friend hpx::traits::range_iterator_t<Rng> tag_fallback_invoke(
            hpx::ranges::min_element_t, Rng&& rng, F&& f = F(),
            Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_forward_iterator_v<
                              hpx::traits::range_iterator_t<Rng>>,
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::min_element<
                hpx::traits::range_iterator_t<Rng>>()
                .call(hpx::execution::seq, hpx::util::begin(rng),
                    hpx::util::end(rng), HPX_FORWARD(F, f),
                    HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename F = hpx::parallel::v1::detail::less,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<FwdIter> &&
                hpx::parallel::traits::is_projected_v<Proj, FwdIter> &&
                hpx::traits::is_sentinel_for_v<Sent, FwdIter> &&
                hpx::parallel::traits::is_indirect_callable_v<
                    ExPolicy, F,
                    hpx::parallel::traits::projected<Proj, FwdIter>,
                    hpx::parallel::traits::projected<Proj, FwdIter>
                >
            )>
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy,
            FwdIter>
        tag_fallback_invoke(hpx::ranges::min_element_t, ExPolicy&& policy,
            FwdIter first, Sent last, F&& f = F(), Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::min_element<FwdIter>().call(
                HPX_FORWARD(ExPolicy, policy), first, last, HPX_FORWARD(F, f),
                HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,
            typename F = hpx::parallel::v1::detail::less,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_range_v<Rng> &&
                hpx::parallel::traits::is_projected_range_v<Proj, Rng> &&
                hpx::parallel::traits::is_indirect_callable_v<
                    ExPolicy, F,
                    hpx::parallel::traits::projected_range<Proj, Rng>,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >
            )>
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy,
            hpx::traits::range_iterator_t<Rng>>
        tag_fallback_invoke(hpx::ranges::min_element_t, ExPolicy&& policy,
            Rng&& rng, F&& f = F(), Proj&& proj = Proj())
        {
            static_assert((hpx::traits::is_forward_iterator_v<
                              hpx::traits::range_iterator_t<Rng>>),
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::min_element<
                hpx::traits::range_iterator_t<Rng>>()
                .call(HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
                    hpx::util::end(rng), HPX_FORWARD(F, f),
                    HPX_FORWARD(Proj, proj));
        }
    } min_element{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::max_element
    inline constexpr struct max_element_t final
      : hpx::detail::tag_parallel_algorithm<max_element_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Sent,
            typename F = hpx::parallel::v1::detail::less,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<FwdIter> &&
                hpx::parallel::traits::is_projected_v<Proj, FwdIter> &&
                hpx::traits::is_sentinel_for_v<Sent, FwdIter> &&
                hpx::parallel::traits::is_indirect_callable_v<
                    hpx::execution::sequenced_policy, F,
                    hpx::parallel::traits::projected<Proj, FwdIter>,
                    hpx::parallel::traits::projected<Proj, FwdIter>
                >
            )>
        // clang-format on
        friend FwdIter tag_fallback_invoke(hpx::ranges::max_element_t,
            FwdIter first, Sent last, F&& f = F(), Proj&& proj = Proj())
        {
            static_assert((hpx::traits::is_forward_iterator_v<FwdIter>),
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::max_element<FwdIter>().call(
                hpx::execution::seq, first, last, HPX_FORWARD(F, f),
                HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename Rng,
            typename F = hpx::parallel::v1::detail::less,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range_v<Rng> &&
                hpx::parallel::traits::is_projected_range_v<Proj, Rng> &&
                hpx::parallel::traits::is_indirect_callable_v<
                    hpx::execution::sequenced_policy, F,
                    hpx::parallel::traits::projected_range<Proj, Rng>,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >
            )>
        // clang-format on
        friend hpx::traits::range_iterator_t<Rng> tag_fallback_invoke(
            hpx::ranges::max_element_t, Rng&& rng, F&& f = F(),
            Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_forward_iterator_v<
                              hpx::traits::range_iterator_t<Rng>>,
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::max_element<
                hpx::traits::range_iterator_t<Rng>>()
                .call(hpx::execution::seq, hpx::util::begin(rng),
                    hpx::util::end(rng), HPX_FORWARD(F, f),
                    HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename F = hpx::parallel::v1::detail::less,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<FwdIter> &&
                hpx::parallel::traits::is_projected_v<Proj, FwdIter> &&
                hpx::traits::is_sentinel_for_v<Sent, FwdIter> &&
                hpx::parallel::traits::is_indirect_callable_v<
                    ExPolicy, F,
                    hpx::parallel::traits::projected<Proj, FwdIter>,
                    hpx::parallel::traits::projected<Proj, FwdIter>
                >
            )>
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy,
            FwdIter>
        tag_fallback_invoke(hpx::ranges::max_element_t, ExPolicy&& policy,
            FwdIter first, Sent last, F&& f = F(), Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::max_element<FwdIter>().call(
                HPX_FORWARD(ExPolicy, policy), first, last, HPX_FORWARD(F, f),
                HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,
            typename F = hpx::parallel::v1::detail::less,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_range_v<Rng> &&
                hpx::parallel::traits::is_projected_range_v<Proj, Rng> &&
                hpx::parallel::traits::is_indirect_callable_v<
                    ExPolicy, F,
                    hpx::parallel::traits::projected_range<Proj, Rng>,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >
            )>
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy,
            hpx::traits::range_iterator_t<Rng>>
        tag_fallback_invoke(hpx::ranges::max_element_t, ExPolicy&& policy,
            Rng&& rng, F&& f = F(), Proj&& proj = Proj())
        {
            static_assert((hpx::traits::is_forward_iterator_v<
                              hpx::traits::range_iterator_t<Rng>>),
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::max_element<
                hpx::traits::range_iterator_t<Rng>>()
                .call(HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
                    hpx::util::end(rng), HPX_FORWARD(F, f),
                    HPX_FORWARD(Proj, proj));
        }
    } max_element{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::minmax_element
    inline constexpr struct minmax_element_t final
      : hpx::detail::tag_parallel_algorithm<minmax_element_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Sent,
            typename F = hpx::parallel::v1::detail::less,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<FwdIter> &&
                hpx::parallel::traits::is_projected_v<Proj, FwdIter> &&
                hpx::traits::is_sentinel_for_v<Sent, FwdIter> &&
                hpx::parallel::traits::is_indirect_callable_v<
                    hpx::execution::sequenced_policy, F,
                    hpx::parallel::traits::projected<Proj, FwdIter>,
                    hpx::parallel::traits::projected<Proj, FwdIter>
                >
            )>
        // clang-format on
        friend minmax_element_result<FwdIter> tag_fallback_invoke(
            hpx::ranges::minmax_element_t, FwdIter first, Sent last,
            F&& f = F(), Proj&& proj = Proj())
        {
            static_assert((hpx::traits::is_forward_iterator_v<FwdIter>),
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::minmax_element<FwdIter>().call(
                hpx::execution::seq, first, last, HPX_FORWARD(F, f),
                HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename Rng,
            typename F = hpx::parallel::v1::detail::less,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range_v<Rng> &&
                hpx::parallel::traits::is_projected_range_v<Proj, Rng> &&
                hpx::parallel::traits::is_indirect_callable_v<
                    hpx::execution::sequenced_policy, F,
                    hpx::parallel::traits::projected_range<Proj, Rng>,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >
            )>
        // clang-format on
        friend minmax_element_result<hpx::traits::range_iterator_t<Rng>>
        tag_fallback_invoke(hpx::ranges::minmax_element_t, Rng&& rng,
            F&& f = F(), Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_forward_iterator_v<
                              hpx::traits::range_iterator_t<Rng>>,
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::minmax_element<
                hpx::traits::range_iterator_t<Rng>>()
                .call(hpx::execution::seq, hpx::util::begin(rng),
                    hpx::util::end(rng), HPX_FORWARD(F, f),
                    HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename F = hpx::parallel::v1::detail::less,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<FwdIter> &&
                hpx::parallel::traits::is_projected_v<Proj, FwdIter> &&
                hpx::traits::is_sentinel_for_v<Sent, FwdIter> &&
                hpx::parallel::traits::is_indirect_callable_v<
                    ExPolicy, F,
                    hpx::parallel::traits::projected<Proj, FwdIter>,
                    hpx::parallel::traits::projected<Proj, FwdIter>
                >
            )>
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy,
            minmax_element_result<FwdIter>>
        tag_fallback_invoke(hpx::ranges::minmax_element_t, ExPolicy&& policy,
            FwdIter first, Sent last, F&& f = F(), Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::minmax_element<FwdIter>().call(
                HPX_FORWARD(ExPolicy, policy), first, last, HPX_FORWARD(F, f),
                HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,
            typename F = hpx::parallel::v1::detail::less,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_range_v<Rng> &&
                hpx::parallel::traits::is_projected_range_v<Proj, Rng> &&
                hpx::parallel::traits::is_indirect_callable_v<
                    ExPolicy, F,
                    hpx::parallel::traits::projected_range<Proj, Rng>,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >
            )>
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy,
            minmax_element_result<hpx::traits::range_iterator_t<Rng>>>
        tag_fallback_invoke(hpx::ranges::minmax_element_t, ExPolicy&& policy,
            Rng&& rng, F&& f = F(), Proj&& proj = Proj())
        {
            static_assert((hpx::traits::is_forward_iterator_v<
                              hpx::traits::range_iterator_t<Rng>>),
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::minmax_element<
                hpx::traits::range_iterator_t<Rng>>()
                .call(HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
                    hpx::util::end(rng), HPX_FORWARD(F, f),
                    HPX_FORWARD(Proj, proj));
        }
    } minmax_element{};
}}    // namespace hpx::ranges

#endif    // DOXYGEN
