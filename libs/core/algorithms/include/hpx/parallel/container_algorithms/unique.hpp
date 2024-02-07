//  Copyright (c) 2017 Taeguk Kwon
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/unique.hpp
/// \page hpx::ranges::unique, hpx::ranges::unique_copy
/// \headerfile hpx/algorithm.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx { namespace ranges {
    // clang-format off

    ///////////////////////////////////////////////////////////////////////////
    /// Eliminates all but the first element from every consecutive group of
    /// equivalent elements from the range [first, last) and returns a
    /// past-the-end iterator for the new logical end of the range.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first - 1 applications of
    ///         the predicate \a pred and no more than twice as many
    ///         applications of the projection \a proj.
    ///
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a unique requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     binary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be
    ///                     such that objects of types \a FwdIter can be
    ///                     dereferenced and then implicitly converted to
    ///                     both \a Type1 and \a Type2
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a unique algorithm invoked without
    /// an execution policy object execute in sequential order in the
    /// calling thread.
    ///
    /// \returns  The \a unique algorithm returns \a subrange_t<FwdIter, Sent>.
    ///           The \a unique algorithm returns an object {ret, last},
    ///           where ret is a past-the-end iterator for a new
    ///           subrange.
    ///
    template <typename FwdIter, typename Sent,
        typename Pred = ranges::equal_to,
        typename Proj = hpx::identity>
    subrange_t<FwdIter, Sent> unique(FwdIter first, Sent last,
        Pred&& pred = Pred(), Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Eliminates all but the first element from every consecutive group of
    /// equivalent elements from the range [first, last) and returns a
    /// past-the-end iterator for the new logical end of the range.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first - 1 applications of
    ///         the predicate \a pred and no more than twice as many
    ///         applications of the projection \a proj.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a unique requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     binary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be
    ///                     such that objects of types \a FwdIter can be
    ///                     dereferenced and then implicitly converted to
    ///                     both \a Type1 and \a Type2
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a unique algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a unique algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a unique algorithm returns \a subrange_t<FwdIter, Sent>.
    ///           The \a unique algorithm returns an object {ret, last},
    ///           where ret is a past-the-end iterator for a new
    ///           subrange.
    ///
    template <typename ExPolicy, typename FwdIter, typename Sent,
        typename Pred = ranges::equal_to,
        typename Proj = hpx::identity>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        subrange_t<FwdIter, Sent>>::type
    unique(ExPolicy&& policy, FwdIter first, Sent last, Pred&& pred = Pred(),
        Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Eliminates all but the first element from every consecutive group of
    /// equivalent elements from the range \a rng and returns a
    /// past-the-end iterator for the new logical end of the range.
    ///
    /// \note   Complexity: Performs not more than N assignments,
    ///         exactly N - 1 applications of the predicate \a pred and
    ///         no more than twice as many applications of the projection
    ///         \a proj, where N = std::distance(begin(rng), end(rng)).
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a unique requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     binary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a unique algorithm invoked without
    /// an execution policy object execute in sequential order in the
    /// calling thread.
    ///
    /// \returns  The \a unique algorithm returns
    ///           \a subrange_t<hpx::traits::range_iterator_t<Rng>,
    ///           hpx::traits::range_iterator_t<Rng>>.
    ///           The \a unique algorithm returns an object {ret, last},
    ///           where ret is a past-the-end iterator for a new
    ///           subrange.
    ///
    template <typename Rng,
        typename Pred = ranges::equal_to,
        typename Proj = hpx::identity>
    subrange_t<hpx::traits::range_iterator_t<Rng>,
        hpx::traits::range_iterator_t<Rng>>
    unique(Rng&& rng, Pred&& pred = Pred(), Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Eliminates all but the first element from every consecutive group of
    /// equivalent elements from the range \a rng and returns a
    /// past-the-end iterator for the new logical end of the range.
    ///
    /// \note   Complexity: Performs not more than N assignments,
    ///         exactly N - 1 applications of the predicate \a pred and
    ///         no more than twice as many applications of the projection
    ///         \a proj, where N = std::distance(begin(rng), end(rng)).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a unique requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     binary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a unique algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a unique algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a unique algorithm returns a \a hpx::future
    ///           <subrange_t<hpx::traits::range_iterator_t<Rng>,
    ///           hpx::traits::range_iterator_t<Rng>>>
    ///           if the execution policy is of type \a sequenced_task_policy
    ///           or \a parallel_task_policy and returns \a
    ///           subrange_t<hpx::traits::range_iterator_t<Rng>,
    ///           hpx::traits::range_iterator_t<Rng>> otherwise.
    ///           The \a unique algorithm returns an object {ret, last},
    ///           where ret is a past-the-end iterator for a new
    ///           subrange.
    ///
    template <typename ExPolicy, typename Rng,
        typename Pred = ranges::equal_to,
        typename Proj = hpx::identity>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        subrange_t<hpx::traits::range_iterator_t<Rng>,
            hpx::traits::range_iterator_t<Rng>>>
    unique(ExPolicy&& policy, Rng&& rng, Pred&& pred = Pred(),
        Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Copies the elements from the range [first, last),
    /// to another range beginning at \a dest in such a way that
    /// there are no consecutive equal elements. Only the first element of
    /// each group of equal elements is copied.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first - 1 applications of
    ///         the predicate \a pred and no more than twice as many
    ///         applications of the projection \a proj
    ///
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    /// \tparam O           The type of the iterator representing the
    ///                     destination range (deduced).
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a unique_copy requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     binary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a InIter1 can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a unique_copy algorithm invoked
    /// without an execution policy object  will execute in sequential
    /// order in the calling thread.
    ///
    /// \returns  The \a unique_copy algorithm returns a
    ///           returns unique_copy_result<InIter, O>.
    ///           The \a unique_copy algorithm returns an in_out_result with
    ///           the source iterator to one past the last element and out
    ///           containing the destination iterator to the end of the
    ///           \a dest range.
    ///
    template <typename InIter, typename Sent, typename O,
        typename Pred = ranges::equal_to,
        typename Proj = hpx::identity>
    unique_copy_result<InIter, O> unique_copy(InIter first,
        Sent last, O dest, Pred&& pred = Pred(), Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Copies the elements from the range [first, last),
    /// to another range beginning at \a dest in such a way that
    /// there are no consecutive equal elements. Only the first element of
    /// each group of equal elements is copied.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first - 1 applications of
    ///         the predicate \a pred and no more than twice as many
    ///         applications of the projection \a proj
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter1.
    /// \tparam O           The type of the iterator representing the
    ///                     destination range (deduced).
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a unique_copy requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     binary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a unique_copy algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a unique_copy algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a unique_copy algorithm returns returns a hpx::future<
    ///           unique_copy_result<FwdIter, O>> if the
    ///           execution policy is of type \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a
    ///           unique_copy_result<FwdIter, O> otherwise.
    ///           The \a unique_copy algorithm returns an in_out_result with
    ///           the source iterator to one past the last element and out
    ///           containing the destination iterator to the end of the
    ///           \a dest range.
    ///
    template <typename ExPolicy, typename FwdIter, typename Sent,
        typename O,
        typename Pred = ranges::equal_to,
        typename Proj = hpx::identity>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        unique_copy_result<FwdIter, O>>::type
    unique_copy(ExPolicy&& policy, FwdIter first, Sent last,
        O dest, Pred&& pred = Pred(), Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Copies the elements from the range \a rng,
    /// to another range beginning at \a dest in such a way that
    /// there are no consecutive equal elements. Only the first element of
    /// each group of equal elements is copied.
    ///
    /// \note   Complexity: Performs not more than N assignments,
    ///         exactly N - 1 applications of the predicate \a pred,
    ///         where N = std::distance(begin(rng), end(rng)).
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam O           The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a unique_copy requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by the range \a rng. This is an
    ///                     binary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a unique_copy algorithm invoked
    /// without an execution policy object  will execute in sequential
    /// order in the calling thread.
    ///
    /// \returns  The \a unique_copy algorithm returns \a
    ///           unique_copy_result<
    ///           hpx::traits::range_iterator_t<Rng>, O>.
    ///           The \a unique_copy algorithm returns the pair of
    ///           the source iterator to \a last, and
    ///           the destination iterator to the end of the \a dest range.
    ///
    template <typename Rng, typename O,
        typename Pred = ranges::equal_to,
        typename Proj = hpx::identity>
    unique_copy_result<hpx::traits::range_iterator_t<Rng>, O>
    unique_copy(Rng&& rng, O dest, Pred&& pred = Pred(), Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Copies the elements from the range \a rng,
    /// to another range beginning at \a dest in such a way that
    /// there are no consecutive equal elements. Only the first element of
    /// each group of equal elements is copied.
    ///
    /// \note   Complexity: Performs not more than N assignments,
    ///         exactly N - 1 applications of the predicate \a pred,
    ///         where N = std::distance(begin(rng), end(rng)).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam O           The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a unique_copy requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by the range \a rng. This is an
    ///                     binary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a unique_copy algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a unique_copy algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a unique_copy algorithm returns a
    ///           \a hpx::future<unique_copy_result<
    ///           hpx::traits::range_iterator_t<Rng>, O>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a unique_copy_result<
    ///           hpx::traits::range_iterator_t<Rng>, O>
    ///           otherwise.
    ///           The \a unique_copy algorithm returns the pair of
    ///           the source iterator to \a last, and
    ///           the destination iterator to the end of the \a dest range.
    ///
    template <typename ExPolicy, typename Rng, typename O,
        typename Pred = ranges::equal_to,
        typename Proj = hpx::identity>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        unique_copy_result<hpx::traits::range_iterator_t<Rng>, O>>
    unique_copy(ExPolicy&& policy, Rng&& rng, O dest,
        Pred&& pred = Pred(), Proj&& proj = Proj());

    // clang-format on
}}    // namespace hpx::ranges

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/iterator_support/iterator_range.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/parallel/algorithms/unique.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <type_traits>
#include <utility>

namespace hpx::parallel {

    // clang-format off
    template <typename ExPolicy, typename Rng,
        typename Pred = detail::equal_to,
        typename Proj = hpx::identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_range_v<Rng> &&
            traits::is_projected_range_v<Proj, Rng> &&
            traits::is_indirect_callable_v<ExPolicy, Pred,
                traits::projected_range<Proj, Rng>,
                traits::projected_range<Proj, Rng>
            >
        )>
    // clang-format on
    HPX_DEPRECATED_V(
        1, 8, "hpx::parallel::unique is deprecated, use hpx::unique instead")
        util::detail::algorithm_result_t<ExPolicy,
            hpx::traits::range_iterator_t<Rng>> unique(ExPolicy&& policy,
            Rng&& rng, Pred&& pred = Pred(), Proj&& proj = Proj())
    {
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return unique(HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
            hpx::util::end(rng), HPX_FORWARD(Pred, pred),
            HPX_FORWARD(Proj, proj));
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }

    // clang-format off
    template <typename ExPolicy, typename Rng, typename FwdIter2,
        typename Pred = detail::equal_to,
        typename Proj = hpx::identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_range_v<Rng> &&
            hpx::traits::is_iterator_v<FwdIter2> &&
            traits::is_projected_range_v<Proj, Rng> &&
            traits::is_indirect_callable<ExPolicy, Pred,
                traits::projected_range<Proj, Rng>,
                traits::projected_range<Proj, Rng>>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::unique_copy is deprecated, use hpx::unique_copy "
        "instead") util::detail::algorithm_result_t<ExPolicy,
        util::in_out_result<hpx::traits::range_iterator_t<Rng>,
            FwdIter2>> unique_copy(ExPolicy&& policy, Rng&& rng, FwdIter2 dest,
        Pred&& pred = Pred(), Proj&& proj = Proj())
    {
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return unique_copy(HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
            hpx::util::end(rng), dest, HPX_FORWARD(Pred, pred),
            HPX_FORWARD(Proj, proj));
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }
}    // namespace hpx::parallel

namespace hpx::ranges {

    ///////////////////////////////////////////////////////////////////////////
    template <typename I, typename S>
    using subrange_t = hpx::util::iterator_range<I, S>;

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::unique
    inline constexpr struct unique_t final
      : hpx::detail::tag_parallel_algorithm<unique_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Sent,
            typename Pred = ranges::equal_to,
            typename Proj = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<FwdIter> &&
                hpx::traits::is_sentinel_for_v<Sent, FwdIter> &&
                parallel::traits::is_projected_v<Proj, FwdIter> &&
                parallel::traits::is_indirect_callable_v<
                    hpx::execution::sequenced_policy, Pred,
                    parallel::traits::projected<Proj, FwdIter>,
                    parallel::traits::projected<Proj, FwdIter>
                >
            )>
        // clang-format on
        friend subrange_t<FwdIter, Sent> tag_fallback_invoke(
            hpx::ranges::unique_t, FwdIter first, Sent last, Pred pred = Pred(),
            Proj proj = Proj())
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return hpx::parallel::util::make_subrange<FwdIter, Sent>(
                hpx::parallel::detail::unique<FwdIter>().call(
                    hpx::execution::seq, first, last, HPX_MOVE(pred),
                    HPX_MOVE(proj)),
                last);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename Pred = ranges::equal_to,
            typename Proj = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<FwdIter> &&
                hpx::traits::is_sentinel_for_v<Sent, FwdIter> &&
                parallel::traits::is_projected_v<Proj, FwdIter> &&
                parallel::traits::is_indirect_callable_v<
                    ExPolicy, Pred,
                    parallel::traits::projected<Proj, FwdIter>,
                    parallel::traits::projected<Proj, FwdIter>
                >
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            subrange_t<FwdIter, Sent>>::type
        tag_fallback_invoke(hpx::ranges::unique_t, ExPolicy&& policy,
            FwdIter first, Sent last, Pred pred = Pred(), Proj proj = Proj())
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return hpx::parallel::util::make_subrange<FwdIter, Sent>(
                hpx::parallel::detail::unique<FwdIter>().call(
                    HPX_FORWARD(ExPolicy, policy), first, last, HPX_MOVE(pred),
                    HPX_MOVE(proj)),
                last);
        }

        // clang-format off
        template <typename Rng,
            typename Pred = ranges::equal_to,
            typename Proj = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range_v<Rng> &&
                hpx::parallel::traits::is_projected_range_v<Proj, Rng> &&
                hpx::parallel::traits::is_indirect_callable_v<
                    hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected_range<Proj, Rng>,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >
            )>
        // clang-format on
        friend subrange_t<hpx::traits::range_iterator_t<Rng>,
            hpx::traits::range_iterator_t<Rng>>
        tag_fallback_invoke(hpx::ranges::unique_t, Rng&& rng,
            Pred pred = Pred(), Proj proj = Proj())
        {
            using iterator_type = hpx::traits::range_iterator_t<Rng>;

            static_assert(hpx::traits::is_forward_iterator_v<iterator_type>,
                "Requires at least input iterator.");

            return hpx::parallel::util::make_subrange<
                hpx::traits::range_iterator_t<Rng>,
                typename hpx::traits::range_sentinel<Rng>::type>(
                hpx::parallel::detail::unique<
                    hpx::traits::range_iterator_t<Rng>>()
                    .call(hpx::execution::seq, hpx::util::begin(rng),
                        hpx::util::end(rng), HPX_MOVE(pred), HPX_MOVE(proj)),
                hpx::util::end(rng));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,
            typename Pred = ranges::equal_to,
            typename Proj = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_range_v<Rng> &&
                hpx::parallel::traits::is_projected_range_v<Proj, Rng> &&
                hpx::parallel::traits::is_indirect_callable_v<
                    ExPolicy, Pred,
                    hpx::parallel::traits::projected_range<Proj, Rng>,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy,
            subrange_t<hpx::traits::range_iterator_t<Rng>,
                hpx::traits::range_iterator_t<Rng>>>
        tag_fallback_invoke(hpx::ranges::unique_t, ExPolicy&& policy, Rng&& rng,
            Pred pred = Pred(), Proj proj = Proj())
        {
            using iterator_type = hpx::traits::range_iterator_t<Rng>;

            static_assert(hpx::traits::is_forward_iterator_v<iterator_type>,
                "Requires at least forward iterator.");

            return hpx::parallel::util::make_subrange<
                hpx::traits::range_iterator_t<Rng>,
                typename hpx::traits::range_sentinel<Rng>::type>(
                hpx::parallel::detail::unique<
                    hpx::traits::range_iterator_t<Rng>>()
                    .call(HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
                        hpx::util::end(rng), HPX_MOVE(pred), HPX_MOVE(proj)),
                hpx::util::end(rng));
        }
    } unique{};

    template <typename I, typename O>
    using unique_copy_result = parallel::util::in_out_result<I, O>;

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::unique_copy
    inline constexpr struct unique_copy_t final
      : hpx::detail::tag_parallel_algorithm<unique_copy_t>
    {
    private:
        // clang-format off
        template <typename InIter, typename Sent, typename O,
            typename Pred = ranges::equal_to,
            typename Proj = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<InIter> &&
                hpx::traits::is_sentinel_for_v<Sent, InIter> &&
                parallel::traits::is_projected_v<Proj, InIter> &&
                parallel::traits::is_indirect_callable_v<
                    hpx::execution::sequenced_policy, Pred,
                    parallel::traits::projected<Proj, InIter>,
                    parallel::traits::projected<Proj, InIter>
                >
            )>
        // clang-format on
        friend unique_copy_result<InIter, O> tag_fallback_invoke(
            hpx::ranges::unique_copy_t, InIter first, Sent last, O dest,
            Pred pred = Pred(), Proj proj = Proj())
        {
            static_assert(hpx::traits::is_input_iterator_v<InIter>,
                "Requires at least input iterator.");

            using result_type = unique_copy_result<InIter, O>;

            return hpx::parallel::detail::unique_copy<result_type>().call(
                hpx::execution::seq, first, last, dest, HPX_MOVE(pred),
                HPX_MOVE(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename O,
            typename Pred = ranges::equal_to,
            typename Proj = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<FwdIter> &&
                hpx::traits::is_sentinel_for_v<Sent, FwdIter> &&
                parallel::traits::is_projected_v<Proj, FwdIter> &&
                parallel::traits::is_indirect_callable_v<
                    ExPolicy, Pred,
                    parallel::traits::projected<Proj, FwdIter>,
                    parallel::traits::projected<Proj, FwdIter>
                >
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy,
            unique_copy_result<FwdIter, O>>
        tag_fallback_invoke(hpx::ranges::unique_copy_t, ExPolicy&& policy,
            FwdIter first, Sent last, O dest, Pred pred = Pred(),
            Proj proj = Proj())
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            using result_type = unique_copy_result<FwdIter, O>;

            return hpx::parallel::detail::unique_copy<result_type>().call(
                HPX_FORWARD(ExPolicy, policy), first, last, dest,
                HPX_MOVE(pred), HPX_MOVE(proj));
        }

        // clang-format off
        template <typename Rng, typename O,
            typename Pred = ranges::equal_to,
            typename Proj = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range_v<Rng> &&
                hpx::parallel::traits::is_projected_range_v<Proj, Rng> &&
                hpx::parallel::traits::is_indirect_callable_v<
                    hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected_range<Proj, Rng>,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >
            )>
        // clang-format on
        friend unique_copy_result<hpx::traits::range_iterator_t<Rng>, O>
        tag_fallback_invoke(hpx::ranges::unique_copy_t, Rng&& rng, O dest,
            Pred pred = Pred(), Proj proj = Proj())
        {
            using iterator_type = hpx::traits::range_iterator_t<Rng>;

            static_assert(hpx::traits::is_input_iterator_v<iterator_type>,
                "Requires at least input iterator.");

            using result_type = unique_copy_result<iterator_type, O>;

            return hpx::parallel::detail::unique_copy<result_type>().call(
                hpx::execution::seq, hpx::util::begin(rng), hpx::util::end(rng),
                dest, HPX_MOVE(pred), HPX_MOVE(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename O,
            typename Pred = ranges::equal_to,
            typename Proj = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_range_v<Rng> &&
                hpx::parallel::traits::is_projected_range_v<Proj, Rng> &&
                hpx::parallel::traits::is_indirect_callable_v<
                    ExPolicy, Pred,
                    hpx::parallel::traits::projected_range<Proj, Rng>,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy,
            unique_copy_result<hpx::traits::range_iterator_t<Rng>, O>>
        tag_fallback_invoke(hpx::ranges::unique_copy_t, ExPolicy&& policy,
            Rng&& rng, O dest, Pred pred = Pred(), Proj proj = Proj())
        {
            using iterator_type = hpx::traits::range_iterator_t<Rng>;

            static_assert(hpx::traits::is_forward_iterator_v<iterator_type>,
                "Requires at least input iterator.");

            using result_type = unique_copy_result<iterator_type, O>;

            return hpx::parallel::detail::unique_copy<result_type>().call(
                HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
                hpx::util::end(rng), dest, HPX_MOVE(pred), HPX_MOVE(proj));
        }
    } unique_copy{};
}    // namespace hpx::ranges

#endif    // DOXYGEN
