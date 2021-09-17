//  Copyright (c) 2017 Taeguk Kwon
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/partition.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    // clang-format off

    ///////////////////////////////////////////////////////////////////////////
    /// Reorders the elements in the range [first, last) in such a way that
    /// all elements for which the predicate \a pred returns true precede
    /// the elements for which the predicate \a pred returns false.
    /// Relative order of the elements is not preserved.
    ///
    /// \note   Complexity: At most 2 * (last - first) swaps.
    ///         Exactly \a last - \a first applications of the predicate and
    ///         projection.
    ///
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a partition requires \a Pred to meet
    ///                     the requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     unary predicate for partitioning the source
    ///                     iterators. The signature of
    ///                     this predicate should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a partition algorithm invoked without
    /// an execution policy object execute in sequential order in the calling
    /// thread.
    ///
    /// \returns  The \a partition algorithm returns returns \a
    ///           subrange_t<FwdIter>.
    ///           The \a partition algorithm returns a subrange starting with
    ///           an iterator to the first element of the second group and
    ///           finishing with an iterator equal to last.
    ///
    template <typename FwdIter, typename Sent, typename Pred, typename Proj>
    subrange_t<FwdIter> partition(ExPolicy&& policy, Sent first, Sent last,
        Pred&& pred, Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Reorders the elements in the range [first, last) in such a way that
    /// all elements for which the predicate \a pred returns true precede
    /// the elements for which the predicate \a pred returns false.
    /// Relative order of the elements is not preserved.
    ///
    /// \note   Complexity: At most 2 * (last - first) swaps.
    ///         Exactly \a last - \a first applications of the predicate and
    ///         projection.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a partition requires \a Pred to meet
    ///                     the requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
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
    ///                     unary predicate for partitioning the source
    ///                     iterators. The signature of
    ///                     this predicate should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a partition algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a partition algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a partition algorithm returns a \a
    ///           hpx::future<subrange_t<FwdIter>>
    ///           if the execution policy is of type \a parallel_task_policy
    ///           and returns \a subrange_t<FwdIter> otherwise.
    ///           The \a partition algorithm returns a subrange starting with
    ///           an iterator to the first element of the second group and
    ///           finishing with an iterator equal to last.
    ///
    template <typename ExPolicy, typename FwdIter, typename Sent,
        typename Pred, typename Proj>
    typename util::detail::algorithm_result<ExPolicy,
        subrange_t<FwdIter>>::type
    partition(ExPolicy&& policy, FwdIter first, Sent last, Pred&& pred,
        Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Reorders the elements in the range \a rng in such a way that
    /// all elements for which the predicate \a pred returns true precede
    /// the elements for which the predicate \a pred returns false.
    /// Relative order of the elements is not preserved.
    ///
    /// \note   Complexity: Performs at most 2 * N swaps,
    ///         exactly N applications of the predicate and projection,
    ///         where N = std::distance(begin(rng), end(rng)).
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a partition requires \a Pred to meet
    ///                     the requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by the range \a rng. This is an
    ///                     unary predicate for partitioning the source
    ///                     iterators. The signature of this predicate should
    ///                     be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
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
    /// The assignments in the parallel \a partition algorithm invoked without
    /// an execution policy object execute in sequential order in the calling
    /// thread.
    ///
    /// \returns  The \a partition algorithm returns
    ///           \a subrange_t<hpx::traits::range_iterator_t<Rng>>
    ///           The \a partition algorithm returns a subrange starting with
    ///           an iterator to the first element of the second group and
    ///           finishing with an iterator equal to last.
    ///
    template <typename Rng, typename Pred, typename Proj>
    subrange_t<hpx::traits::range_iterator_t<Rng>>
    partition(Rng&& rng, Pred&& pred, Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Reorders the elements in the range \a rng in such a way that
    /// all elements for which the predicate \a pred returns true precede
    /// the elements for which the predicate \a pred returns false.
    /// Relative order of the elements is not preserved.
    ///
    /// \note   Complexity: Performs at most 2 * N swaps,
    ///         exactly N applications of the predicate and projection,
    ///         where N = std::distance(begin(rng), end(rng)).
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
    ///                     overload of \a partition requires \a Pred to meet
    ///                     the requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by the range \a rng. This is an
    ///                     unary predicate for partitioning the source
    ///                     iterators. The signature of this predicate should
    ///                     be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
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
    /// The assignments in the parallel \a partition algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a partition algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a partition algorithm returns a \a
    ///           hpx::future<subrange_t<hpx::traits::range_iterator_t<Rng>>>
    ///           if the execution policy is of type \a parallel_task_policy
    ///           and returns \a subrange_t<hpx::traits::range_iterator_t<Rng>>
    ///           The \a partition algorithm returns a subrange starting with
    ///           an iterator to the first element of the second group and
    ///           finishing with an iterator equal to last.
    ///
    template <typename ExPolicy, typename Rng, typename Pred,
        typename Proj>
    typename util::detail::algorithm_result<ExPolicy,
        subrange_t<hpx::traits::range_iterator_t<Rng>>>::type
    partition(ExPolicy&& policy, Rng&& rng, Pred&& pred, Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Permutes the elements in the range [first, last) such that there exists
    /// an iterator i such that for every iterator j in the range [first, i)
    /// INVOKE(f, INVOKE (proj, *j)) != false, and for every iterator k in the
    /// range [i, last), INVOKE(f, INVOKE (proj, *k)) == false
    ///
    /// \note   Complexity: At most (last - first) * log(last - first) swaps,
    ///         but only linear number of swaps if there is enough extra memory
    ///         Exactly \a last - \a first applications of the predicate and
    ///         projection.
    ///
    /// \tparam BidirIter   The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for BidirIter.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a transform requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param f            Unary predicate which returns true if the element
    ///                     should be ordered before other elements.
    ///                     Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). The signature
    ///                     of this predicate should be equivalent to:
    ///                     \code
    ///                     bool fun(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&.
    ///                     The type \a Type must be such that an object of
    ///                     type \a BidirIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a f is invoked.
    ///
    /// The invocations of \a f in the parallel \a stable_partition algorithm
    /// invoked without an execution policy object executes in sequential order
    /// in the calling thread.
    ///
    /// \returns  The \a stable_partition algorithm returns an iterator i such
    ///           that for every iterator j in the range [first, i), f(*j) !=
    ///           false INVOKE(f, INVOKE(proj, *j)) != false, and for every
    ///           iterator k in the range [i, last), f(*k) == false
    ///           INVOKE(f, INVOKE (proj, *k)) == false. The relative order of
    ///           the elements in both groups is preserved.
    ///
    template <typename BidirIter, typename Sent, typename F, typename Proj>
    subrange_t<BidirIter> stable_partition(BidirIter first, Sent last, F&& f,
        Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Permutes the elements in the range [first, last) such that there exists
    /// an iterator i such that for every iterator j in the range [first, i)
    /// INVOKE(f, INVOKE (proj, *j)) != false, and for every iterator k in the
    /// range [i, last), INVOKE(f, INVOKE (proj, *k)) == false
    ///
    /// \note   Complexity: At most (last - first) * log(last - first) swaps,
    ///         but only linear number of swaps if there is enough extra memory
    ///         Exactly \a last - \a first applications of the predicate and
    ///         projection.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the invocations of \a f.
    /// \tparam BidirIter   The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for BidirIter.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a transform requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param f            Unary predicate which returns true if the element
    ///                     should be ordered before other elements.
    ///                     Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). The signature
    ///                     of this predicate should be equivalent to:
    ///                     \code
    ///                     bool fun(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&.
    ///                     The type \a Type must be such that an object of
    ///                     type \a BidirIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a f is invoked.
    ///
    /// The invocations of \a f in the parallel \a stable_partition algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy executes in sequential order in the
    /// calling thread.
    ///
    /// The invocations of \a f in the parallel \a stable_partition algorithm
    /// invoked with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a stable_partition algorithm returns an iterator i such
    ///           that for every iterator j in the range [first, i), f(*j) !=
    ///           false INVOKE(f, INVOKE(proj, *j)) != false, and for every
    ///           iterator k in the range [i, last), f(*k) == false
    ///           INVOKE(f, INVOKE (proj, *k)) == false. The relative order of
    ///           the elements in both groups is preserved.
    ///           If the execution policy is of type \a parallel_task_policy
    ///           the algorithm returns a future<> referring to this iterator.
    ///
    template <typename ExPolicy, typename BidirIter, typename Sent, typename F,
        typename Proj>
    typename util::detail::algorithm_result<ExPolicy,
        subrange_t<BidirIter>>::type
    stable_partition(ExPolicy&& policy, BidirIter first, Sent last,
        F&& f, Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Permutes the elements in the range [first, last) such that there exists
    /// an iterator i such that for every iterator j in the range [first, i)
    /// INVOKE(f, INVOKE (proj, *j)) != false, and for every iterator k in the
    /// range [i, last), INVOKE(f, INVOKE (proj, *k)) == false
    ///
    /// \note   Complexity: At most (last - first) * log(last - first) swaps,
    ///         but only linear number of swaps if there is enough extra memory
    ///         Exactly \a last - \a first applications of the predicate and
    ///         projection.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an birdirectional iterator
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a transform requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param f            Unary predicate which returns true if the element
    ///                     should be ordered before other elements.
    ///                     Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). The signature
    ///                     of this predicate should be equivalent to:
    ///                     \code
    ///                     bool fun(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&.
    ///                     The type \a Type must be such that an object of
    ///                     type \a BidirIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a f is invoked.
    ///
    /// The invocations of \a f in the parallel \a stable_partition algorithm
    /// invoked without an execution policy object executes in sequential order
    /// in the calling thread.
    ///
    /// \returns  The \a stable_partition algorithm returns an iterator i such
    ///           that for every iterator j in the range [first, i), f(*j) !=
    ///           false INVOKE(f, INVOKE(proj, *j)) != false, and for every
    ///           iterator k in the range [i, last), f(*k) == false
    ///           INVOKE(f, INVOKE (proj, *k)) == false. The relative order of
    ///           the elements in both groups is preserved.
    ///
    template <typename Rng, typename F, typename Proj>
    subrange_t<hpx::traits::range_iterator_t<Rng>> stable_partition(Rng&& rng,
        F&& f, Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Permutes the elements in the range [first, last) such that there exists
    /// an iterator i such that for every iterator j in the range [first, i)
    /// INVOKE(f, INVOKE (proj, *j)) != false, and for every iterator k in the
    /// range [i, last), INVOKE(f, INVOKE (proj, *k)) == false
    ///
    /// \note   Complexity: At most (last - first) * log(last - first) swaps,
    ///         but only linear number of swaps if there is enough extra memory.
    ///         Exactly \a last - \a first applications of the predicate and
    ///         projection.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the invocations of \a f.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an birdirectional iterator
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a transform requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param f            Unary predicate which returns true if the element
    ///                     should be ordered before other elements.
    ///                     Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). The signature
    ///                     of this predicate should be equivalent to:
    ///                     \code
    ///                     bool fun(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&.
    ///                     The type \a Type must be such that an object of
    ///                     type \a BidirIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a f is invoked.
    ///
    /// The invocations of \a f in the parallel \a stable_partition algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy executes in sequential order in the
    /// calling thread.
    ///
    /// The invocations of \a f in the parallel \a stable_partition algorithm
    /// invoked with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a stable_partition algorithm returns an iterator i such
    ///           that for every iterator j in the range [first, i), f(*j) !=
    ///           false INVOKE(f, INVOKE(proj, *j)) != false, and for every
    ///           iterator k in the range [i, last), f(*k) == false
    ///           INVOKE(f, INVOKE (proj, *k)) == false. The relative order of
    ///           the elements in both groups is preserved.
    ///           If the execution policy is of type \a parallel_task_policy
    ///           the algorithm returns a future<> referring to this iterator.
    ///
    template <typename ExPolicy, typename Rng, typename F, typename Proj>
    typename util::detail::algorithm_result<ExPolicy,
        subrange_t<hpx::traits::range_iterator_t<Rng>>>::type
    stable_partition(ExPolicy&& policy, Rng&& rng, F&& f, Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Copies the elements in the range, defined by [first, last),
    /// to two different ranges depending on the value returned by
    /// the predicate \a pred. The elements, that satisfy the predicate \a pred
    /// are copied to the range beginning at \a dest_true. The rest of
    /// the elements are copied to the range beginning at \a dest_false.
    /// The order of the elements is preserved.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first applications of the
    ///         predicate \a f.
    ///
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range for the elements that satisfy
    ///                     the predicate \a pred (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter3    The type of the iterator representing the
    ///                     destination range for the elements that don't
    ///                     satisfy the predicate \a pred (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a partition_copy requires \a Pred to
    ///                     meet the requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param dest_true    Refers to the beginning of the destination range
    ///                     for the elements that satisfy the predicate \a pred
    /// \param dest_false   Refers to the beginning of the destination range
    ///                     for the elements that don't satisfy the predicate
    ///                     \a pred.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     unary predicate for partitioning the source
    ///                     iterators. The signature of
    ///                     this predicate should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a partition_copy algorithm invoked
    /// without an execution policy object execute in sequential order in the
    /// calling thread.
    ///
    /// \returns  The \a partition_copy algorithm returns a
    ///           \a partition_copy_result<FwdIter, OutIter2, OutIter3>.
    ///           The \a partition_copy algorithm returns the tuple of
    ///           the source iterator \a last,
    ///           the destination iterator to the end of the \a
    ///           dest_true range, and the destination iterator to the end of
    ///           the \a dest_false range.
    ///
    template <typename FwdIter1, typename Sent, typename FwdIter2,
        typename FwdIter3, typename Pred, typename Proj>
    hpx::util::tagged_tuple<tag::in(FwdIter1), tag::out1(FwdIter2),
            tag::out2(FwdIter3)>
    partition_copy(FwdIter1 first, Sent last,
        FwdIter2 dest_true, FwdIter3 dest_false, Pred&& pred, Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Copies the elements in the range, defined by [first, last),
    /// to two different ranges depending on the value returned by
    /// the predicate \a pred. The elements, that satisfy the predicate \a pred
    /// are copied to the range beginning at \a dest_true. The rest of
    /// the elements are copied to the range beginning at \a dest_false.
    /// The order of the elements is preserved.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first applications of the
    ///         predicate \a f.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range for the elements that satisfy
    ///                     the predicate \a pred (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter3    The type of the iterator representing the
    ///                     destination range for the elements that don't
    ///                     satisfy the predicate \a pred (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a partition_copy requires \a Pred to
    ///                     meet the requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param dest_true    Refers to the beginning of the destination range
    ///                     for the elements that satisfy the predicate \a pred
    /// \param dest_false   Refers to the beginning of the destination range
    ///                     for the elements that don't satisfy the predicate
    ///                     \a pred.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     unary predicate for partitioning the source
    ///                     iterators. The signature of
    ///                     this predicate should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a partition_copy algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a partition_copy algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a partition_copy algorithm returns a
    ///           hpx::future<partition_copy_result<FwdIter, OutIter2,
    ///           OutIter3>>
    ///           if the execution policy is of type \a parallel_task_policy
    ///           and returns
    ///           \a partition_copy_result<FwdIter, OutIter2, OutIter3>
    ///           otherwise.
    ///           The \a partition_copy algorithm returns the tuple of
    ///           the source iterator \a last,
    ///           the destination iterator to the end of the \a
    ///           dest_true range, and the destination iterator to the end of
    ///           the \a dest_false range.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Sent,
        typename FwdIter2, typename FwdIter3, typename Pred, typename Proj>
    typename util::detail::algorithm_result<ExPolicy,
        partition_copy_result<FwdIter, OutIter2, OutIter3>>::type
    partition_copy(ExPolicy&& policy, FwdIter1 first, Sent last,
        FwdIter2 dest_true, FwdIter3 dest_false, Pred&& pred, Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Copies the elements in the range \a rng,
    /// to two different ranges depending on the value returned by
    /// the predicate \a pred. The elements, that satisfy the predicate \a pred
    /// are copied to the range beginning at \a dest_true. The rest of
    /// the elements are copied to the range beginning at \a dest_false.
    /// The order of the elements is preserved.
    ///
    /// \note   Complexity: Performs not more than N assignments,
    ///         exactly N applications of the predicate \a pred,
    ///         where N = std::distance(begin(rng), end(rng)).
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range for the elements that satisfy
    ///                     the predicate \a pred (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter3    The type of the iterator representing the
    ///                     destination range for the elements that don't
    ///                     satisfy the predicate \a pred (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a partition_copy requires \a Pred to
    ///                     meet the requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest_true    Refers to the beginning of the destination range
    ///                     for the elements that satisfy the predicate \a pred
    /// \param dest_false   Refers to the beginning of the destination range
    ///                     for the elements that don't satisfy the predicate
    ///                     \a pred.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     unary predicate for partitioning the source
    ///                     iterators. The signature of
    ///                     this predicate should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a partition_copy algorithm invoked
    /// without an execution policy object execute in sequential order in the
    /// calling thread.
    ///
    /// \returns  The \a partition_copy algorithm returns a
    ///           partition_copy_result<hpx::traits::range_iterator_t<Rng>,
    ///           FwdIter2, FwdIter3>>.
    ///           The \a partition_copy algorithm returns the tuple of
    ///           the source iterator \a last,
    ///           the destination iterator to the end of the \a dest_true
    ///           range, and the destination iterator to the end of the \a
    ///           dest_false range.
    ///
    template <typename Rng, typename FwdIter2,
        typename FwdIter3, typename Pred, typename Proj>
    friend partition_copy_result<hpx::traits::range_iterator_t<Rng>, FwdIter2,
        FwdIter3> >::type
    partition_copy(Rng&& rng, FwdIter2 dest_true, FwdIter3 dest_false,
        Pred&& pred, Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Copies the elements in the range \a rng,
    /// to two different ranges depending on the value returned by
    /// the predicate \a pred. The elements, that satisfy the predicate \a pred
    /// are copied to the range beginning at \a dest_true. The rest of
    /// the elements are copied to the range beginning at \a dest_false.
    /// The order of the elements is preserved.
    ///
    /// \note   Complexity: Performs not more than N assignments,
    ///         exactly N applications of the predicate \a pred,
    ///         where N = std::distance(begin(rng), end(rng)).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range for the elements that satisfy
    ///                     the predicate \a pred (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter3    The type of the iterator representing the
    ///                     destination range for the elements that don't
    ///                     satisfy the predicate \a pred (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a partition_copy requires \a Pred to
    ///                     meet the requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest_true    Refers to the beginning of the destination range
    ///                     for the elements that satisfy the predicate \a pred
    /// \param dest_false   Refers to the beginning of the destination range
    ///                     for the elements that don't satisfy the predicate
    ///                     \a pred.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     unary predicate for partitioning the source
    ///                     iterators. The signature of
    ///                     this predicate should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a partition_copy algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a partition_copy algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a partition_copy algorithm returns a
    ///           \a hpx::future<partition_copy_result
    ///           <hpx::traits::range_iterator_t<Rng>,
    ///           FwdIter2, FwdIter3>>
    ///           if the execution policy is of type \a parallel_task_policy
    ///           and returns
    ///           partition_copy_result<hpx::traits::range_iterator_t<Rng>,
    ///           FwdIter2, FwdIter3>
    ///           otherwise.
    ///           The \a partition_copy algorithm returns the tuple of
    ///           the source iterator \a last,
    ///           the destination iterator to the end of the \a dest_true
    ///           range, and the destination iterator to the end of the \a
    ///           dest_false range.
    ///
    template <typename ExPolicy, typename Rng, typename FwdIter2,
        typename FwdIter3, typename Pred, typename Proj>
    friend typename parallel::util::detail::algorithm_result<ExPolicy,
        partition_copy_result<hpx::traits::range_iterator_t<Rng>, FwdIter2,
        FwdIter3>>::type
    partition_copy(ExPolicy&& policy, Rng&& rng, FwdIter2 dest_true,
        FwdIter3 dest_false, Pred&& pred, Proj&& proj);

    // clang-format on
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/parallel/util/result_types.hpp>
#include <hpx/parallel/util/tagged_tuple.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/parallel/algorithms/partition.hpp>
#include <hpx/parallel/tagspec.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {
    // clang-format off
    template <typename ExPolicy, typename Rng, typename Pred,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_range<Rng>::value &&
            traits::is_projected_range<Proj, Rng>::value &&
            traits::is_indirect_callable<ExPolicy, Pred,
                traits::projected_range<Proj, Rng>>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::partition is deprecated, use hpx::partition instead")
        typename util::detail::algorithm_result<ExPolicy,
            typename hpx::traits::range_iterator<Rng>::type>::type
        partition(
            ExPolicy&& policy, Rng&& rng, Pred&& pred, Proj&& proj = Proj())
    {
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return partition(std::forward<ExPolicy>(policy), hpx::util::begin(rng),
            hpx::util::end(rng), std::forward<Pred>(pred),
            std::forward<Proj>(proj));
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }

    // clang-format off
    template <typename ExPolicy, typename Rng, typename FwdIter2,
        typename FwdIter3, typename Pred,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_range<Rng>::value &&
            hpx::traits::is_iterator_v<FwdIter2> &&
            hpx::traits::is_iterator_v<FwdIter3> &&
            traits::is_projected_range<Proj, Rng>::value &&
            traits::is_indirect_callable<ExPolicy, Pred,
                traits::projected_range<Proj, Rng>>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::partition_copy is deprecated, use hpx::partition_copy "
        "instead") typename util::detail::algorithm_result<ExPolicy,
        hpx::util::tagged_tuple<tag::in(hpx::traits::range_iterator_t<Rng>),
            tag::out1(FwdIter2), tag::out2(FwdIter3)>>::type
        partition_copy(ExPolicy&& policy, Rng&& rng, FwdIter2 dest_true,
            FwdIter3 dest_false, Pred&& pred, Proj&& proj = Proj())
    {
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return partition_copy(std::forward<ExPolicy>(policy),
            hpx::util::begin(rng), hpx::util::end(rng), dest_true, dest_false,
            std::forward<Pred>(pred), std::forward<Proj>(proj));
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }
}}}    // namespace hpx::parallel::v1

namespace hpx { namespace ranges {

    template <typename I, typename O1, typename O2>
    using partition_copy_result = parallel::util::in_out_out_result<I, O1, O2>;

    HPX_INLINE_CONSTEXPR_VARIABLE struct partition_t final
      : hpx::detail::tag_parallel_algorithm<partition_t>
    {
    private:
        // clang-format off
        template <typename Rng, typename Pred,
            typename Proj = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                parallel::traits::is_projected_range_v<Proj, Rng> &&
                parallel::traits::is_indirect_callable_v<
                    hpx::execution::sequenced_policy, Pred,
                    parallel::traits::projected_range<Proj, Rng>>
        )>
        // clang-format on
        friend subrange_t<hpx::traits::range_iterator_t<Rng>>
        tag_fallback_dispatch(hpx::ranges::partition_t, Rng&& rng, Pred&& pred,
            Proj&& proj = Proj())
        {
            using iterator = hpx::traits::range_iterator_t<Rng>;

            static_assert(hpx::traits::is_forward_iterator_v<iterator>,
                "Requires at least forward iterator.");

            return hpx::parallel::util::make_subrange<
                typename hpx::traits::range_iterator<Rng>::type,
                typename hpx::traits::range_sentinel<Rng>::type>(
                hpx::parallel::v1::detail::partition<iterator>().call(
                    hpx::execution::seq, hpx::util::begin(rng),
                    hpx::util::end(rng), std::forward<Pred>(pred),
                    std::forward<Proj>(proj)),
                hpx::util::end(rng));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename Pred,
            typename Proj = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_range_v<Rng> &&
                parallel::traits::is_projected_range_v<Proj, Rng> &&
                parallel::traits::is_indirect_callable_v<ExPolicy, Pred,
                    parallel::traits::projected_range<Proj, Rng>>
        )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            subrange_t<hpx::traits::range_iterator_t<Rng>>>::type
        tag_fallback_dispatch(hpx::ranges::partition_t, ExPolicy&& policy,
            Rng&& rng, Pred&& pred, Proj&& proj = Proj())
        {
            using iterator = hpx::traits::range_iterator_t<Rng>;

            static_assert(hpx::traits::is_forward_iterator_v<iterator>,
                "Requires at least forward iterator.");

            return hpx::parallel::util::make_subrange<
                typename hpx::traits::range_iterator<Rng>::type,
                typename hpx::traits::range_sentinel<Rng>::type>(
                hpx::parallel::v1::detail::partition<iterator>().call(
                    std::forward<ExPolicy>(policy), hpx::util::begin(rng),
                    hpx::util::end(rng), std::forward<Pred>(pred),
                    std::forward<Proj>(proj)),
                hpx::util::end(rng));
        }

        // clang-format off
        template <typename FwdIter, typename Sent, typename Pred,
            typename Proj = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<FwdIter> &&
                hpx::traits::is_sentinel_for_v<Sent, FwdIter> &&
                parallel::traits::is_indirect_callable_v<
                    hpx::execution::sequenced_policy, Pred,
                    parallel::traits::projected<Proj, FwdIter>>
        )>
        // clang-format on
        friend subrange_t<FwdIter> tag_fallback_dispatch(
            hpx::ranges::partition_t, FwdIter first, Sent last, Pred&& pred,
            Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return hpx::parallel::util::make_subrange<FwdIter, FwdIter>(
                hpx::parallel::v1::detail::partition<FwdIter>().call(
                    hpx::execution::seq, first, last, std::forward<Pred>(pred),
                    std::forward<Proj>(proj)),
                parallel::v1::detail::advance_to_sentinel(first, last));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename Pred,
            typename Proj = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<FwdIter> &&
                hpx::traits::is_sentinel_for_v<Sent, FwdIter> &&
                parallel::traits::is_indirect_callable_v<ExPolicy, Pred,
                    parallel::traits::projected<Proj, FwdIter>>
        )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            subrange_t<FwdIter>>::type
        tag_fallback_dispatch(hpx::ranges::partition_t, ExPolicy&& policy,
            FwdIter first, Sent last, Pred&& pred, Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return hpx::parallel::util::make_subrange<FwdIter, FwdIter>(
                hpx::parallel::v1::detail::partition<FwdIter>().call(
                    std::forward<ExPolicy>(policy), first, last,
                    std::forward<Pred>(pred), std::forward<Proj>(proj)),
                parallel::v1::detail::advance_to_sentinel(first, last));
        }
    } partition{};

    HPX_INLINE_CONSTEXPR_VARIABLE struct stable_partition_t final
      : hpx::detail::tag_parallel_algorithm<stable_partition_t>
    {
    private:
        // clang-format off
        template <typename Rng, typename Pred,
            typename Proj = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                parallel::traits::is_projected_range_v<Proj, Rng> &&
                parallel::traits::is_indirect_callable_v<
                    hpx::execution::sequenced_policy, Pred,
                    parallel::traits::projected_range<Proj, Rng>>
            )>
        // clang-format on
        friend subrange_t<hpx::traits::range_iterator_t<Rng>>
        tag_fallback_dispatch(hpx::ranges::stable_partition_t, Rng&& rng,
            Pred&& pred, Proj&& proj = Proj())
        {
            using iterator = hpx::traits::range_iterator_t<Rng>;

            static_assert(hpx::traits::is_bidirectional_iterator_v<iterator>,
                "Requires at least bidirectional iterator.");

            return hpx::parallel::util::make_subrange<
                typename hpx::traits::range_iterator<Rng>::type,
                typename hpx::traits::range_sentinel<Rng>::type>(
                hpx::parallel::v1::detail::stable_partition<iterator>().call2(
                    hpx::execution::seq, std::true_type{},
                    hpx::util::begin(rng), hpx::util::end(rng),
                    std::forward<Pred>(pred), std::forward<Proj>(proj)),
                hpx::util::end(rng));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename Pred,
            typename Proj = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_range_v<Rng> &&
                parallel::traits::is_projected_range_v<Proj, Rng> &&
                parallel::traits::is_indirect_callable<ExPolicy, Pred,
                    parallel::traits::projected_range<Proj, Rng>>::value
        )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            subrange_t<hpx::traits::range_iterator_t<Rng>>>::type
        tag_fallback_dispatch(hpx::ranges::stable_partition_t,
            ExPolicy&& policy, Rng&& rng, Pred&& pred, Proj&& proj = Proj())
        {
            using iterator = hpx::traits::range_iterator_t<Rng>;

            static_assert(hpx::traits::is_bidirectional_iterator_v<iterator>,
                "Requires at least bidirectional iterator.");

            using is_seq = std::integral_constant<bool,
                hpx::is_sequenced_execution_policy_v<ExPolicy> ||
                    !hpx::traits::is_random_access_iterator_v<iterator>>;

            return hpx::parallel::util::make_subrange<
                typename hpx::traits::range_iterator<Rng>::type,
                typename hpx::traits::range_sentinel<Rng>::type>(
                hpx::parallel::v1::detail::stable_partition<iterator>().call2(
                    std::forward<ExPolicy>(policy), is_seq(),
                    hpx::util::begin(rng), hpx::util::end(rng),
                    std::forward<Pred>(pred), std::forward<Proj>(proj)),
                hpx::util::end(rng));
        }

        // clang-format off
        template <typename BidirIter, typename Sent, typename Pred,
            typename Proj = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<BidirIter> &&
                hpx::traits::is_sentinel_for_v<Sent, BidirIter> &&
                parallel::traits::is_indirect_callable_v<
                    hpx::execution::sequenced_policy, Pred,
                    parallel::traits::projected<Proj, BidirIter>>
        )>
        // clang-format on
        friend subrange_t<BidirIter> tag_fallback_dispatch(
            hpx::ranges::stable_partition_t, BidirIter first, Sent last,
            Pred&& pred, Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_bidirectional_iterator_v<BidirIter>,
                "Requires at least bidirectional iterator.");

            return hpx::parallel::util::make_subrange<BidirIter, BidirIter>(
                hpx::parallel::v1::detail::stable_partition<BidirIter>().call2(
                    hpx::execution::seq, std::true_type{}, first, last,
                    std::forward<Pred>(pred), std::forward<Proj>(proj)),
                parallel::v1::detail::advance_to_sentinel(first, last));
        }

        // clang-format off
        template <typename ExPolicy, typename BidirIter, typename Sent,
            typename Pred,
            typename Proj = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<BidirIter> &&
                hpx::traits::is_sentinel_for_v<Sent, BidirIter> &&
                parallel::traits::is_indirect_callable_v<ExPolicy, Pred,
                    parallel::traits::projected<Proj, BidirIter>>
        )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            subrange_t<BidirIter>>::type
        tag_fallback_dispatch(hpx::ranges::stable_partition_t,
            ExPolicy&& policy, BidirIter first, Sent last, Pred&& pred,
            Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_bidirectional_iterator_v<BidirIter>,
                "Requires at least bidirectional iterator.");

            using is_seq = std::integral_constant<bool,
                !hpx::traits::is_random_access_iterator_v<BidirIter>>;

            return hpx::parallel::util::make_subrange<BidirIter, BidirIter>(
                hpx::parallel::v1::detail::stable_partition<BidirIter>().call2(
                    std::forward<ExPolicy>(policy), is_seq(), first, last,
                    std::forward<Pred>(pred), std::forward<Proj>(proj)),
                parallel::v1::detail::advance_to_sentinel(first, last));
        }
    } stable_partition{};

    HPX_INLINE_CONSTEXPR_VARIABLE struct partition_copy_t final
      : hpx::detail::tag_parallel_algorithm<partition_copy_t>
    {
    private:
        // clang-format off
        template <typename Rng, typename OutIter2,
            typename OutIter3, typename Pred,
            typename Proj = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range_v<Rng> &&
                hpx::traits::is_iterator_v<OutIter2> &&
                hpx::traits::is_iterator_v<OutIter3> &&
                parallel::traits::is_projected_range_v<Proj, Rng> &&
                parallel::traits::is_indirect_callable_v<
                    hpx::execution::sequenced_policy, Pred,
                    parallel::traits::projected_range<Proj, Rng>>
            )>
        // clang-format on
        friend partition_copy_result<hpx::traits::range_iterator_t<Rng>,
            OutIter2, OutIter3>
        tag_fallback_dispatch(hpx::ranges::partition_copy_t, Rng&& rng,
            OutIter2 dest_true, OutIter3 dest_false, Pred&& pred,
            Proj&& proj = Proj())
        {
            using iterator = hpx::traits::range_iterator_t<Rng>;
            using result_type = hpx::tuple<iterator, OutIter2, OutIter3>;

            static_assert(hpx::traits::is_input_iterator_v<iterator>,
                "Requires at least input iterator.");

            return parallel::util::make_in_out_out_result(
                parallel::v1::detail::partition_copy<result_type>().call(
                    hpx::execution::seq, hpx::util::begin(rng),
                    hpx::util::end(rng), dest_true, dest_false,
                    std::forward<Pred>(pred), std::forward<Proj>(proj)));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename FwdIter2,
            typename FwdIter3, typename Pred,
            typename Proj = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_range_v<Rng> &&
                hpx::traits::is_iterator_v<FwdIter2> &&
                hpx::traits::is_iterator_v<FwdIter3> &&
                parallel::traits::is_projected_range_v<Proj, Rng> &&
                parallel::traits::is_indirect_callable_v<ExPolicy, Pred,
                    parallel::traits::projected_range<Proj, Rng>>
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            partition_copy_result<hpx::traits::range_iterator_t<Rng>, FwdIter2,
                FwdIter3>>::type
        tag_fallback_dispatch(hpx::ranges::partition_copy_t, ExPolicy&& policy,
            Rng&& rng, FwdIter2 dest_true, FwdIter3 dest_false, Pred&& pred,
            Proj&& proj = Proj())
        {
            using iterator = hpx::traits::range_iterator_t<Rng>;
            using result_type = hpx::tuple<iterator, FwdIter2, FwdIter3>;

            static_assert(hpx::traits::is_forward_iterator_v<iterator>,
                "Requires at least forward iterator.");

            return parallel::util::make_in_out_out_result(
                parallel::v1::detail::partition_copy<result_type>().call(
                    std::forward<ExPolicy>(policy), hpx::util::begin(rng),
                    hpx::util::end(rng), dest_true, dest_false,
                    std::forward<Pred>(pred), std::forward<Proj>(proj)));
        }

        // clang-format off
        template <typename InIter, typename Sent, typename OutIter2,
            typename OutIter3, typename Pred,
            typename Proj = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<InIter> &&
                hpx::traits::is_sentinel_for<Sent, InIter>::value &&
                hpx::traits::is_iterator_v<OutIter2> &&
                hpx::traits::is_iterator_v<OutIter3> &&
                parallel::traits::is_projected_v<Proj, InIter> &&
                parallel::traits::is_indirect_callable_v<
                    hpx::execution::sequenced_policy, Pred,
                    parallel::traits::projected<Proj, InIter>>
            )>
        // clang-format on
        friend partition_copy_result<InIter, OutIter2, OutIter3>
        tag_fallback_dispatch(hpx::ranges::partition_copy_t, InIter first,
            Sent last, OutIter2 dest_true, OutIter3 dest_false, Pred&& pred,
            Proj&& proj = Proj())
        {
            using result_type = hpx::tuple<InIter, OutIter2, OutIter3>;

            static_assert(hpx::traits::is_input_iterator_v<InIter>,
                "Requires at least input iterator.");

            return parallel::util::make_in_out_out_result(
                parallel::v1::detail::partition_copy<result_type>().call(
                    hpx::execution::seq, first, last, dest_true, dest_false,
                    std::forward<Pred>(pred), std::forward<Proj>(proj)));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename OutIter2, typename OutIter3, typename Pred,
            typename Proj = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<FwdIter> &&
                hpx::traits::is_sentinel_for_v<Sent, FwdIter> &&
                hpx::traits::is_iterator_v<OutIter2> &&
                hpx::traits::is_iterator_v<OutIter3> &&
                parallel::traits::is_projected_v<Proj, FwdIter> &&
                parallel::traits::is_indirect_callable_v<
                    ExPolicy, Pred,
                    parallel::traits::projected<Proj, FwdIter>>
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            partition_copy_result<FwdIter, OutIter2, OutIter3>>::type
        tag_fallback_dispatch(hpx::ranges::partition_copy_t, ExPolicy&& policy,
            FwdIter first, Sent last, OutIter2 dest_true, OutIter3 dest_false,
            Pred&& pred, Proj&& proj = Proj())
        {
            using result_type = hpx::tuple<FwdIter, OutIter2, OutIter3>;

            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return parallel::util::make_in_out_out_result(
                parallel::v1::detail::partition_copy<result_type>().call(
                    std::forward<ExPolicy>(policy), first, last, dest_true,
                    dest_false, std::forward<Pred>(pred),
                    std::forward<Proj>(proj)));
        }
    } partition_copy{};
}}    // namespace hpx::ranges

#endif    // DOXYGEN
