//  Copyright (c) 2015 Hartmut Kaiser
//  Copyright (c) 2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/replace.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx { namespace ranges {

    ///////////////////////////////////////////////////////////////////////////
    /// Replaces all elements satisfying specific criteria with \a new_value
    /// in the range [first, last).
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// Effects: Substitutes elements referred by the iterator it in the range
    ///          [first,last) with new_value, when the following corresponding
    ///          conditions hold: INVOKE(proj, *i) == old_value
    ///
    /// \tparam Iter        The type of the source iterator used (deduced).
    ///                     The iterator type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Sent        The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for Iter.
    /// \tparam T1          The type of the old value to replace (deduced).
    /// \tparam T2          The type of the new values to replace (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param old_value    Refers to the old value of the elements to replace.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a replace algorithm returns an \a Iter.
    ///
    template <typename Iter, typename Sent, typename T1, typename T2,
        typename Proj = hpx::parallel::util::projection_identity>
    Iter replace(Iter first, Sent sent, T1 const& old_value,
        T2 const& new_value, Proj&& proj = Proj());

    /// Replaces all elements satisfying specific criteria with \a new_value
    /// in the range \a rng.
    ///
    /// \note   Complexity: Performs exactly \a util::end(rng) - \a util::begin(rng)
    ///         assignments.
    ///
    /// Effects: Substitutes elements referred by the iterator it in the range
    ///          rng with new_value, when the following corresponding
    ///          conditions hold: INVOKE(proj, *i) == old_value
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam T1          The type of the old value to replace (deduced).
    /// \tparam T2          The type of the new values to replace (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param old_value    Refers to the old value of the elements to replace.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a replace algorithm returns an
    ///           \a hpx::traits::range_iterator<Rng>::type.
    ///
    template <typename Rng, typename T1, typename T2,
        typename Proj = hpx::parallel::util::projection_identity>
    typename hpx::traits::range_iterator<Rng>::type replace(Rng&& rng,
        T1 const& old_value, T2 const& new_value, Proj&& proj = Proj());

    /// Replaces all elements satisfying specific criteria with \a new_value
    /// in the range [first, last).
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// Effects: Substitutes elements referred by the iterator it in the range
    ///          [first,last) with new_value, when the following corresponding
    ///          conditions hold: INVOKE(proj, *i) == old_value
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Iter        The type of the source iterator used (deduced).
    ///                     The iterator type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam Sent        The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for Iter.
    /// \tparam T1          The type of the old value to replace (deduced).
    /// \tparam T2          The type of the new values to replace (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param old_value    Refers to the old value of the elements to replace.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace algorithm invoked with an
    /// execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a replace algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a replace algorithm returns a \a hpx::future<Iter> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a Iter otherwise.
    ///
    template <typename ExPolicy, typename Iter, typename Sent, typename T1,
        typename T2, typename Proj = hpx::parallel::util::projection_identity>
    typename parallel::util::detail::algorithm_result<ExPolicy, Iter>::type
    replace(ExPolicy&& policy, Iter first, Sent sent, T1 const& old_value,
        T2 const& new_value, Proj&& proj = Proj());

    /// Replaces all elements satisfying specific criteria with \a new_value
    /// in the range \a rng.
    ///
    /// \note   Complexity: Performs exactly \a util::end(rng) - \a util::begin(rng)
    ///         assignments.
    ///
    /// Effects: Substitutes elements referred by the iterator it in the range
    ///          rng with new_value, when the following corresponding
    ///          conditions hold: INVOKE(proj, *i) == old_value
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam T1          The type of the old value to replace (deduced).
    /// \tparam T2          The type of the new values to replace (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param old_value    Refers to the old value of the elements to replace.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    /// The assignments in the parallel \a replace algorithm invoked with an
    /// execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a replace algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a replace algorithm returns an
    ///           \a hpx::future<hpx::traits::range_iterator<Rng>::type> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a hpx::traits::range_iterator<Rng>::type otherwise.
    ///
    template <typename ExPolicy, typename Rng, typename T1, typename T2,
        typename Proj = hpx::parallel::util::projection_identity>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        typename hpx::traits::range_iterator<Rng>::type>::type
    replace(ExPolicy&& policy, Rng&& rng, T1 const& old_value,
        T2 const& new_value, Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Replaces all elements satisfying specific criteria (for which predicate
    /// \a f returns true) with \a new_value in the range [first, sent).
    ///
    /// \note   Complexity: Performs exactly \a sent - \a first applications of
    ///         the predicate.
    ///
    /// Effects: Substitutes elements referred by the iterator it in the range
    ///          [first, sent) with new_value, when the following corresponding
    ///          conditions hold: INVOKE(f, INVOKE(proj, *it)) != false
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Iter        The type of the source iterator used (deduced).
    ///                     The iterator type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam Sent        The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for Iter.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///                     (deduced).
    /// \tparam T           The type of the new values to replace (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     elements which need to replaced. The
    ///                     signature of this predicate should be equivalent
    ///                     to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a Iter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace_if algorithm
    /// execute in sequential order in the calling thread.
    ///
    ///
    /// \returns  The \a replace_if algorithm returns an \a Iter
    ///           It returns \a last.
    ///
    template <typename Iter, typename Sent, typename Pred, typename T,
        typename Proj = hpx::parallel::util::projection_identity>
    Iter replace_if(Iter first, Sent sent, Pred&& pred, T const& new_value,
        Proj&& proj = Proj());

    /// Replaces all elements satisfying specific criteria (for which predicate
    /// \a pred returns true) with \a new_value in the range rng.
    ///
    /// \note   Complexity: Performs exactly \a util::end(rng) - \a util::begin(rng)
    ///         applications of the predicate.
    ///
    /// Effects: Substitutes elements referred by the iterator it in the range
    ///          rng with new_value, when the following corresponding
    ///          conditions hold: INVOKE(f, INVOKE(proj, *it)) != false
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///                     (deduced).
    /// \tparam T           The type of the new values to replace (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by rng.This is an
    ///                     unary predicate which returns \a true for the
    ///                     elements which need to replaced. The
    ///                     signature of this predicate should be equivalent
    ///                     to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace algorithm invoked with an
    /// execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a replace algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a replace_if algorithm returns an \a hpx::traits::range_iterator<Rng>::type
    ///           It returns \a last.
    ///
    template <typename Rng, typename Pred, typename T,
        typename Proj = hpx::parallel::util::projection_identity>
    typename hpx::traits::range_iterator<Rng>::type replace_if(
        Rng&& rng, Pred&& pred, T const& new_value, Proj&& proj = Proj());

    /// Replaces all elements satisfying specific criteria (for which predicate
    /// \a pred returns true) with \a new_value in the range rng.
    ///
    /// \note   Complexity: Performs exactly \a util::end(rng) - \a util::begin(rng)
    ///         applications of the predicate.
    ///
    /// Effects: Substitutes elements referred by the iterator it in the range
    ///          rng with new_value, when the following corresponding
    ///          conditions hold: INVOKE(f, INVOKE(proj, *it)) != false
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Iter        The type of the source iterator used (deduced).
    ///                     The iterator type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam Sent        The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for Iter.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible.
    ///                     (deduced).
    /// \tparam T           The type of the new values to replace (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     elements which need to replaced. The
    ///                     signature of this predicate should be equivalent
    ///                     to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace_if algorithm invoked with an
    /// execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a replace_if algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a replace_if algorithm returns a \a hpx::future<Iter>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy.
    ///           It returns \a last.
    ///
    template <typename ExPolicy, typename Iter, typename Sent, typename Pred,
        typename T, typename Proj = hpx::parallel::util::projection_identity>
    typename parallel::util::detail::algorithm_result<ExPolicy, Iter>::type t
    replace_if(ExPolicy&& policy, Iter first, Sent sent, Pred&& pred,
        T const& new_value, Proj&& proj = Proj());

    /// Replaces all elements satisfying specific criteria (for which predicate
    /// \a pred returns true) with \a new_value in the range rng.
    ///
    /// \note   Complexity: Performs exactly \a util::end(rng) - \a util::begin(rng)
    ///         applications of the predicate.
    ///
    /// Effects: Substitutes elements referred by the iterator it in the range
    ///          rng with new_value, when the following corresponding
    ///          conditions hold: INVOKE(f, INVOKE(proj, *it)) != false
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///                     (deduced).
    /// \tparam T           The type of the new values to replace (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by rng.This is an
    ///                     unary predicate which returns \a true for the
    ///                     elements which need to replaced. The
    ///                     signature of this predicate should be equivalent
    ///                     to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace algorithm invoked with an
    /// execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a replace algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a replace_if algorithm returns a \a
    ///           hpx::future<typename hpx::traits::range_iterator<Rng>::type>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy.
    ///           It returns \a last.
    ///
    template <typename ExPolicy, typename Rng, typename Pred, typename T,
        typename Proj = hpx::parallel::util::projection_identity>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        typename hpx::traits::range_iterator<Rng>::type>::type
    replace_if(ExPolicy&& policy, Rng&& rng, Pred&& pred, T const& new_value,
        Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Copies the all elements from the range [first, sent) to another range
    /// beginning at \a dest replacing all elements satisfying a specific
    /// criteria with \a new_value.
    ///
    /// Effects: Assigns to every iterator it in the range
    ///          [result, result + (sent - first)) either new_value or
    ///          *(first + (it - result)) depending on whether the following
    ///          corresponding condition holds:
    ///          INVOKE(proj, *(first + (i - result))) == old_value
    ///
    /// \note   Complexity: Performs exactly \a sent - \a first applications of
    ///         the predicate.
    ///
    /// \tparam Iter        The type of the source iterator used (deduced).
    ///                     The iterator type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Sent        The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for Iter.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam T1          The type of the old value to replace (deduced).
    /// \tparam T2          The type of the new values to replace (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param sent         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param old_value    Refers to the old value of the elements to replace.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace_copy algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a replace_copy algorithm returns an
    ///           \a in_out_result<InIter, OutIter>.
    ///           The \a copy algorithm returns the pair of the input iterator
    ///           \a last and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename Initer, typename Sent, typename OutIter, typename T1,
        typename T2, typename Proj = hpx::parallel::util::projection_identity>
    replace_copy_result<InIter, OutIter> replace_copy(InIter first, Sent sent,
        OutIter dest, T1 const& old_value, T2 const& new_value,
        Proj&& proj = Proj());

    /// Copies the all elements from the range rbg to another range
    /// beginning at \a dest replacing all elements satisfying a specific
    /// criteria with \a new_value.
    ///
    /// Effects: Assigns to every iterator it in the range
    ///          [result, result + (util::end(rng) - util::begin(rng))) either new_value or
    ///          *(first + (it - result)) depending on whether the following
    ///          corresponding condition holds:
    ///          INVOKE(proj, *(first + (i - result))) == old_value
    ///
    /// \note   Complexity: Performs exactly \a util::end(rng) - \a util::begin(rng)
    ///         applications of the predicate.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam T1          The type of the old value to replace (deduced).
    /// \tparam T2          The type of the new values to replace (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param old_value    Refers to the old value of the elements to replace.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace_copy algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a replace_copy algorithm returns an
    ///           \a in_out_result<typename hpx::traits::range_iterator<
    ///             Rng>::type, OutIter>.
    ///           The \a copy algorithm returns the pair of the input iterator
    ///           \a last and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename Rng, typename OutIter, typename T1, typename T2,
        typename Proj = hpx::parallel::util::projection_identity>
    replace_copy_result<typename hpx::traits::range_iterator<Rng>::type,
        OutIter>
    replace_copy(Rng&& rng, OutIter dest, T1 const& old_value,
        T2 const& new_value, Proj&& proj = Proj());

    /// Copies the all elements from the range [first, sent) to another range
    /// beginning at \a dest replacing all elements satisfying a specific
    /// criteria with \a new_value.
    ///
    /// Effects: Assigns to every iterator it in the range
    ///          [result, result + (sent - first)) either new_value or
    ///          *(first + (it - result)) depending on whether the following
    ///          corresponding condition holds:
    ///          INVOKE(proj, *(first + (i - result))) == old_value
    ///
    /// \note   Complexity: Performs exactly \a sent - \a first applications of
    ///         the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterator used (deduced).
    ///                     The iterator type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Sent        The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for Iter.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam T1          The type of the old value to replace (deduced).
    /// \tparam T2          The type of the new values to replace (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param sent         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param old_value    Refers to the old value of the elements to replace.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace_copy algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a replace_copy algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a replace_copy algorithm returns a
    ///           \a hpx::future<in_out_result<FwdIter1, FwdIter2>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a in_out_result<FwdIter1, FwdIter2>
    ///           otherwise.
    ///           The \a copy algorithm returns the pair of the forward iterator
    ///           \a last and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Sent,
        typename FwdIter2, typename T1, typename T2,
        typename Proj = hpx::parallel::util::projection_identity>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        replace_copy_result<FwdIter1, FwdIter2>>::type
    replace_copy(ExPolicy&& policy, FwdIter1 first, Sent sent, FwdIter2 dest,
        T1 const& old_value, T2 const& new_value, Proj&& proj = Proj());

    /// Copies the all elements from the range rbg to another range
    /// beginning at \a dest replacing all elements satisfying a specific
    /// criteria with \a new_value.
    ///
    /// Effects: Assigns to every iterator it in the range
    ///          [result, result + (util::end(rng) - util::begin(rng))) either new_value or
    ///          *(first + (it - result)) depending on whether the following
    ///          corresponding condition holds:
    ///          INVOKE(proj, *(first + (i - result))) == old_value
    ///
    /// \note   Complexity: Performs exactly \a util::end(rng) - \a util::begin(rng)
    ///         applications of the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam FwdIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam T1          The type of the old value to replace (deduced).
    /// \tparam T2          The type of the new values to replace (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param old_value    Refers to the old value of the elements to replace.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace_copy algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a replace_copy algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a replace_copy algorithm returns a
    ///           \a hpx::future<in_out_result<
    ///            typename hpx::traits::range_iterator<Rng>::type, FwdIter>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a in_out_result<
    ///            typename hpx::traits::range_iterator<Rng>::type, FwdIter>>
    ///           The \a copy algorithm returns the pair of the input iterator
    ///           \a last and the forward iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename Rng, typename FwdIter, typename T1,
        typename T2, typename Proj = hpx::parallel::util::projection_identity>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        replace_copy_result<typename hpx::traits::range_iterator<Rng>::type,
            FwdIter>>::type
    replace_copy(ExPolicy&& policy, Rng&& rng, FwdIter dest,
        T1 const& old_value, T2 const& new_value, Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Copies the all elements from the range [first, sent) to another range
    /// beginning at \a dest replacing all elements satisfying a specific
    /// criteria with \a new_value.
    ///
    /// Effects: Assigns to every iterator it in the range
    ///          [result, result + (sent - first)) either new_value or
    ///          *(first + (it - result)) depending on whether the following
    ///          corresponding condition holds:
    ///          INVOKE(f, INVOKE(proj, *(first + (i - result)))) != false
    ///
    /// \note   Complexity: Performs exactly \a sent - \a first applications of
    ///         the predicate.
    ///
    /// \tparam InIter      The type of the source iterator used (deduced).
    ///                     The iterator type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Sent        The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible.
    ///                     (deduced).
    /// \tparam T           The type of the new values to replace (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param sent         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     elements which need to replaced. The
    ///                     signature of this predicate should be equivalent
    ///                     to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace_copy_if algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a replace_copy_if algorithm returns a
    ///           \a in_out_result<InIter, OutIter>.
    ///           The \a replace_copy_if algorithm returns the input iterator
    ///           \a last and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename InIter, typename Sent, typename OutIter, typename Pred,
        typename T, typename Proj = hpx::parallel::util::projection_identity>
    replace_copy_if_result<InIter, OutIter> replace_copy_if(InIter first,
        Sent sent, OutIter dest, Pred&& pred, T const& new_value,
        Proj&& proj = Proj());

    /// Copies the all elements from the range rng to another range
    /// beginning at \a dest replacing all elements satisfying a specific
    /// criteria with \a new_value.
    ///
    /// Effects: Assigns to every iterator it in the range
    ///          [result, result + (util::end(rng) - util::begin(rng))) either new_value or
    ///          *(first + (it - result)) depending on whether the following
    ///          corresponding condition holds:
    ///          INVOKE(f, INVOKE(proj, *(first + (i - result)))) != false
    ///
    /// \note   Complexity: Performs exactly \a util::end(rng) - \a util::begin(rng)
    ///         applications of the predicate.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible.
    ///                     (deduced).
    /// \tparam T           The type of the new values to replace (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     elements which need to replaced. The
    ///                     signature of this predicate should be equivalent
    ///                     to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace_copy_if algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a replace_copy_if algorithm returns an
    ///           \a in_out_result<typename hpx::traits::range_iterator<Rng>::type,
    ///             OutIter>.
    ///           The \a replace_copy_if algorithm returns the input iterator
    ///           \a last and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename Rng, typename OutIter, typename Pred, typename T,
        typename Proj = hpx::parallel::util::projection_identity>
    replace_copy_if_result<typename hpx::traits::range_iterator<Rng>::type,
        OutIter>
    replace_copy_if(Rng&& rng, OutIter dest, Pred&& pred, T const& new_value,
        Proj&& proj = Proj());

    /// Copies the all elements from the range [first, sent) to another range
    /// beginning at \a dest replacing all elements satisfying a specific
    /// criteria with \a new_value.
    ///
    /// Effects: Assigns to every iterator it in the range
    ///          [result, result + (sent - first)) either new_value or
    ///          *(first + (it - result)) depending on whether the following
    ///          corresponding condition holds:
    ///          INVOKE(f, INVOKE(proj, *(first + (i - result)))) != false
    ///
    /// \note   Complexity: Performs exactly \a sent - \a first applications of
    ///         the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterator used (deduced).
    ///                     The iterator type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam Sent        The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible.
    ///                     (deduced).
    /// \tparam T           The type of the new values to replace (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param sent         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     elements which need to replaced. The
    ///                     signature of this predicate should be equivalent
    ///                     to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace_copy_if algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a replace_copy_if algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a replace_copy_if algorithm returns an
    ///           \a hpx::future<FwdIter1, FwdIter2>.
    ///           The \a replace_copy_if algorithm returns the input iterator
    ///           \a last and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Sent,
        typename FwdIter2, typename Pred, typename T,
        typename Proj = hpx::parallel::util::projection_identity>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        replace_copy_if_result<FwdIter1, FwdIter2>>::type
    replace_copy_if(ExPolicy&& policy, FwdIter1 first, Sent sent, FwdIter2 dest,
        Pred&& pred, T const& new_value, Proj&& proj = Proj());

    /// Copies the all elements from the range rng to another range
    /// beginning at \a dest replacing all elements satisfying a specific
    /// criteria with \a new_value.
    ///
    /// Effects: Assigns to every iterator it in the range
    ///          [result, result + (util::end(rng) - util::begin(rng))) either new_value or
    ///          *(first + (it - result)) depending on whether the following
    ///          corresponding condition holds:
    ///          INVOKE(f, INVOKE(proj, *(first + (i - result)))) != false
    ///
    /// \note   Complexity: Performs exactly \a util::end(rng) - \a util::begin(rng)
    ///         applications of the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible.
    ///                     (deduced).
    /// \tparam T           The type of the new values to replace (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     elements which need to replaced. The
    ///                     signature of this predicate should be equivalent
    ///                     to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace_copy_if algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a replace_copy_if algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a replace_copy_if algorithm returns an
    ///           \a hpx::future<in_out_result<typename hpx::traits::range_iterator<Rng>::type,
    ///             OutIter>>.
    ///           The \a replace_copy_if algorithm returns the input iterator
    ///           \a last and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename Rng, typename FwdIter, typename Pred,
        typename T, typename Proj = hpx::parallel::util::projection_identity>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        replace_copy_if_result<typename hpx::traits::range_iterator<Rng>::type,
            FwdIter>>::type
    replace_copy_if(ExPolicy&& policy, Rng&& rng, FwdIter dest, Pred&& pred,
        T const& new_value, Proj&& proj = Proj());

}}    // namespace hpx::ranges

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/tagged_pair.hpp>

#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/parallel/algorithms/replace.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {

    // clang-format off
    template <typename ExPolicy, typename Rng, typename T1, typename T2,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_range<Rng>::value &&
            traits::is_projected_range<Proj, Rng>::value &&
            traits::is_indirect_callable<ExPolicy,
                std::equal_to<T1>, traits::projected_range<Proj, Rng>,
                traits::projected<Proj, T1 const*>>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 7,
        "hpx::parallel::replace is deprecated, use hpx::ranges::replace "
        "instead") typename util::detail::algorithm_result<ExPolicy,
        typename hpx::traits::range_traits<Rng>::iterator_type>::type
        replace(ExPolicy&& policy, Rng&& rng, T1 const& old_value,
            T2 const& new_value, Proj&& proj = Proj())
    {
        return hpx::parallel::v1::detail::replace<
            typename hpx::traits::range_traits<Rng>::iterator_type>()
            .call(std::forward<ExPolicy>(policy), hpx::util::begin(rng),
                hpx::util::end(rng), old_value, new_value,
                std::forward<Proj>(proj));
    }

    // clang-format off
    template <typename ExPolicy, typename Rng, typename F, typename T,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_range<Rng>::value &&
            traits::is_projected_range<Proj, Rng>::value &&
            traits::is_indirect_callable<ExPolicy,
                    F, traits::projected_range<Proj, Rng>>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 7,
        "hpx::parallel::replace_if is deprecated, use hpx::ranges::replace_if "
        "instead") typename util::detail::algorithm_result<ExPolicy,
        typename hpx::traits::range_traits<Rng>::iterator_type>::type
        replace_if(ExPolicy&& policy, Rng&& rng, F&& f, T const& new_value,
            Proj&& proj = Proj())
    {
        return hpx::parallel::v1::detail::replace_if<
            typename hpx::traits::range_traits<Rng>::iterator_type>()
            .call(std::forward<ExPolicy>(policy), hpx::util::begin(rng),
                hpx::util::end(rng), std::forward<F>(f), new_value,
                std::forward<Proj>(proj));
    }

    // clang-format off
    template <typename ExPolicy, typename Rng, typename OutIter, typename T1,
        typename T2, typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_range<Rng>::value &&
            traits::is_projected_range<Proj, Rng>::value &&
            traits::is_indirect_callable<
                ExPolicy,std::equal_to<T1>, traits::projected_range<Proj, Rng>,
                    traits::projected<Proj, T1 const*>>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 7,
        "hpx::parallel::replace_copy is deprecated, use "
        "hpx::ranges::replace_copy "
        "instead") typename util::detail::algorithm_result<ExPolicy,
        util::in_out_result<
            typename hpx::traits::range_traits<Rng>::iterator_type, OutIter>>::
        type replace_copy(ExPolicy&& policy, Rng&& rng, OutIter dest,
            T1 const& old_value, T2 const& new_value, Proj&& proj = Proj())
    {
        return hpx::parallel::v1::detail::replace_copy<util::in_out_result<
            typename hpx::traits::range_traits<Rng>::iterator_type, OutIter>>()
            .call(std::forward<ExPolicy>(policy), hpx::util::begin(rng),
                hpx::util::end(rng), dest, old_value, new_value,
                std::forward<Proj>(proj));
    }

    // clang-format off
    template <typename ExPolicy, typename Rng, typename OutIter, typename F,
        typename T, typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_range<Rng>::value &&
            traits::is_projected_range<Proj, Rng>::value &&
            traits::is_indirect_callable<ExPolicy,
                F, traits::projected_range<Proj, Rng>>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 7,
        "hpx::parallel::replace_copy_if is deprecated, use "
        "hpx::ranges::replace_copy_if "
        "instead") typename util::detail::algorithm_result<ExPolicy,
        util::in_out_result<
            typename hpx::traits::range_traits<Rng>::iterator_type,
            OutIter>>::type replace_copy_if(ExPolicy&& policy, Rng&& rng,
        OutIter dest, F&& f, T const& new_value, Proj&& proj = Proj())
    {
        return hpx::parallel::v1::detail::replace_copy_if<util::in_out_result<
            typename hpx::traits::range_traits<Rng>::iterator_type, OutIter>>()
            .call(std::forward<ExPolicy>(policy), hpx::util::begin(rng),
                hpx::util::end(rng), dest, std::forward<F>(f), new_value,
                std::forward<Proj>(proj));
    }
}}}    // namespace hpx::parallel::v1

namespace hpx { namespace ranges {

    /// `replace_copy_if_result` is equivalent to
    /// `hpx::parallel::util::in_out_result`
    template <typename I, typename O>
    using replace_copy_if_result = hpx::parallel::util::in_out_result<I, O>;

    /// `replace_copy_result` is equivalent to
    /// `hpx::parallel::util::in_out_result`
    template <typename I, typename O>
    using replace_copy_result = hpx::parallel::util::in_out_result<I, O>;

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::ranges::replace_if
    HPX_INLINE_CONSTEXPR_VARIABLE struct replace_if_t final
      : hpx::detail::tag_parallel_algorithm<replace_if_t>
    {
    private:
        // clang-format off
        template <typename Iter, typename Sent, typename Pred,
        typename T, typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<Iter>::value &&
                hpx::parallel::traits::is_projected<Proj, Iter>::value &&
                hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<Iter>::value_type
                >
            )>
        // clang-format on
        friend Iter tag_fallback_dispatch(hpx::ranges::replace_if_t, Iter first,
            Sent sent, Pred&& pred, T const& new_value, Proj&& proj = Proj())
        {
            static_assert((hpx::traits::is_input_iterator<Iter>::value),
                "Required at least input iterator.");

            return hpx::parallel::v1::detail::replace_if<Iter>().call(
                hpx::execution::seq, first, sent, std::forward<Pred>(pred),
                new_value, std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename Rng, typename Pred,
        typename T, typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<
                        typename hpx::traits::range_iterator<Rng>::type
                    >::value_type
                >
            )>
        // clang-format on
        friend typename hpx::traits::range_iterator<Rng>::type
        tag_fallback_dispatch(hpx::ranges::replace_if_t, Rng&& rng, Pred&& pred,
            T const& new_value, Proj&& proj = Proj())
        {
            static_assert(
                (hpx::traits::is_input_iterator<
                    typename hpx::traits::range_iterator<Rng>::type>::value),
                "Required at least input iterator.");

            return hpx::parallel::v1::detail::replace_if<
                typename hpx::traits::range_iterator<Rng>::type>()
                .call(hpx::execution::seq, hpx::util::begin(rng),
                    hpx::util::end(rng), std::forward<Pred>(pred), new_value,
                    std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent, typename Pred,
        typename T, typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<Iter>::value &&
                hpx::parallel::traits::is_projected<Proj, Iter>::value &&
                hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy,
                    Pred, hpx::parallel::traits::projected<Proj, Iter>>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            Iter>::type
        tag_fallback_dispatch(hpx::ranges::replace_if_t, ExPolicy&& policy,
            Iter first, Sent sent, Pred&& pred, T const& new_value,
            Proj&& proj = Proj())
        {
            static_assert((hpx::traits::is_forward_iterator<Iter>::value),
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::replace_if<Iter>().call(
                std::forward<ExPolicy>(policy), first, sent,
                std::forward<Pred>(pred), new_value, std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename Pred,
        typename T, typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy,
                    Pred, hpx::parallel::traits::projected_range<Proj, Rng>>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            typename hpx::traits::range_iterator<Rng>::type>::type
        tag_fallback_dispatch(hpx::ranges::replace_if_t, ExPolicy&& policy,
            Rng&& rng, Pred&& pred, T const& new_value, Proj&& proj = Proj())
        {
            static_assert(
                (hpx::traits::is_forward_iterator<
                    typename hpx::traits::range_iterator<Rng>::type>::value),
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::replace_if<
                typename hpx::traits::range_iterator<Rng>::type>()
                .call(std::forward<ExPolicy>(policy), hpx::util::begin(rng),
                    hpx::util::end(rng), std::forward<Pred>(pred), new_value,
                    std::forward<Proj>(proj));
        }
    } replace_if{};

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::ranges::replace
    HPX_INLINE_CONSTEXPR_VARIABLE struct replace_t final
      : hpx::detail::tag_parallel_algorithm<replace_t>
    {
    private:
        // clang-format off
        template <typename Iter, typename Sent, typename T1,
        typename T2, typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<Iter>::value &&
                hpx::parallel::traits::is_projected<Proj, Iter>::value &&
                hpx::traits::is_sentinel_for<Sent, Iter>::value
            )>
        // clang-format on
        friend Iter tag_fallback_dispatch(hpx::ranges::replace_t, Iter first,
            Sent sent, T1 const& old_value, T2 const& new_value,
            Proj&& proj = Proj())
        {
            static_assert((hpx::traits::is_input_iterator<Iter>::value),
                "Required at least input iterator.");

            typedef typename std::iterator_traits<Iter>::value_type Type;

            return hpx::ranges::replace_if(
                first, sent,
                [old_value](Type const& a) -> bool { return old_value == a; },
                new_value, std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename Rng, typename T1,
        typename T2, typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value
            )>
        // clang-format on
        friend typename hpx::traits::range_iterator<Rng>::type
        tag_fallback_dispatch(hpx::ranges::replace_t, Rng&& rng,
            T1 const& old_value, T2 const& new_value, Proj&& proj = Proj())
        {
            static_assert(
                (hpx::traits::is_input_iterator<
                    typename hpx::traits::range_iterator<Rng>::type>::value),
                "Required at least input iterator.");

            typedef typename std::iterator_traits<
                typename hpx::traits::range_iterator<Rng>::type>::value_type
                Type;

            return hpx::ranges::replace_if(
                std::forward<Rng>(rng),
                [old_value](Type const& a) -> bool { return old_value == a; },
                new_value, std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent, typename T1,
        typename T2, typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<Iter>::value &&
                hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                hpx::parallel::traits::is_projected<Proj, Iter>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            Iter>::type
        tag_fallback_dispatch(hpx::ranges::replace_t, ExPolicy&& policy,
            Iter first, Sent sent, T1 const& old_value, T2 const& new_value,
            Proj&& proj = Proj())
        {
            static_assert((hpx::traits::is_forward_iterator<Iter>::value),
                "Required at least forward iterator.");

            typedef typename std::iterator_traits<Iter>::value_type Type;

            return hpx::ranges::replace_if(
                std::forward<ExPolicy>(policy), first, sent,
                [old_value](Type const& a) -> bool { return old_value == a; },
                new_value, std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename T1,
        typename T2, typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            typename hpx::traits::range_iterator<Rng>::type>::type
        tag_fallback_dispatch(hpx::ranges::replace_t, ExPolicy&& policy,
            Rng&& rng, T1 const& old_value, T2 const& new_value,
            Proj&& proj = Proj())
        {
            static_assert(
                (hpx::traits::is_forward_iterator<
                    typename hpx::traits::range_iterator<Rng>::type>::value),
                "Required at least forward iterator.");

            typedef typename std::iterator_traits<
                typename hpx::traits::range_iterator<Rng>::type>::value_type
                Type;

            return hpx::ranges::replace_if(
                std::forward<ExPolicy>(policy), std::forward<Rng>(rng),
                [old_value](Type const& a) -> bool { return old_value == a; },
                new_value, std::forward<Proj>(proj));
        }
    } replace{};

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::ranges::replace_copy_if
    HPX_INLINE_CONSTEXPR_VARIABLE struct replace_copy_if_t final
      : hpx::detail::tag_parallel_algorithm<replace_copy_if_t>
    {
    private:
        // clang-format off
        template <typename InIter, typename Sent, typename OutIter, typename Pred,
        typename T, typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<InIter>::value &&
                hpx::traits::is_iterator<OutIter>::value &&
                hpx::parallel::traits::is_projected<Proj, InIter>::value &&
                hpx::traits::is_sentinel_for<Sent, InIter>::value &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<InIter>::value_type
                >
            )>
        // clang-format on
        friend replace_copy_if_result<InIter, OutIter> tag_fallback_dispatch(
            hpx::ranges::replace_copy_if_t, InIter first, Sent sent,
            OutIter dest, Pred&& pred, T const& new_value, Proj&& proj = Proj())
        {
            static_assert((hpx::traits::is_input_iterator<InIter>::value),
                "Required at least input iterator.");

            static_assert((hpx::traits::is_output_iterator<OutIter>::value),
                "Required at least output iterator.");

            return hpx::parallel::v1::detail::replace_copy_if<
                hpx::parallel::util::in_out_result<InIter, OutIter>>()
                .call(hpx::execution::seq, first, sent, dest,
                    std::forward<Pred>(pred), new_value,
                    std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename Rng, typename OutIter, typename Pred,
        typename T, typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::traits::is_iterator<OutIter>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<
                        typename hpx::traits::range_iterator<Rng>::type
                    >::value_type
                >
            )>
        // clang-format on
        friend replace_copy_if_result<
            typename hpx::traits::range_iterator<Rng>::type, OutIter>
        tag_fallback_dispatch(hpx::ranges::replace_copy_if_t, Rng&& rng,
            OutIter dest, Pred&& pred, T const& new_value, Proj&& proj = Proj())
        {
            static_assert(
                (hpx::traits::is_input_iterator<
                    typename hpx::traits::range_iterator<Rng>::type>::value),
                "Required at least input iterator.");

            static_assert((hpx::traits::is_output_iterator<OutIter>::value),
                "Required at least output iterator.");

            return hpx::parallel::v1::detail::replace_copy_if<
                hpx::parallel::util::in_out_result<
                    typename hpx::traits::range_iterator<Rng>::type, OutIter>>()
                .call(hpx::execution::seq, hpx::util::begin(rng),
                    hpx::util::end(rng), dest, std::forward<Pred>(pred),
                    new_value, std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename Sent,
        typename FwdIter2, typename Pred, typename T,
        typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::parallel::traits::is_projected<Proj, FwdIter1>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter1>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy,
                    Pred, hpx::parallel::traits::projected<Proj, FwdIter1>>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            replace_copy_if_result<FwdIter1, FwdIter2>>::type
        tag_fallback_dispatch(hpx::ranges::replace_copy_if_t, ExPolicy&& policy,
            FwdIter1 first, Sent sent, FwdIter2 dest, Pred&& pred,
            T const& new_value, Proj&& proj = Proj())
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Required at least forward iterator.");

            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::replace_copy_if<
                hpx::parallel::util::in_out_result<FwdIter1, FwdIter2>>()
                .call(std::forward<ExPolicy>(policy), first, sent, dest,
                    std::forward<Pred>(pred), new_value,
                    std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename FwdIter, typename Pred,
        typename T, typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::traits::is_iterator<FwdIter>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy,
                    Pred, hpx::parallel::traits::projected_range<Proj, Rng>>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            replace_copy_if_result<
                typename hpx::traits::range_iterator<Rng>::type, FwdIter>>::type
        tag_fallback_dispatch(hpx::ranges::replace_copy_if_t, ExPolicy&& policy,
            Rng&& rng, FwdIter dest, Pred&& pred, T const& new_value,
            Proj&& proj = Proj())
        {
            static_assert(
                (hpx::traits::is_forward_iterator<
                    typename hpx::traits::range_iterator<Rng>::type>::value),
                "Required at least forward iterator.");

            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::replace_copy_if<
                hpx::parallel::util::in_out_result<
                    typename hpx::traits::range_iterator<Rng>::type, FwdIter>>()
                .call(std::forward<ExPolicy>(policy), hpx::util::begin(rng),
                    hpx::util::end(rng), dest, std::forward<Pred>(pred),
                    new_value, std::forward<Proj>(proj));
        }
    } replace_copy_if{};

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::ranges::replace_copy
    HPX_INLINE_CONSTEXPR_VARIABLE struct replace_copy_t final
      : hpx::detail::tag_parallel_algorithm<replace_copy_t>
    {
    private:
        // clang-format off
        template <typename InIter, typename Sent,
        typename OutIter, typename T1, typename T2,
        typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<InIter>::value &&
                hpx::parallel::traits::is_projected<Proj, InIter>::value &&
                hpx::traits::is_sentinel_for<Sent, InIter>::value
            )>
        // clang-format on
        friend replace_copy_result<InIter, OutIter> tag_fallback_dispatch(
            hpx::ranges::replace_copy_t, InIter first, Sent sent, OutIter dest,
            T1 const& old_value, T2 const& new_value, Proj&& proj = Proj())
        {
            static_assert((hpx::traits::is_input_iterator<InIter>::value),
                "Required at least input iterator.");

            static_assert((hpx::traits::is_output_iterator<OutIter>::value),
                "Required at least output iterator.");

            typedef typename std::iterator_traits<InIter>::value_type Type;

            return hpx::ranges::replace_copy_if(
                hpx::execution::seq, first, sent, dest,
                [old_value](Type const& a) -> bool { return old_value == a; },
                new_value, std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename Rng, typename OutIter,
        typename T1, typename T2,
        typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value
            )>
        // clang-format on
        friend replace_copy_result<
            typename hpx::traits::range_iterator<Rng>::type, OutIter>
        tag_fallback_dispatch(hpx::ranges::replace_copy_t, Rng&& rng,
            OutIter dest, T1 const& old_value, T2 const& new_value,
            Proj&& proj = Proj())
        {
            static_assert(
                (hpx::traits::is_input_iterator<
                    typename hpx::traits::range_iterator<Rng>::type>::value),
                "Required at least input iterator.");

            static_assert((hpx::traits::is_output_iterator<OutIter>::value),
                "Required at least output iterator.");

            typedef typename std::iterator_traits<
                typename hpx::traits::range_iterator<Rng>::type>::value_type
                Type;

            return hpx::ranges::replace_copy_if(
                hpx::execution::seq, hpx::util::begin(rng), hpx::util::end(rng),
                dest,
                [old_value](Type const& a) -> bool { return old_value == a; },
                new_value, std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1,
        typename Sent, typename FwdIter2, typename T1,
        typename T2, typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter1>::value &&
                hpx::parallel::traits::is_projected<Proj, FwdIter1>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            replace_copy_result<FwdIter1, FwdIter2>>::type
        tag_fallback_dispatch(hpx::ranges::replace_copy_t, ExPolicy&& policy,
            FwdIter1 first, Sent sent, FwdIter2 dest, T1 const& old_value,
            T2 const& new_value, Proj&& proj = Proj())
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Required at least forward iterator.");

            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Required at least forward iterator.");

            typedef typename std::iterator_traits<FwdIter1>::value_type Type;

            return hpx::ranges::replace_copy_if(
                std::forward<ExPolicy>(policy), first, sent, dest,
                [old_value](Type const& a) -> bool { return old_value == a; },
                new_value, std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename FwdIter,
        typename T1, typename T2,
        typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            replace_copy_result<typename hpx::traits::range_iterator<Rng>::type,
                FwdIter>>::type
        tag_fallback_dispatch(hpx::ranges::replace_copy_t, ExPolicy&& policy,
            Rng&& rng, FwdIter dest, T1 const& old_value, T2 const& new_value,
            Proj&& proj = Proj())
        {
            static_assert(
                (hpx::traits::is_forward_iterator<
                    typename hpx::traits::range_iterator<Rng>::type>::value),
                "Required at least forward iterator.");

            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Required at least forward iterator.");

            typedef typename std::iterator_traits<
                typename hpx::traits::range_iterator<Rng>::type>::value_type
                Type;

            return hpx::ranges::replace_copy_if(
                std::forward<ExPolicy>(policy), hpx::util::begin(rng),
                hpx::util::end(rng), dest,
                [old_value](Type const& a) -> bool { return old_value == a; },
                new_value, std::forward<Proj>(proj));
        }
    } replace_copy{};
}}    // namespace hpx::ranges

#endif    // DOXYGEN
