//  Copyright (c) 2014-2017 Hartmut Kaiser
//  Copyright (c)      2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/replace.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx {

    /// Replaces all elements satisfying specific criteria with \a new_value
    /// in the range [first, last).
    ///
    /// Effects: Substitutes elements referred by the iterator it in the range
    ///          [first, last) with new_value, when the following corresponding
    ///          conditions hold: *it == old_value
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam T           The type of the old and new values to replace (deduced).
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param old_value    Refers to the old value of the elements to replace.
    /// \param new_value    Refers to the new value to use as the replacement.
    ///
    /// The assignments in the parallel \a replace algorithm
    /// execute in sequential order in the calling thread.
    ///
    ///
    /// \returns  The \a replace algorithm returns a \a void.
    ///
    template <typename Initer, typename T>
    void replace(
        InIter first, InIter last, T const& old_value, T const& new_value);

    /// Replaces all elements satisfying specific criteria with \a new_value
    /// in the range [first, last).
    ///
    /// Effects: Substitutes elements referred by the iterator it in the range
    ///          [first, last) with new_value, when the following corresponding
    ///          conditions hold: *it == old_value
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam T          The type of the old and new values to replace (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param old_value    Refers to the old value of the elements to replace.
    /// \param new_value    Refers to the new value to use as the replacement.
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
    /// \returns  The \a replace algorithm returns a \a hpx::future<void> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a void otherwise.
    ///
    template <typename ExPolicy, typename FwdIter, typename T>
    typename parallel::util::detail::algorithm_result<ExPolicy, void>::type
    replace(ExPolicy&& policy, FwdIter first, FwdIter last, T const& old_value,
        T const& new_value);

    /// Replaces all elements satisfying specific criteria (for which predicate
    /// \a f returns true) with \a new_value in the range [first, last).
    ///
    /// Effects: Substitutes elements referred by the iterator it in the range
    ///          [first, last) with new_value, when the following corresponding
    ///          conditions hold: INVOKE(f, *it) != false
    ///
    /// \note   Complexity: Performs exactly \a last - \a first applications of
    ///         the predicate.
    ///
    /// \tparam Iter        The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible.
    ///                     (deduced).
    /// \tparam T           The type of the new values to replace (deduced).
    ///
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
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param new_value    Refers to the new value to use as the replacement.
    ///
    /// The assignments in the parallel \a replace_if algorithm
    /// execute in sequential order in the calling thread.
    ///
    ///
    /// \returns  The \a replace_if algorithm returns \a void.
    ///
    template <typename Iter, typename Pred, typename T>
    void replace_if(Iter first, Iter last, Pred&& pred, T const& new_value);

    /// Replaces all elements satisfying specific criteria (for which predicate
    /// \a f returns true) with \a new_value in the range [first, last).
    ///
    /// Effects: Substitutes elements referred by the iterator it in the range
    ///          [first, last) with new_value, when the following corresponding
    ///          conditions hold: INVOKE(f, *it) != false
    ///
    /// \note   Complexity: Performs exactly \a last - \a first applications of
    ///         the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible.
    ///                     (deduced).
    /// \tparam T           The type of the new values to replace (deduced).
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
    /// \returns  The \a replace_if algorithm returns a \a hpx::future<void>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy
    ///           and returns \a void otherwise.
    ///
    template <typename ExPolicy, typename FwdIter, typename Pred, typename T>
    typename parallel::util::detail::algorithm_result<ExPolicy, void>::type
    replace_if(ExPolicy&& policy, FwdIter first, FwdIter last, Pred&& pred,
        T const& new_value);

    /// Copies the all elements from the range [first, last) to another range
    /// beginning at \a dest replacing all elements satisfying a specific
    /// criteria with \a new_value.
    ///
    /// Effects: Assigns to every iterator it in the range
    ///          [result, result + (last - first)) either new_value or
    ///          *(first + (it - result)) depending on whether the following
    ///          corresponding condition holds:
    ///          *(first + (i - result)) == old_value
    ///
    /// \note   Complexity: Performs exactly \a last - \a first applications of
    ///         the predicate.
    ///

    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam T          The type of the old and new values (deduced).
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param old_value    Refers to the old value of the elements to replace.
    /// \param new_value    Refers to the new value to use as the replacement.
    ///
    /// The assignments in the parallel \a replace_copy algorithm
    /// execute in sequential order in the calling thread.
    ///
    ///
    /// \returns  The \a replace_copy algorithm returns an
    ///           \a OutIter
    ///           The \a replace_copy algorithm returns the
    ///           Iterator to the element past the last element copied.
    ///
    template <typename InIter, typename OutIter, typename T>
    OutIter replace_copy(InIter first, InIter last, OutIter dest,
        T const& old_value, T const& new_value);

    /// Copies the all elements from the range [first, last) to another range
    /// beginning at \a dest replacing all elements satisfying a specific
    /// criteria with \a new_value.
    ///
    /// Effects: Assigns to every iterator it in the range
    ///          [result, result + (last - first)) either new_value or
    ///          *(first + (it - result)) depending on whether the following
    ///          corresponding condition holds:
    ///          *(first + (i - result)) == old_value
    ///
    /// \note   Complexity: Performs exactly \a last - \a first applications of
    ///         the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam T           The type of the old and new values (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param old_value    Refers to the old value of the elements to replace.
    /// \param new_value    Refers to the new value to use as the replacement.
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
    ///           \a hpx::future<FwdIter2>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter2
    ///           otherwise.
    ///           The \a replace_copy algorithm returns the
    ///           Iterator to the element past the last element copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename T>
    typename parallel::util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    replace_copy(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
        FwdIter2 dest, T const& old_value, T const& new_value);

    /// Copies the all elements from the range [first, last) to another range
    /// beginning at \a dest replacing all elements satisfying a specific
    /// criteria with \a new_value.
    ///
    /// Effects: Assigns to every iterator it in the range
    ///          [result, result + (last - first)) either new_value or
    ///          *(first + (it - result)) depending on whether the following
    ///          corresponding condition holds:
    ///          INVOKE(f, *(first + (i - result))) != false
    ///
    /// \note   Complexity: Performs exactly \a last - \a first applications of
    ///         the predicate.
    ///
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
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
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
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
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param new_value    Refers to the new value to use as the replacement.
    ///
    /// The assignments in the parallel \a replace_copy_if algorithm
    /// execute in sequential order in the calling thread.
    ///
    ///
    /// \returns  The \a replace_copy_if algorithm returns an
    ///           \a OutIter.
    ///           The \a replace_copy_if algorithm returns
    ///           the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename InIter, typename OutIter, typename Pred, typename T>
    OutIter replace_copy_if(InIter first, InIter last, OutIter dest,
        Pred&& pred, T const& new_value);

    /// Copies the all elements from the range [first, last) to another range
    /// beginning at \a dest replacing all elements satisfying a specific
    /// criteria with \a new_value.
    ///
    /// Effects: Assigns to every iterator it in the range
    ///          [result, result + (last - first)) either new_value or
    ///          *(first + (it - result)) depending on whether the following
    ///          corresponding condition holds:
    ///          INVOKE(f, *(first + (i - result))) != false
    ///
    /// \note   Complexity: Performs exactly \a last - \a first applications of
    ///         the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///                     (deduced).
    /// \tparam T           The type of the new values to replace (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
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
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param new_value    Refers to the new value to use as the replacement.
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
    /// \returns  The \a replace_copy_if algorithm returns a
    ///           \a hpx::future<FwdIter2>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy
    ///           and returns \a FwdIter2
    ///           otherwise.
    ///           The \a replace_copy_if algorithm returns the
    ///           iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Pred, typename T>
    typename parallel::util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    replace_copy_if(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
        FwdIter2 dest, Pred&& pred, T const& new_value);

}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/type_support/unused.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // replace
    namespace detail {
        /// \cond NOINTERNAL

        // sequential replace
        template <typename InIter, typename T1, typename T2, typename Proj>
        inline InIter sequential_replace(InIter first, InIter last,
            T1 const& old_value, T2 const& new_value, Proj&& proj)
        {
            for (/* */; first != last; ++first)
            {
                if (hpx::util::invoke(proj, *first) == old_value)
                {
                    *first = new_value;
                }
            }
            return first;
        }

        template <typename Iter>
        struct replace : public detail::algorithm<replace<Iter>, Iter>
        {
            replace()
              : replace::algorithm("replace")
            {
            }

            template <typename ExPolicy, typename InIter, typename T1,
                typename T2, typename Proj>
            static InIter sequential(ExPolicy, InIter first, InIter last,
                T1 const& old_value, T2 const& new_value, Proj&& proj)
            {
                return sequential_replace(first, last, old_value, new_value,
                    std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename FwdIter, typename T1,
                typename T2, typename Proj>
            static
                typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
                parallel(ExPolicy&& policy, FwdIter first, FwdIter last,
                    T1 const& old_value, T2 const& new_value, Proj&& proj)
            {
                typedef typename std::iterator_traits<FwdIter>::value_type type;

                return for_each_n<FwdIter>().call(
                    std::forward<ExPolicy>(policy), first,
                    std::distance(first, last),
                    [old_value, new_value, proj = std::forward<Proj>(proj)](
                        type& t) -> void {
                        if (hpx::util::invoke(proj, t) == old_value)
                        {
                            t = new_value;
                        }
                    },
                    util::projection_identity());
            }
        };
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter, typename T1, typename T2,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter>::value &&
            traits::is_projected<Proj,FwdIter>::value &&
            traits::is_indirect_callable<ExPolicy,std::equal_to<T1>,
                traits::projected<Proj, FwdIter>,
                traits::projected<Proj, T1 const*>>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 7,
        "hpx::parallel::replace is deprecated, use hpx::replace "
        "instead")
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        replace(ExPolicy&& policy, FwdIter first, FwdIter last,
            T1 const& old_value, T2 const& new_value, Proj&& proj = Proj())
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
            "Required at least forward iterator.");

        return detail::replace<FwdIter>().call(std::forward<ExPolicy>(policy),
            first, last, old_value, new_value, std::forward<Proj>(proj));
    }

    ///////////////////////////////////////////////////////////////////////////
    // replace_if
    namespace detail {
        /// \cond NOINTERNAL

        // sequential replace_if
        template <typename InIter, typename Sent, typename F, typename T,
            typename Proj>
        inline InIter sequential_replace_if(
            InIter first, Sent sent, F&& f, T const& new_value, Proj&& proj)
        {
            for (/* */; first != sent; ++first)
            {
                using hpx::util::invoke;
                if (invoke(f, invoke(proj, *first)))
                {
                    *first = new_value;
                }
            }
            return first;
        }

        template <typename Iter>
        struct replace_if : public detail::algorithm<replace_if<Iter>, Iter>
        {
            replace_if()
              : replace_if::algorithm("replace_if")
            {
            }

            template <typename ExPolicy, typename InIter, typename Sent,
                typename F, typename T, typename Proj>
            static InIter sequential(ExPolicy, InIter first, Sent last, F&& f,
                T const& new_value, Proj&& proj)
            {
                return sequential_replace_if(first, last, std::forward<F>(f),
                    new_value, std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename FwdIter, typename Sent,
                typename F, typename T, typename Proj>
            static
                typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
                parallel(ExPolicy&& policy, FwdIter first, Sent last, F&& f,
                    T const& new_value, Proj&& proj)
            {
                typedef typename std::iterator_traits<FwdIter>::value_type type;

                return for_each_n<FwdIter>().call(
                    std::forward<ExPolicy>(policy), first,
                    detail::distance(first, last),
                    [new_value, f = std::forward<F>(f),
                        proj = std::forward<Proj>(proj)](type& t) -> void {
                        using hpx::util::invoke;
                        if (invoke(f, invoke(proj, t)))
                            t = new_value;
                    },
                    util::projection_identity());
            }
        };
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter, typename F, typename T,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter>::value &&
            traits::is_projected<Proj,FwdIter>::value &&
            traits::is_indirect_callable<ExPolicy, F,
                traits::projected<Proj, FwdIter>>::value)>
    // clang-format on
    HPX_DEPRECATED_V(1, 7,
        "hpx::parallel::replace_if is deprecated, use hpx::replace_if "
        "instead")
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        replace_if(ExPolicy&& policy, FwdIter first, FwdIter last, F&& f,
            T const& new_value, Proj&& proj = Proj())
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
            "Required at least forward iterator.");

        return detail::replace_if<FwdIter>().call(
            std::forward<ExPolicy>(policy), first, last, std::forward<F>(f),
            new_value, std::forward<Proj>(proj));
    }

    ///////////////////////////////////////////////////////////////////////////
    // replace_copy
    namespace detail {
        /// \cond NOINTERNAL

        // sequential replace_copy
        template <typename InIter, typename Sent, typename OutIter, typename T,
            typename Proj>
        inline util::in_out_result<InIter, OutIter> sequential_replace_copy(
            InIter first, Sent sent, OutIter dest, T const& old_value,
            T const& new_value, Proj&& proj)
        {
            for (/* */; first != sent; ++first)
            {
                if (hpx::util::invoke(proj, *first) == old_value)
                    *dest++ = new_value;
                else
                    *dest++ = *first;
            }
            return util::in_out_result<InIter, OutIter>(first, dest);
        }

        template <typename IterPair>
        struct replace_copy
          : public detail::algorithm<replace_copy<IterPair>, IterPair>
        {
            replace_copy()
              : replace_copy::algorithm("replace_copy")
            {
            }

            template <typename ExPolicy, typename InIter, typename Sent,
                typename OutIter, typename T, typename Proj>
            static util::in_out_result<InIter, OutIter> sequential(ExPolicy,
                InIter first, Sent sent, OutIter dest, T const& old_value,
                T const& new_value, Proj&& proj)
            {
                return sequential_replace_copy(first, sent, dest, old_value,
                    new_value, std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename FwdIter1, typename Sent,
                typename FwdIter2, typename T, typename Proj>
            static typename util::detail::algorithm_result<ExPolicy,
                util::in_out_result<FwdIter1, FwdIter2>>::type
            parallel(ExPolicy&& policy, FwdIter1 first, Sent sent,
                FwdIter2 dest, T const& old_value, T const& new_value,
                Proj&& proj)
            {
                typedef hpx::util::zip_iterator<FwdIter1, FwdIter2>
                    zip_iterator;
                typedef typename zip_iterator::reference reference;

                return util::detail::get_in_out_result(
                    for_each_n<zip_iterator>().call(
                        std::forward<ExPolicy>(policy),
                        hpx::util::make_zip_iterator(first, dest),
                        detail::distance(first, sent),
                        [old_value, new_value, proj = std::forward<Proj>(proj)](
                            reference t) -> void {
                            using hpx::get;
                            if (hpx::util::invoke(proj, get<0>(t)) == old_value)
                                get<1>(t) = new_value;
                            else
                                get<1>(t) = get<0>(t);    //-V573
                        },
                        util::projection_identity()));
            }
        };
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter1,
        typename FwdIter2, typename T1, typename T2,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter1>::value &&
            traits::is_projected<Proj, FwdIter1>::value &&
            traits::is_indirect_callable<ExPolicy, std::equal_to<T1>,
                traits::projected<Proj, FwdIter1>,
                traits::projected<Proj, T1 const*>>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 7,
        "hpx::parallel::replace_copy is deprecated, use "
        "hpx::replace_copy "
        "instead") typename util::detail::algorithm_result<ExPolicy,
        util::in_out_result<FwdIter1, FwdIter2>>::type
        replace_copy(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
            FwdIter2 dest, T1 const& old_value, T2 const& new_value,
            Proj&& proj = Proj())
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");
        static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");

        return detail::replace_copy<util::in_out_result<FwdIter1, FwdIter2>>()
            .call(std::forward<ExPolicy>(policy), first, last, dest, old_value,
                new_value, std::forward<Proj>(proj));
    }

    ///////////////////////////////////////////////////////////////////////////
    // replace_copy_if
    namespace detail {
        /// \cond NOINTERNAL

        // sequential replace_copy_if
        template <typename InIter, typename Sent, typename OutIter, typename F,
            typename T, typename Proj>
        inline util::in_out_result<InIter, OutIter> sequential_replace_copy_if(
            InIter first, Sent sent, OutIter dest, F&& f, T const& new_value,
            Proj&& proj)
        {
            for (/* */; first != sent; ++first)
            {
                using hpx::util::invoke;
                if (invoke(f, invoke(proj, *first)))
                    *dest++ = new_value;
                else
                    *dest++ = *first;
            }
            return util::in_out_result<InIter, OutIter>{first, dest};
        }

        template <typename IterPair>
        struct replace_copy_if
          : public detail::algorithm<replace_copy_if<IterPair>, IterPair>
        {
            replace_copy_if()
              : replace_copy_if::algorithm("replace_copy_if")
            {
            }

            template <typename ExPolicy, typename InIter, typename Sent,
                typename OutIter, typename F, typename T, typename Proj>
            static util::in_out_result<InIter, OutIter> sequential(ExPolicy,
                InIter first, Sent sent, OutIter dest, F&& f,
                T const& new_value, Proj&& proj)
            {
                return sequential_replace_copy_if(first, sent, dest,
                    std::forward<F>(f), new_value, std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename FwdIter1, typename Sent,
                typename FwdIter2, typename F, typename T, typename Proj>
            static typename util::detail::algorithm_result<ExPolicy,
                util::in_out_result<FwdIter1, FwdIter2>>::type
            parallel(ExPolicy&& policy, FwdIter1 first, Sent sent,
                FwdIter2 dest, F&& f, T const& new_value, Proj&& proj)
            {
                typedef hpx::util::zip_iterator<FwdIter1, FwdIter2>
                    zip_iterator;
                typedef typename zip_iterator::reference reference;

                return util::detail::get_in_out_result(
                    for_each_n<zip_iterator>().call(
                        std::forward<ExPolicy>(policy),
                        hpx::util::make_zip_iterator(first, dest),
                        detail::distance(first, sent),
                        [new_value, f = std::forward<F>(f),
                            proj = std::forward<Proj>(proj)](
                            reference t) -> void {
                            using hpx::get;
                            using hpx::util::invoke;
                            if (invoke(f, invoke(proj, get<0>(t))))
                                get<1>(t) = new_value;
                            else
                                get<1>(t) = get<0>(t);    //-V573
                        },
                        util::projection_identity()));
            }
        };
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename F, typename T, typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter1>::value &&
            traits::is_projected<Proj, FwdIter1>::value &&
            traits::is_indirect_callable<ExPolicy, F,
                traits::projected<Proj, FwdIter1>>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 7,
        "hpx::parallel::replace_copy_if is deprecated, use "
        "hpx::replace_copy_if "
        "instead") typename util::detail::algorithm_result<ExPolicy,
        util::in_out_result<FwdIter1, FwdIter2>>::type
        replace_copy_if(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
            FwdIter2 dest, F&& f, T const& new_value, Proj&& proj = Proj())
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");
        static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");

        return detail::replace_copy_if<
            util::in_out_result<FwdIter1, FwdIter2>>()
            .call(std::forward<ExPolicy>(policy), first, last, dest,
                std::forward<F>(f), new_value, std::forward<Proj>(proj));
    }
}}}    // namespace hpx::parallel::v1

namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::replace_if
    HPX_INLINE_CONSTEXPR_VARIABLE struct replace_if_t final
      : hpx::detail::tag_parallel_algorithm<replace_if_t>
    {
    private:
        // clang-format off
        template <typename Iter,
            typename Pred, typename T, HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<Iter>::value &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<Iter>::value_type
                >
            )>
        // clang-format on
        friend void tag_fallback_dispatch(hpx::replace_if_t, Iter first,
            Iter last, Pred&& pred, T const& new_value)
        {
            static_assert((hpx::traits::is_input_iterator<Iter>::value),
                "Required at least input iterator.");

            hpx::parallel::v1::detail::replace_if<Iter>().call(
                hpx::execution::sequenced_policy{}, first, last,
                std::forward<Pred>(pred), new_value,
                hpx::parallel::util::projection_identity());
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter,
            typename Pred,  typename T, HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter>::value &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<FwdIter>::value_type
                >
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            void>::type
        tag_fallback_dispatch(hpx::replace_if_t, ExPolicy&& policy,
            FwdIter first, FwdIter last, Pred&& pred, T const& new_value)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Required at least forward iterator.");

            return parallel::util::detail::algorithm_result<ExPolicy>::get(
                hpx::parallel::v1::detail::replace_if<FwdIter>().call(
                    std::forward<ExPolicy>(policy), first, last,
                    std::forward<Pred>(pred), new_value,
                    hpx::parallel::util::projection_identity()));
        }
    } replace_if{};

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::replace
    HPX_INLINE_CONSTEXPR_VARIABLE struct replace_t final
      : hpx::detail::tag_parallel_algorithm<replace_t>
    {
    private:
        // clang-format off
        template <typename InIter,
            typename T, HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<InIter>::value
            )>
        // clang-format on
        friend void tag_fallback_dispatch(hpx::replace_t, InIter first,
            InIter last, T const& old_value, T const& new_value)
        {
            static_assert((hpx::traits::is_input_iterator<InIter>::value),
                "Required at least input iterator.");

            typedef typename std::iterator_traits<InIter>::value_type Type;

            return hpx::replace_if(
                hpx::execution::seq, first, last,
                [old_value](Type const& a) -> bool { return old_value == a; },
                new_value);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter,
            typename T, HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            void>::type
        tag_fallback_dispatch(hpx::replace_t, ExPolicy&& policy, FwdIter first,
            FwdIter last, T const& old_value, T const& new_value)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Required at least forward iterator.");

            typedef typename std::iterator_traits<FwdIter>::value_type Type;

            return hpx::replace_if(
                std::forward<ExPolicy>(policy), first, last,
                [old_value](Type const& a) -> bool { return old_value == a; },
                new_value);
        }
    } replace{};

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::replace_copy_if
    HPX_INLINE_CONSTEXPR_VARIABLE struct replace_copy_if_t final
      : hpx::detail::tag_parallel_algorithm<replace_copy_if_t>
    {
    private:
        // clang-format off
        template <typename InIter, typename OutIter,
            typename Pred, typename T, HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<InIter>::value &&
                hpx::traits::is_iterator<OutIter>::value &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<InIter>::value_type
                >
            )>
        // clang-format on
        friend OutIter tag_fallback_dispatch(hpx::replace_copy_if_t,
            InIter first, InIter last, OutIter dest, Pred&& pred,
            T const& new_value)
        {
            static_assert((hpx::traits::is_input_iterator<InIter>::value),
                "Required at least input iterator.");

            static_assert((hpx::traits::is_output_iterator<OutIter>::value),
                "Required at least output iterator.");

            return parallel::util::get_second_element(
                hpx::parallel::v1::detail::replace_copy_if<
                    hpx::parallel::util::in_out_result<InIter, OutIter>>()
                    .call(hpx::execution::sequenced_policy{}, first, last, dest,
                        std::forward<Pred>(pred), new_value,
                        hpx::parallel::util::projection_identity()));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Pred,  typename T, HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type
                >
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter2>::type
        tag_fallback_dispatch(hpx::replace_copy_if_t, ExPolicy&& policy,
            FwdIter1 first, FwdIter1 last, FwdIter2 dest, Pred&& pred,
            T const& new_value)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Required at least forward iterator.");

            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Required at least forward iterator.");

            return parallel::util::get_second_element(
                hpx::parallel::v1::detail::replace_copy_if<
                    hpx::parallel::util::in_out_result<FwdIter1, FwdIter2>>()
                    .call(std::forward<ExPolicy>(policy), first, last, dest,
                        std::forward<Pred>(pred), new_value,
                        hpx::parallel::util::projection_identity()));
        }
    } replace_copy_if{};

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::replace_copy
    HPX_INLINE_CONSTEXPR_VARIABLE struct replace_copy_t final
      : hpx::detail::tag_parallel_algorithm<replace_copy_t>
    {
    private:
        // clang-format off
        template <typename InIter, typename OutIter,
            typename T, HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<InIter>::value &&
                hpx::traits::is_iterator<OutIter>::value
            )>
        // clang-format on
        friend OutIter tag_fallback_dispatch(hpx::replace_copy_t, InIter first,
            InIter last, OutIter dest, T const& old_value, T const& new_value)
        {
            static_assert((hpx::traits::is_input_iterator<InIter>::value),
                "Required at least input iterator.");

            static_assert((hpx::traits::is_output_iterator<OutIter>::value),
                "Required at least output iterator.");

            typedef typename std::iterator_traits<InIter>::value_type Type;

            return hpx::replace_copy_if(
                hpx::execution::seq, first, last, dest,
                [old_value](Type const& a) -> bool { return old_value == a; },
                new_value);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename T, HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter2>::type
        tag_fallback_dispatch(hpx::replace_copy_t, ExPolicy&& policy,
            FwdIter1 first, FwdIter1 last, FwdIter2 dest, T const& old_value,
            T const& new_value)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Required at least forward iterator.");

            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Required at least forward iterator.");

            typedef typename std::iterator_traits<FwdIter1>::value_type Type;

            return hpx::replace_copy_if(
                std::forward<ExPolicy>(policy), first, last, dest,
                [old_value](Type const& a) -> bool { return old_value == a; },
                new_value);
        }
    } replace_copy{};
}    // namespace hpx

#endif    // DOXYGEN
