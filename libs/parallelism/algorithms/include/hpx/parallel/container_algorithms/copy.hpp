//  Copyright (c) 2015-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/copy.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx { namespace ranges {
    // clang-format off

    /// Copies the elements in the range, defined by [first, last), to another
    /// range beginning at \a dest.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Iter1       The type of the begin source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent1       The type of the end source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter1.
    /// \tparam FwdIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param iter         Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param sent         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    ///
    /// The assignments in the parallel \a copy algorithm invoked with an
    /// execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a copy algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a copy algorithm returns a
    ///           \a hpx::future<ranges::copy_result<FwdIter1, FwdIter> > if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or \a parallel_task_policy and
    ///           returns \a ranges::copy_result<FwdIter1, FwdIter> otherwise.
    ///           The \a copy algorithm returns the pair of the input iterator
    ///           \a last and the output iterator to the element in the
    ///           destination range, one past the last element copied.
    template <typename ExPolicy, typename Iter1, typename Sent1,
        typename FwdIter>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        hpx::ranges::copy_result<Iter1, Iter>>::type
    copy(ExPolicy&& policy, Iter1 iter, Sent1 sent, FwdIter dest);

    /// Copies the elements in the range \a rng to another
    /// range beginning at \a dest.
    ///
    /// \note   Complexity: Performs exactly
    ///         std::distance(begin(rng), end(rng)) assignments.
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
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    ///
    /// The assignments in the parallel \a copy algorithm invoked with an
    /// execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a copy algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a copy algorithm returns a
    ///           \a hpx::future<ranges::copy_result<iterator_t<Rng>, FwdIter2>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or \a parallel_task_policy and
    ///           returns \a ranges::copy_result<iterator_t<Rng>, FwdIter2>
    ///           otherwise.
    ///           The \a copy algorithm returns the pair of the input iterator
    ///           \a last and the output iterator to the element in the
    ///           destination range, one past the last element copied.
    template <typename ExPolicy, typename Rng, typename FwdIter>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        hpx::ranges::copy_result<
            typename hpx::traits::range_traits<Rng>::iterator_type,
            FwdIter>
        >::type
    copy(ExPolicy&& policy, Rng&& rng, FwdIter dest);

    /// Copies the elements in the range [first, first + count), starting from
    /// first and proceeding to first + count - 1., to another range beginning
    /// at dest.
    ///
    /// \note   Complexity: Performs exactly \a count assignments, if
    ///         count > 0, no assignments otherwise.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    ///
    /// The assignments in the parallel \a copy_n algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a copy_n algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a copy_n algorithm returns a
    ///           \a hpx::future<ranges::copy_n_result<FwdIter1, FwdIter2> >
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a ranges::copy_n_result<FwdIter1, FwdIter2>
    ///           otherwise.
    ///           The \a copy algorithm returns the pair of the input iterator
    ///           forwarded to the first element after the last in the input
    ///           sequence and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Size,
        typename FwdIter2>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        hpx::ranges::copy_n_result<FwdIter1, FwdIter2>>::type
    copy_n(ExPolicy&& policy, FwdIter1 first, Size count, FwdIter2 dest);

    /// Copies the elements in the range, defined by [first, last) to another
    /// range beginning at \a dest. Copies only the elements for which the
    /// predicate \a f returns true. The order of the elements that are not
    /// removed is preserved.
    ///
    /// \note   Complexity: Performs not more than
    ///         std::distance(begin(rng), end(rng)) assignments,
    ///         exactly std::distance(begin(rng), end(rng)) applications
    ///         of the predicate \a f.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the begin source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent1       The type of the end source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for FwdIter1.
    /// \tparam FwdIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a copy_if requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param iter         Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param sent         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
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
    /// The assignments in the parallel \a copy_if algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a copy_if algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a copy_if algorithm returns a
    ///           \a hpx::future<ranges::copy_if_result<iterator_t<Rng>, FwdIter2>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or \a parallel_task_policy and
    ///           returns \a ranges::copy_if_result<iterator_t<Rng>, FwdIter2>
    ///           otherwise.
    ///           The \a copy_if algorithm returns the pair of the input iterator
    ///           \a last and the output iterator to the element in the
    ///           destination range, one past the last element copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Sent1,
        typename FwdIter, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        hpx::ranges::copy_if_result<
            typename hpx::traits::range_traits<Rng>::iterator_type,
            OutIter>>::type
    copy_if(ExPolicy&& policy, FwdIter1 iter, Sent1 sent, FwdIter dest, F&& f,
        Proj&& proj = Proj());

    /// Copies the elements in the range \a rng to another
    /// range beginning at \a dest. Copies only the elements for which the
    /// predicate \a f returns true. The order of the elements that are not
    /// removed is preserved.
    ///
    /// \note   Complexity: Performs not more than
    ///         std::distance(begin(rng), end(rng)) assignments,
    ///         exactly std::distance(begin(rng), end(rng)) applications
    ///         of the predicate \a f.
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
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a copy_if requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
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
    /// The assignments in the parallel \a copy_if algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a copy_if algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a copy_if algorithm returns a
    ///           \a hpx::future<ranges::copy_if_result<iterator_t<Rng>, FwdIter2>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or \a parallel_task_policy and
    ///           returns \a ranges::copy_if_result<iterator_t<Rng>, FwdIter2>
    ///           otherwise.
    ///           The \a copy_if algorithm returns the pair of the input iterator
    ///           \a last and the output iterator to the element in the
    ///           destination range, one past the last element copied.
    ///
    template <typename ExPolicy, typename Rng, typename OutIter, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        hpx::ranges::copy_if_result<
            typename hpx::traits::range_traits<Rng>::iterator_type, OutIter>
    >::type
    copy_if(
        ExPolicy&& policy, Rng&& rng, OutIter dest, F&& f, Proj&& proj = Proj());

    // clang-format on
}}    // namespace hpx::ranges

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace ranges {

    template <typename I, typename O>
    using copy_result = parallel::util::in_out_result<I, O>;

    template <typename I, typename O>
    using copy_n_result = parallel::util::in_out_result<I, O>;

    template <typename I, typename O>
    using copy_if_result = parallel::util::in_out_result<I, O>;

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::copy
    HPX_INLINE_CONSTEXPR_VARIABLE struct copy_t final
      : hpx::functional::tag<copy_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename Sent1,
            typename FwdIter,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_sentinel_for<Sent1, FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            ranges::copy_result<FwdIter1, FwdIter>>::type
        tag_invoke(hpx::ranges::copy_t, ExPolicy&& policy, FwdIter1 iter,
            Sent1 sent, FwdIter dest)
        {
            using copy_iter_t =
                hpx::parallel::v1::detail::copy_iter<FwdIter1, FwdIter>;

            return hpx::parallel::v1::detail::transfer<copy_iter_t>(
                std::forward<ExPolicy>(policy), iter, sent, dest);
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename FwdIter,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            ranges::copy_result<
                typename hpx::traits::range_traits<Rng>::iterator_type,
                FwdIter>>::type
        tag_invoke(
            hpx::ranges::copy_t, ExPolicy&& policy, Rng&& rng, FwdIter dest)
        {
            using copy_iter_t = hpx::parallel::v1::detail::copy_iter<
                typename hpx::traits::range_traits<Rng>::iterator_type,
                FwdIter>;

            return hpx::parallel::v1::detail::transfer<copy_iter_t>(
                std::forward<ExPolicy>(policy), hpx::util::begin(rng),
                hpx::util::end(rng), dest);
        }

        // clang-format off
        template <typename FwdIter1, typename Sent1, typename FwdIter,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value&&
                hpx::traits::is_sentinel_for<Sent1, FwdIter1>::value&&
                hpx::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend ranges::copy_result<FwdIter1, FwdIter> tag_invoke(
            hpx::ranges::copy_t, FwdIter1 iter, Sent1 sent, FwdIter dest)
        {
            using copy_iter_t =
                hpx::parallel::v1::detail::copy_iter<FwdIter1, FwdIter>;

            return hpx::parallel::v1::detail::transfer<copy_iter_t>(
                hpx::parallel::execution::seq, iter, sent, dest);
        }

        // clang-format off
        template <typename Rng, typename FwdIter,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value&&
                hpx::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend ranges::copy_result<
            typename hpx::traits::range_traits<Rng>::iterator_type, FwdIter>
        tag_invoke(hpx::ranges::copy_t, Rng&& rng, FwdIter dest)
        {
            using copy_iter_t = hpx::parallel::v1::detail::copy_iter<
                typename hpx::traits::range_traits<Rng>::iterator_type,
                FwdIter>;

            return hpx::parallel::v1::detail::transfer<copy_iter_t>(
                hpx::parallel::execution::seq, hpx::util::begin(rng),
                hpx::util::end(rng), dest);
        }
    } copy{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::copy_n
    HPX_INLINE_CONSTEXPR_VARIABLE struct copy_n_t final
      : hpx::functional::tag<copy_n_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename Size,
            typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value)>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            ranges::copy_n_result<FwdIter1, FwdIter2>>::type
        tag_invoke(hpx::ranges::copy_n_t, ExPolicy&& policy, FwdIter1 first,
            Size count, FwdIter2 dest)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Required at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::v1::detail::is_negative(count))
            {
                return hpx::parallel::util::detail::algorithm_result<ExPolicy,
                    ranges::copy_n_result<FwdIter1, FwdIter2>>::
                    get(ranges::copy_n_result<FwdIter1, FwdIter2>{
                        std::move(first), std::move(dest)});
            }

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            return hpx::parallel::v1::detail::copy_n<
                ranges::copy_n_result<FwdIter1, FwdIter2>>()
                .call(std::forward<ExPolicy>(policy), is_seq{}, first,
                    std::size_t(count), dest);
        }

        // clang-format off
        template <typename FwdIter1, typename Size, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value)>
        // clang-format on
        friend ranges::copy_n_result<FwdIter1, FwdIter2> tag_invoke(
            hpx::ranges::copy_n_t, FwdIter1 first, Size count, FwdIter2 dest)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Required at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::v1::detail::is_negative(count))
            {
                return ranges::copy_n_result<FwdIter1, FwdIter2>{
                    std::move(first), std::move(dest)};
            }

            return hpx::parallel::v1::detail::copy_n<
                ranges::copy_n_result<FwdIter1, FwdIter2>>()
                .call(hpx::parallel::execution::seq, std::true_type{}, first,
                    std::size_t(count), dest);
        }
    } copy_n{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::copy_if
    HPX_INLINE_CONSTEXPR_VARIABLE struct copy_if_t final
      : hpx::functional::tag<copy_if_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename Sent1,
            typename FwdIter, typename Pred,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_sentinel_for<Sent1, FwdIter1>::value &&
                hpx::parallel::traits::is_projected<Proj, FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy, Pred,
                    hpx::parallel::traits::projected<Proj, FwdIter1>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            ranges::copy_if_result<FwdIter1, FwdIter>>::type
        tag_invoke(hpx::ranges::copy_if_t, ExPolicy&& policy, FwdIter1 iter,
            Sent1 sent, FwdIter dest, Pred&& pred, Proj&& proj = Proj())
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Required at least forward iterator.");

            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Required at least forward iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            return hpx::parallel::v1::detail::copy_if<
                hpx::parallel::util::in_out_result<FwdIter1, FwdIter>>()
                .call(std::forward<ExPolicy>(policy), is_seq{}, iter, sent,
                    dest, std::forward<Pred>(pred), std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename FwdIter,
            typename Pred,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
                hpx::traits::is_iterator<FwdIter>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy, Pred,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            ranges::copy_if_result<
                typename hpx::traits::range_traits<Rng>::iterator_type,
                FwdIter>>::type
        tag_invoke(hpx::ranges::copy_if_t, ExPolicy&& policy, Rng&& rng,
            FwdIter dest, Pred&& pred, Proj&& proj = Proj())
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Required at least forward iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            return hpx::parallel::v1::detail::copy_if<
                hpx::parallel::util::in_out_result<
                    typename hpx::traits::range_traits<Rng>::iterator_type,
                    FwdIter>>()
                .call(std::forward<ExPolicy>(policy), is_seq(),
                    hpx::util::begin(rng), hpx::util::end(rng), dest,
                    std::forward<Pred>(pred), std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename FwdIter1, typename Sent1, typename FwdIter,
            typename Pred,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_sentinel_for<Sent1, FwdIter1>::value &&
                hpx::parallel::traits::is_projected<Proj, FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::parallel::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected<Proj, FwdIter1>
                >::value
            )>
        // clang-format on
        friend ranges::copy_if_result<FwdIter1, FwdIter> tag_invoke(
            hpx::ranges::copy_if_t, FwdIter1 iter, Sent1 sent, FwdIter dest,
            Pred&& pred, Proj&& proj = Proj())
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Required at least forward iterator.");

            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::copy_if<
                hpx::parallel::util::in_out_result<FwdIter1, FwdIter>>()
                .call(hpx::parallel::execution::seq, std::true_type{}, iter,
                    sent, dest, std::forward<Pred>(pred),
                    std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename Rng, typename FwdIter, typename Pred,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
                hpx::traits::is_iterator<FwdIter>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::parallel::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend ranges::copy_if_result<
            typename hpx::traits::range_traits<Rng>::iterator_type, FwdIter>
        tag_invoke(hpx::ranges::copy_if_t, Rng&& rng, FwdIter dest, Pred&& pred,
            Proj&& proj = Proj())
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::copy_if<
                hpx::parallel::util::in_out_result<
                    typename hpx::traits::range_traits<Rng>::iterator_type,
                    FwdIter>>()
                .call(hpx::parallel::execution::seq, std::true_type{},
                    hpx::util::begin(rng), hpx::util::end(rng), dest,
                    std::forward<Pred>(pred), std::forward<Proj>(proj));
        }

    } copy_if{};

}}    // namespace hpx::ranges

namespace hpx { namespace parallel { inline namespace v1 {

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename Sent1,
        typename FwdIter,
        HPX_CONCEPT_REQUIRES_(
            execution::is_execution_policy<ExPolicy>::value&&
            hpx::traits::is_iterator<FwdIter1>::value&&
            hpx::traits::is_sentinel_for<Sent1, FwdIter1>::value&&
            hpx::traits::is_iterator<FwdIter>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::copy is deprecated, use hpx::ranges::copy instead")
        typename util::detail::algorithm_result<ExPolicy,
            hpx::ranges::copy_result<FwdIter1, FwdIter>>::type
        copy(ExPolicy&& policy, FwdIter1 iter, Sent1 sent, FwdIter dest)
    {
        using copy_iter_t = detail::copy_iter<FwdIter1, FwdIter>;
        return detail::transfer<copy_iter_t>(
            std::forward<ExPolicy>(policy), iter, sent, dest);
    }

    // clang-format off
    template <typename ExPolicy, typename Rng, typename FwdIter,
        HPX_CONCEPT_REQUIRES_(
            execution::is_execution_policy<ExPolicy>::value&&
            hpx::traits::is_range<Rng>::value&&
            hpx::traits::is_iterator<FwdIter>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::copy is deprecated, use hpx::ranges::copy instead")
        typename util::detail::algorithm_result<ExPolicy,
            ranges::copy_result<
                typename hpx::traits::range_traits<Rng>::iterator_type,
                FwdIter>>::type copy(ExPolicy&& policy, Rng&& rng, FwdIter dest)
    {
        using copy_iter_t = detail::copy_iter<
            typename hpx::traits::range_traits<Rng>::iterator_type, FwdIter>;

        return detail::transfer<copy_iter_t>(std::forward<ExPolicy>(policy),
            hpx::util::begin(rng), hpx::util::end(rng), dest);
    }

    // clang-format off
    template <typename ExPolicy, typename Rng, typename OutIter, typename F,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            execution::is_execution_policy<ExPolicy>::value&&
            hpx::traits::is_range<Rng>::value&&
            traits::is_projected_range<Proj, Rng>::value&&
            hpx::traits::is_iterator<OutIter>::value&&
            traits::is_indirect_callable<ExPolicy, F,
                traits::projected_range<Proj, Rng>
            >::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::copy_if is deprecated, use "
        "hpx::ranges::copy_if instead")
        typename util::detail::algorithm_result<ExPolicy,
            ranges::copy_if_result<
                typename hpx::traits::range_traits<Rng>::iterator_type,
                OutIter>>::type copy_if(ExPolicy&& policy, Rng&& rng,
            OutIter dest, F&& f, Proj&& proj = Proj())
    {
        return copy_if(std::forward<ExPolicy>(policy), hpx::util::begin(rng),
            hpx::util::end(rng), dest, std::forward<F>(f),
            std::forward<Proj>(proj));
    }
}}}    // namespace hpx::parallel::v1

#endif    // DOXYGEN
