//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/rotate.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/parallel/util/tagged_pair.hpp>

#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/parallel/algorithms/rotate.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    /// Performs a left rotation on a range of elements. Specifically,
    /// \a rotate swaps the elements in the range [first, last) in such a way
    /// that the element new_first becomes the first element of the new range
    /// and new_first - 1 becomes the last element.
    ///
    /// \note   Complexity: Linear in the distance between \a first and \a last.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param middle       Refers to the element that should appear at the
    ///                     beginning of the rotated range.
    ///
    /// The assignments in the parallel \a rotate algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a rotate algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note The type of dereferenced \a FwdIter must meet the requirements
    ///       of \a MoveAssignable and \a MoveConstructible.
    ///
    /// \returns  The \a rotate algorithm returns a
    ///           \a hpx::future<tagged_pair<tag::begin(FwdIter), tag::end(FwdIter)> >
    ///           if the execution policy is of type
    ///           \a parallel_task_policy and
    ///           returns \a tagged_pair<tag::begin(FwdIter), tag::end(FwdIter)>
    ///           otherwise.
    ///           The \a rotate algorithm returns the iterator equal to
    ///           pair(first + (last - new_first), last).
    ///
    template <typename ExPolicy, typename Rng,
        HPX_CONCEPT_REQUIRES_(hpx::is_execution_policy<ExPolicy>::value&&
                hpx::traits::is_range<Rng>::value)>
    typename util::detail::algorithm_result<ExPolicy,
        util::in_out_result<typename hpx::traits::range_iterator<Rng>::type,
            typename hpx::traits::range_iterator<Rng>::type>>::type
    rotate(ExPolicy&& policy, Rng&& rng,
        typename hpx::traits::range_iterator<Rng>::type middle)
    {
        return rotate(std::forward<ExPolicy>(policy), hpx::util::begin(rng),
            middle, hpx::util::end(rng));
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Copies the elements from the range [first, last), to another range
    /// beginning at \a dest_first in such a way, that the element
    /// \a new_first becomes the first element of the new range and
    /// \a new_first - 1 becomes the last element.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param middle       Refers to the element that should appear at the
    ///                     beginning of the rotated range.
    /// \param dest_first   Refers to the begin of the destination range.
    ///
    /// The assignments in the parallel \a rotate_copy algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a rotate_copy algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a rotate_copy algorithm returns a
    ///           \a hpx::future<tagged_pair<tag::in(FwdIter), tag::out(OutIter)> >
    ///           if the execution policy is of type
    ///           \a parallel_task_policy and
    ///           returns \a tagged_pair<tag::in(FwdIter), tag::out(OutIter)>
    ///           otherwise.
    ///           The \a rotate_copy algorithm returns the output iterator to the
    ///           element past the last element copied.
    ///
    template <typename ExPolicy, typename Rng, typename OutIter,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value&& hpx::traits::is_range<
                Rng>::value&& hpx::traits::is_iterator<OutIter>::value)>
    typename util::detail::algorithm_result<ExPolicy,
        util::in_out_result<typename hpx::traits::range_iterator<Rng>::type,
            OutIter>>::type
    rotate_copy(ExPolicy&& policy, Rng&& rng,
        typename hpx::traits::range_iterator<Rng>::type middle,
        OutIter dest_first)
    {
        return rotate_copy(std::forward<ExPolicy>(policy),
            hpx::util::begin(rng), middle, hpx::util::end(rng), dest_first);
    }
}}}    // namespace hpx::parallel::v1

namespace hpx { namespace ranges {
    template <typename I, typename S = I>
    using subrange_t = hpx::util::iterator_range<I, S>;

    template <typename I, typename O>
    using rotate_copy_result = hpx::parallel::util::in_out_result<I, O>;

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::ranges::rotate
    HPX_INLINE_CONSTEXPR_VARIABLE struct rotate_t final
      : hpx::functional::tag_fallback<rotate_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Sent,
          HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator<FwdIter>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value
          )>
        // clang-format on
        friend subrange_t<FwdIter, Sent> tag_fallback_dispatch(
            hpx::ranges::rotate_t, FwdIter first, FwdIter middle, Sent last)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::rotate<FwdIter>().call(
                hpx::execution::seq, first, middle, last);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
          HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_forward_iterator<FwdIter>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value
          )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            subrange_t<FwdIter, Sent>>::type
        tag_fallback_dispatch(hpx::ranges::rotate_t, ExPolicy&& policy,
            FwdIter first, FwdIter middle, Sent last)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");
            typedef std::integral_constant<bool,
                hpx::is_sequenced_execution_policy<ExPolicy>::value ||
                    !hpx::traits::is_bidirectional_iterator<FwdIter>::value>
                is_seq;

            return parallel::util::get_second_element(
                hpx::parallel::v1::detail::rotate<
                    hpx::parallel::util::in_out_result<FwdIter, Sent>>()
                    .call2(std::forward<ExPolicy>(policy), is_seq(), first,
                        middle, last));
        }

        // clang-format off
        template <typename Rng,
            HPX_CONCEPT_REQUIRES_(hpx::traits::is_range<Rng>::value)>
        // clang-format on
        friend parallel::util::in_out_result<
            typename hpx::traits::range_iterator<Rng>::type,
            typename hpx::traits::range_iterator<Rng>::type>
        tag_fallback_dispatch(hpx::ranges::rotate_t, Rng&& rng,
            typename hpx::traits::range_iterator<Rng>::type middle)
        {
            return rotate(hpx::util::begin(rng), middle, hpx::util::end(rng));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,
            HPX_CONCEPT_REQUIRES_(hpx::is_execution_policy<ExPolicy>::value&&
                    hpx::traits::is_range<Rng>::value)>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            hpx::parallel::util::in_out_result<
                typename hpx::traits::range_iterator<Rng>::type,
                typename hpx::traits::range_iterator<Rng>::type>>::type
        tag_fallback_dispatch(hpx::ranges::rotate_t, ExPolicy&& policy,
            Rng&& rng, typename hpx::traits::range_iterator<Rng>::type middle)
        {
            return rotate(std::forward<ExPolicy>(policy), hpx::util::begin(rng),
                middle, hpx::util::end(rng));
        }
    } rotate{};

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::ranges::rotate_copy
    HPX_INLINE_CONSTEXPR_VARIABLE struct rotate_copy_t final
      : hpx::functional::tag_fallback<rotate_copy_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Sent, typename OutIter,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter>::value&& 
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value&&
                hpx::traits::is_iterator<OutIter>::value           
           )>
        // clang-format on   
        friend rotate_copy_result<FwdIter, OutIter> 
        tag_fallback_dispatch(hpx::ranges::rotate_copy_t,FwdIter first, 
            FwdIter middle, Sent last, OutIter dest_first)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_output_iterator<OutIter>::value),
                "Requires at least output iterator.");
                
            return parallel::util::get_second_element(parallel::v1::detail::rotate_copy<
                rotate_copy_result<FwdIter, OutIter>>()
                .call(hpx::execution::seq, first, middle,last, dest_first));
        }
        
        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename Sent, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value&& 
                hpx::is_execution_policy<ExPolicy>::value&&
                hpx::traits::is_sentinel_for<Sent, FwdIter1>::value&&
                hpx::traits::is_iterator<FwdIter2>::value           
           )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            rotate_copy_result<FwdIter1, FwdIter2>>::type
        tag_fallback_dispatch(hpx::ranges::rotate_copy_t, ExPolicy&& policy,
            FwdIter1 first, FwdIter1 middle, Sent last, FwdIter2 dest_first)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            typedef std::integral_constant<bool,
                hpx::is_sequenced_execution_policy<ExPolicy>::value ||
                    !hpx::traits::is_forward_iterator<FwdIter1>::value>
                is_seq;

            return parallel::util::get_second_element(
                parallel::v1::detail::rotate_copy<
                    rotate_copy_result<FwdIter1, FwdIter2>>()
                    .call2(std::forward<ExPolicy>(policy), is_seq(), first,
                        middle, last, dest_first));
        }

        // clang-format off
        template <typename Rng, typename OutIter,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value&& 
                hpx::traits::is_iterator<OutIter>::value
            )>
        // clang-format on 
        friend rotate_copy_result<typename hpx::traits::range_iterator<Rng>::type, 
            OutIter>
        tag_fallback_dispatch(hpx::ranges::rotate_copy_t, Rng&& rng,
            typename hpx::traits::range_iterator<Rng>::type middle,
                OutIter dest_first)
        {
            return rotate_copy(hpx::util::begin(rng), middle, hpx::util::end(rng), 
                dest_first);     
        }   
    
        // clang-format off
        template <typename ExPolicy, typename Rng, typename OutIter,
            HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value&& hpx::traits::is_range<
                Rng>::value&& hpx::traits::is_iterator<OutIter>::value)>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            rotate_copy_result<typename hpx::traits::range_iterator<Rng>::type,
                OutIter>>::type
        tag_fallback_dispatch(hpx::ranges::rotate_copy_t, ExPolicy&& policy,
            Rng&& rng, typename hpx::traits::range_iterator<Rng>::type middle,
            OutIter dest_first)
        {
            return rotate_copy(std::forward<ExPolicy>(policy),
                hpx::util::begin(rng), middle, hpx::util::end(rng), dest_first);
        }
    } rotate_copy{};

}}    // namespace hpx::ranges
