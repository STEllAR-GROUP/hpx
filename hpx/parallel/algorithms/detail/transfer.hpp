//  Copyright (c) 2016 Minh-Khanh Do
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_ALGORITHMS_TRANSFER)
#define HPX_PARALLEL_ALGORITHMS_TRANSFER

#include <hpx/traits/is_iterator.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>
#include <hpx/util/tagged_pair.hpp>

#include <hpx/parallel/segmented_algorithms/detail/transfer.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/traits/projected.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1
{
    ///////////////////////////////////////////////////////////////////////////
    // transfer
    namespace detail
    {
        // parallel version
        template <typename Algo, typename ExPolicy, typename InIter, typename OutIter>
        typename util::detail::algorithm_result<
            ExPolicy, std::pair<InIter, OutIter>
        >::type
        transfer_(ExPolicy && policy, InIter first, InIter last, OutIter dest,
            std::false_type)
        {
            typedef std::integral_constant<bool,
                    parallel::execution::is_sequenced_execution_policy<
                        ExPolicy
                    >::value ||
                   !hpx::traits::is_forward_iterator<InIter>::value ||
                   !hpx::traits::is_forward_iterator<OutIter>::value
                > is_seq;
            return Algo().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, dest);
        }

        // forward declare segmented version
        template <typename Algo, typename ExPolicy, typename InIter, typename OutIter>
        typename util::detail::algorithm_result<
            ExPolicy, std::pair<InIter, OutIter>
        >::type
        transfer_(ExPolicy && policy, InIter first, InIter last, OutIter dest,
            std::true_type);

        // Executes transfer algorithm on the elements in the range [first, last),
        // to another range beginning at \a dest.
        //
        // \note   Complexity: Performs exactly \a last - \a first transfer assignments.
        //
        //
        // \tparam Algo        The algorithm that is used to transfer the elements.
        //                     Should be hpx::parallel::detail::copy or
        //                     hpx::parallel::detail::move.
        // \tparam ExPolicy    The type of the execution policy to use (deduced).
        //                     It describes the manner in which the execution
        //                     of the algorithm may be parallelized and the manner
        //                     in which it executes the move assignments.
        // \tparam InIter      The type of the source iterators used (deduced).
        //                     This iterator type must meet the requirements of an
        //                     input iterator.
        // \tparam OutIter     The type of the iterator representing the
        //                     destination range (deduced).
        //                     This iterator type must meet the requirements of an
        //                     output iterator.
        //
        // \param policy       The execution policy to use for the scheduling of
        //                     the iterations.
        // \param first        Refers to the beginning of the sequence of elements
        //                     the algorithm will be applied to.
        // \param last         Refers to the end of the sequence of elements the
        //                     algorithm will be applied to.
        // \param dest         Refers to the beginning of the destination range.
        //
        // \returns  The \a transfer algorithm returns a \a hpx::future<OutIter> if
        //           the execution policy is of type
        //           \a sequenced_task_policy or
        //           \a parallel_task_policy and
        //           returns \a OutIter otherwise.
        //           The \a move algorithm returns the output iterator to the
        //           element in the destination range, one past the last element
        //           transfered.
        //

        template <typename Algo, typename ExPolicy, typename InIter, typename OutIter,
        HPX_CONCEPT_REQUIRES_(
            hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<InIter>::value &&
            hpx::traits::is_iterator<OutIter>::value)>
        typename util::detail::algorithm_result<
            ExPolicy, hpx::util::tagged_pair<tag::in(InIter), tag::out(OutIter)>
        >::type
        transfer(ExPolicy && policy, InIter first, InIter last, OutIter dest)
        {
            static_assert(
                (hpx::traits::is_input_iterator<InIter>::value),
                "Required at least input iterator.");
            static_assert(
                (hpx::traits::is_output_iterator<OutIter>::value ||
                    hpx::traits::is_forward_iterator<OutIter>::value),
                "Requires at least output iterator.");

            typedef hpx::traits::is_segmented_iterator<InIter> is_segmented;

            return hpx::util::make_tagged_pair<tag::in, tag::out>(
                    transfer_<Algo>(
                        std::forward<ExPolicy>(policy), first, last, dest,
                        is_segmented()));
        }
    }
}}}
#endif
