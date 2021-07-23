//  Copyright (c) 2016 Minh-Khanh Do
//  Copyright (c) 2020 Hartmut Kaiser
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/segmented_algorithms/exclusive_scan.hpp

#pragma once

#include <hpx/config.hpp>

#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/exclusive_scan.hpp>
#include <hpx/parallel/algorithms/transform_exclusive_scan.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/segmented_algorithms/detail/scan.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>

#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // segmented exclusive_scan
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        struct merge_exclusive_scan
        {
            // adds init to each element except for the first one
            template <typename InIter, typename OutIter, typename T,
                typename Op>
            OutIter operator()(
                InIter first, InIter last, OutIter dest, T init, Op&& op)
            {
                *dest = init;
                for (++first, ++dest; first != last; (void) ++first, ++dest)
                {
                    *dest = op(init, *first);
                }
                return dest;
            }
        };

        ///////////////////////////////////////////////////////////////////////

        // do exclusive scan returns result as vector
        // first element of result vector is last T of scan
        // (can be used to transfer to the next partition)

        // first element can be use because it will be
        // overwritten by the last T of the previous partition
        template <typename Value>
        struct segmented_exclusive_scan_vector
          : public detail::algorithm<segmented_exclusive_scan_vector<Value>,
                Value>
        {
            typedef Value vector_type;

            segmented_exclusive_scan_vector()
              : segmented_exclusive_scan_vector::algorithm(
                    "segmented_exclusive_scan_vector")
            {
            }

            template <typename ExPolicy, typename InIter, typename Op>
            static vector_type sequential(
                ExPolicy&& policy, InIter first, InIter last, Op&& op)
            {
                using value_type =
                    typename std::iterator_traits<InIter>::value_type;

                vector_type result(std::distance(first, last));

                // use first element to save the last T of scan
                if (result.size() != 0)
                {
                    exclusive_scan<typename vector_type::iterator>().sequential(
                        std::forward<ExPolicy>(policy), first + 1, last,
                        result.begin() + 1, std::forward<value_type>(*first),
                        std::forward<Op>(op));
                    result[0] = op(result.back(), *(last - 1));
                }
                return result;
            }

            template <typename ExPolicy, typename FwdIter, typename Op>
            static typename util::detail::algorithm_result<ExPolicy,
                vector_type>::type
            parallel(
                ExPolicy&& /* policy */, FwdIter first, FwdIter last, Op&& op)
            {
                using value_type =
                    typename std::iterator_traits<FwdIter>::value_type;
                using result =
                    util::detail::algorithm_result<ExPolicy, vector_type>;

                vector_type res(std::distance(first, last));

                // use first element to save last T of scan
                return result::get(dataflow(
                    [=](vector_type r) {
                        exclusive_scan<typename vector_type::iterator>()
                            .parallel(hpx::execution::par, first + 1, last,
                                r.begin() + 1, std::forward<value_type>(*first),
                                op);
                        r[0] = op(r.back(), *(last - 1));
                        return r;
                    },
                    std::move(res)));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // sequential implementation

        // sequential segmented OutIter implementation
        template <typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op, typename Conv>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_exclusive_scan_seq(ExPolicy&& policy, SegIter first,
            SegIter last, OutIter dest, T const& init, Op&& op, std::true_type,
            Conv&& conv)
        {
            using traits_out = hpx::traits::segmented_iterator_traits<OutIter>;
            return segmented_scan_seq<transform_exclusive_scan<
                typename traits_out::local_raw_iterator>>(
                std::forward<ExPolicy>(policy), first, last, dest,
                std::forward<Conv>(conv), init, std::forward<Op>(op),
                std::true_type());
        }

        // sequential non segmented OutIter implementation
        template <typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op, typename Conv>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_exclusive_scan_seq(ExPolicy&& policy, SegIter first,
            SegIter last, OutIter dest, T const& init, Op&& op, std::false_type,
            Conv&& /* conv */)
        {
            using vector_type = std::vector<T>;
            return segmented_scan_seq_non<
                segmented_exclusive_scan_vector<vector_type>>(
                std::forward<ExPolicy>(policy), first, last, dest, init,
                std::forward<Op>(op), merge_exclusive_scan(),
                // new init value is first element from
                // segmented_excluisve_scan_vector + last init value
                [op](vector_type v, T val) { return op(v.front(), val); });
        }

        ///////////////////////////////////////////////////////////////////////
        // parallel implementation

        // parallel segmented OutIter implementation
        template <typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op, typename Conv>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_exclusive_scan_par(ExPolicy&& policy, SegIter first,
            SegIter last, OutIter dest, T const& init, Op&& op, std::true_type,
            Conv&& conv)
        {
            using traits_out = hpx::traits::segmented_iterator_traits<OutIter>;
            return segmented_scan_par<transform_exclusive_scan<
                typename traits_out::local_raw_iterator>>(
                std::forward<ExPolicy>(policy), first, last, dest,
                std::forward<Conv>(conv), init, std::forward<Op>(op),
                std::true_type());
        }

        // parallel non-segmented OutIter implementation
        template <typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op, typename Conv>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_exclusive_scan_par(ExPolicy&& policy, SegIter first,
            SegIter last, OutIter dest, T const& init, Op&& op, std::false_type,
            Conv&& /* conv */)
        {
            using vector_type = std::vector<T>;
            return segmented_scan_par_non<
                segmented_exclusive_scan_vector<vector_type>>(
                std::forward<ExPolicy>(policy), first, last, dest, init,
                std::forward<Op>(op), merge_exclusive_scan(),
                // last T of scan is on the front
                // see segmented_exclusive_scan_vector
                [](vector_type v) { return v.front(); });
        }

        ///////////////////////////////////////////////////////////////////////
        // sequential remote implementation
        template <typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op, typename Conv>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_exclusive_scan(ExPolicy&& policy, SegIter first, SegIter last,
            OutIter dest, T const& init, Op&& op, std::true_type, Conv&& conv)
        {
            using is_out_seg = typename hpx::traits::segmented_iterator_traits<
                OutIter>::is_segmented_iterator;

            // check if OutIter is segmented in the same way as SegIter
            // NOLINTNEXTLINE(bugprone-branch-clone)
            if (is_segmented_the_same(first, last, dest, is_out_seg()))
            {
                return segmented_exclusive_scan_seq(
                    std::forward<ExPolicy>(policy), first, last, dest, init,
                    std::forward<Op>(op), is_out_seg(),
                    std::forward<Conv>(conv));
            }
            else
            {
                return segmented_exclusive_scan_seq(
                    std::forward<ExPolicy>(policy), first, last, dest, init,
                    std::forward<Op>(op), std::false_type(),
                    std::forward<Conv>(conv));
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // parallel remote implementation
        template <typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op, typename Conv>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_exclusive_scan(ExPolicy&& policy, SegIter first, SegIter last,
            OutIter dest, T const& init, Op&& op, std::false_type, Conv&& conv)
        {
            using is_out_seg = typename hpx::traits::segmented_iterator_traits<
                OutIter>::is_segmented_iterator;

            // NOLINTNEXTLINE(bugprone-branch-clone)
            if (is_segmented_the_same(first, last, dest, is_out_seg()))
            {
                return segmented_exclusive_scan_par(
                    std::forward<ExPolicy>(policy), first, last, dest, init,
                    std::forward<Op>(op), is_out_seg(),
                    std::forward<Conv>(conv));
            }
            else
            {
                return segmented_exclusive_scan_par(
                    std::forward<ExPolicy>(policy), first, last, dest, init,
                    std::forward<Op>(op), std::false_type(),
                    std::forward<Conv>(conv));
            }
        }
        /// \endcond
    }    // namespace detail
}}}      // namespace hpx::parallel::v1

// The segmented iterators we support all live in namespace hpx::segmented
namespace hpx { namespace segmented {

    // clang-format off
    template <typename InIter, typename OutIter,
        typename T, typename Op = std::plus<T>,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_iterator_v<InIter> &&
            hpx::traits::is_segmented_iterator_v<InIter> &&
            hpx::traits::is_iterator_v<OutIter> &&
            hpx::traits::is_segmented_iterator_v<OutIter> &&
            hpx::is_invocable_v<Op,
                typename std::iterator_traits<InIter>::value_type,
                typename std::iterator_traits<InIter>::value_type
            >
        )>
    // clang-format on
    OutIter tag_dispatch(hpx::exclusive_scan_t, InIter first, InIter last,
        OutIter dest, T init, Op&& op = Op())
    {
        static_assert(hpx::traits::is_input_iterator_v<InIter>,
            "Requires at least input iterator.");

        static_assert(hpx::traits::is_output_iterator_v<OutIter>,
            "Requires at least output iterator.");

        if (first == last)
            return dest;

        return hpx::parallel::v1::detail::segmented_exclusive_scan(
            hpx::execution::seq, first, last, dest, std::move(init),
            std::forward<Op>(op), std::true_type{},
            parallel::util::projection_identity{});
    }

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename T, typename Op = std::plus<T>,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator_v<FwdIter1> &&
            hpx::traits::is_segmented_iterator_v<FwdIter1> &&
            hpx::traits::is_iterator_v<FwdIter2> &&
            hpx::traits::is_segmented_iterator_v<FwdIter2> &&
            hpx::is_invocable_v<Op,
                typename std::iterator_traits<FwdIter1>::value_type,
                typename std::iterator_traits<FwdIter1>::value_type
            >
        )>
    // clang-format on
    typename parallel::util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    tag_dispatch(hpx::exclusive_scan_t, ExPolicy&& policy, FwdIter1 first,
        FwdIter1 last, FwdIter2 dest, T init, Op&& op = Op())
    {
        static_assert(hpx::traits::is_forward_iterator_v<FwdIter1>,
            "Requires at least forward iterator.");

        static_assert(hpx::traits::is_forward_iterator_v<FwdIter2>,
            "Requires at least forward iterator.");

        if (first == last)
            return parallel::util::detail::algorithm_result<ExPolicy,
                FwdIter2>::get(std::move(dest));

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

        return hpx::parallel::v1::detail::segmented_exclusive_scan(
            std::forward<ExPolicy>(policy), first, last, dest, std::move(init),
            std::forward<Op>(op), is_seq(),
            parallel::util::projection_identity{});
    }
}}    // namespace hpx::segmented
