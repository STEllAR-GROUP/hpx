//  Copyright (c) 2016 Minh-Khanh Do
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/segmented_algorithms/inclusive_scan.hpp

#pragma once

#include <hpx/config.hpp>

#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/inclusive_scan.hpp>
#include <hpx/parallel/algorithms/transform_inclusive_scan.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/segmented_algorithms/detail/scan.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>

#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // segmented inclusive_scan
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // adds init to each element from first to dest
        struct merge_inclusive_scan
        {
            template <typename InIter, typename OutIter, typename T,
                typename Op>
            OutIter operator()(
                InIter first, InIter last, OutIter dest, T init, Op&& op)
            {
                for (/* */; first != last; (void) ++first, ++dest)
                {
                    *dest = op(init, *first);
                }
                return dest;
            }
        };

        ///////////////////////////////////////////////////////////////////////

        // do inclusive scan returns result as vector
        template <typename Value>
        struct segmented_inclusive_scan_vector
          : public detail::algorithm<segmented_inclusive_scan_vector<Value>,
                Value>
        {
            typedef Value vector_type;

            segmented_inclusive_scan_vector()
              : segmented_inclusive_scan_vector::algorithm(
                    "segmented_inclusive_scan_vector")
            {
            }

            template <typename ExPolicy, typename InIter, typename Op>
            static vector_type sequential(
                ExPolicy&& policy, InIter first, InIter last, Op&& op)
            {
                typedef typename std::iterator_traits<InIter>::value_type
                    value_type;

                vector_type result(std::distance(first, last));

                // use first element as init value for inclusive_scan
                if (result.size() != 0)
                {
                    result[0] = *first;
                    inclusive_scan<typename vector_type::iterator>().sequential(
                        std::forward<ExPolicy>(policy), first + 1, last,
                        result.begin() + 1, std::forward<value_type>(*first),
                        std::forward<Op>(op));
                }
                return result;
            }

            template <typename ExPolicy, typename FwdIter, typename Op>
            static typename util::detail::algorithm_result<ExPolicy,
                vector_type>::type
            parallel(
                ExPolicy&& /* policy */, FwdIter first, FwdIter last, Op&& op)
            {
                typedef typename std::iterator_traits<FwdIter>::value_type
                    value_type;

                typedef util::detail::algorithm_result<ExPolicy, vector_type>
                    result;

                vector_type res(std::distance(first, last));

                // use first element as the init value for inclusive_scan
                if (res.size() != 0)
                {
                    res[0] = *first;
                }

                return result::get(dataflow(
                    [=](vector_type r) {
                        inclusive_scan<typename vector_type::iterator>()
                            .parallel(hpx::execution::par, first + 1, last,
                                r.begin() + 1, std::forward<value_type>(*first),
                                op);
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
        segmented_inclusive_scan_seq(ExPolicy&& policy, SegIter first,
            SegIter last, OutIter dest, T const& init, Op&& op, std::true_type,
            Conv&& conv)
        {
            typedef hpx::traits::segmented_iterator_traits<OutIter> traits_out;

            return segmented_scan_seq<transform_inclusive_scan<
                typename traits_out::local_raw_iterator>>(
                std::forward<ExPolicy>(policy), first, last, dest,
                std::forward<Conv>(conv), init, std::forward<Op>(op),
                std::true_type());
        }

        // sequential non-segmented OutIter implementation
        template <typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op, typename Conv>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_inclusive_scan_seq(ExPolicy&& policy, SegIter first,
            SegIter last, OutIter dest, T const& init, Op&& op, std::false_type,
            Conv&& /* conv */)
        {
            typedef std::vector<T> vector_type;

            return segmented_scan_seq_non<
                segmented_inclusive_scan_vector<vector_type>>(
                std::forward<ExPolicy>(policy), first, last, dest, init,
                std::forward<Op>(op), merge_inclusive_scan(),
                // new init value is last element from
                // segmented_inclusive_scan_vector + last init value
                [op](vector_type v, T val) { return op(v.back(), val); });
        }

        ///////////////////////////////////////////////////////////////////////
        // parallel implementation

        // parallel segmented OutIter implementation
        template <typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op, typename Conv>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_inclusive_scan_par(ExPolicy&& policy, SegIter first,
            SegIter last, OutIter dest, T const& init, Op&& op, std::true_type,
            Conv&& conv)
        {
            typedef hpx::traits::segmented_iterator_traits<OutIter> traits_out;

            return segmented_scan_par<transform_inclusive_scan<
                typename traits_out::local_raw_iterator>>(
                std::forward<ExPolicy>(policy), first, last, dest,
                std::forward<Conv>(conv), init, std::forward<Op>(op),
                std::true_type());
        }

        // parallel non-segmented OutIter implementation
        template <typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op, typename Conv>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_inclusive_scan_par(ExPolicy&& policy, SegIter first,
            SegIter last, OutIter dest, T const& init, Op&& op, std::false_type,
            Conv&&)
        {
            typedef std::vector<T> vector_type;

            return segmented_scan_par_non<
                segmented_inclusive_scan_vector<vector_type>>(
                std::forward<ExPolicy>(policy), first, last, dest, init,
                std::forward<Op>(op), merge_inclusive_scan(),
                // last T of scan is in the back
                [](vector_type v) { return v.back(); });
        }

        ///////////////////////////////////////////////////////////////////////
        // sequential remote implementation
        template <typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op, typename Conv>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_inclusive_scan(ExPolicy&& policy, SegIter first, SegIter last,
            OutIter dest, T const& init, Op&& op, std::true_type, Conv&& conv)
        {
            typedef typename hpx::traits::segmented_iterator_traits<
                OutIter>::is_segmented_iterator is_out_seg;

            // check if OutIter is segmented in the same way as SegIter
            // NOLINTNEXTLINE(bugprone-branch-clone)
            if (is_segmented_the_same(first, last, dest, is_out_seg()))
            {
                return segmented_inclusive_scan_seq(
                    std::forward<ExPolicy>(policy), first, last, dest, init,
                    std::forward<Op>(op), is_out_seg(),
                    std::forward<Conv>(conv));
            }
            else
            {
                return segmented_inclusive_scan_seq(
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
        segmented_inclusive_scan(ExPolicy&& policy, SegIter first, SegIter last,
            OutIter dest, T const& init, Op&& op, std::false_type, Conv&& conv)
        {
            typedef typename hpx::traits::segmented_iterator_traits<
                OutIter>::is_segmented_iterator is_out_seg;

            // check if OutIter is segmented in the same way as SegIter
            // NOLINTNEXTLINE(bugprone-branch-clone)
            if (is_segmented_the_same(first, last, dest, is_out_seg()))
            {
                return segmented_inclusive_scan_par(
                    std::forward<ExPolicy>(policy), first, last, dest, init,
                    std::forward<Op>(op), is_out_seg(),
                    std::forward<Conv>(conv));
            }
            else
            {
                return segmented_inclusive_scan_par(
                    std::forward<ExPolicy>(policy), first, last, dest, init,
                    std::forward<Op>(op), std::false_type(),
                    std::forward<Conv>(conv));
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename InIter, typename OutIter,
            typename T, typename Op, typename Conv>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        inclusive_scan_(ExPolicy&& policy, InIter first, InIter last,
            OutIter dest, T&& init, Op&& op, std::true_type, Conv&& conv)
        {
            typedef hpx::is_sequenced_execution_policy<ExPolicy> is_seq;

            if (first == last)
            {
                return util::detail::algorithm_result<ExPolicy, OutIter>::get(
                    std::move(dest));
            }

            return segmented_inclusive_scan(std::forward<ExPolicy>(policy),
                first, last, dest, std::forward<T>(init), std::forward<Op>(op),
                is_seq(), std::forward<Conv>(conv));
        }

        template <typename ExPolicy, typename InIter, typename OutIter,
            typename Op, typename Conv>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        inclusive_scan_(ExPolicy&& policy, InIter first, InIter last,
            OutIter dest, Op&& op, std::true_type, Conv&& conv)
        {
            typedef hpx::is_sequenced_execution_policy<ExPolicy> is_seq;
            typedef
                typename std::iterator_traits<InIter>::value_type value_type;

            if (first == last)
            {
                return util::detail::algorithm_result<ExPolicy, OutIter>::get(
                    std::move(dest));
            }

            return segmented_inclusive_scan(std::forward<ExPolicy>(policy),
                first, last, dest, value_type{}, std::forward<Op>(op), is_seq(),
                std::forward<Conv>(conv));
        }

        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename InIter, typename OutIter,
            typename T, typename Op, typename Conv>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        inclusive_scan_(ExPolicy&& policy, InIter first, InIter last,
            OutIter dest, T&& init, Op&& op, std::false_type, Conv&& conv);

        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Op, typename Conv>
        static typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
        inclusive_scan_(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
            FwdIter2 dest, Op&& op, std::false_type, Conv&& conv);

        /// \endcond
    }    // namespace detail
}}}      // namespace hpx::parallel::v1
