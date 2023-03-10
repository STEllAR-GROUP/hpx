//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_distributed/dataflow.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <algorithm>
#include <exception>
#include <iterator>
#include <list>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel {
    ///////////////////////////////////////////////////////////////////////////
    // segmented transfer
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename SegOutIter>
        static typename util::detail::algorithm_result<ExPolicy,
            util::in_out_result<SegIter, SegOutIter>>::type
        segmented_transfer(Algo&& algo, ExPolicy const& policy, std::true_type,
            SegIter first, SegIter last, SegOutIter dest)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;

            typedef hpx::traits::segmented_iterator_traits<SegOutIter>
                output_traits;
            typedef typename output_traits::segment_iterator
                segment_output_iterator;
            typedef typename output_traits::local_iterator
                local_output_iterator_type;

            typedef util::in_out_result<local_iterator_type,
                local_output_iterator_type>
                local_iterator_pair;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            segment_output_iterator sdest = traits::segment(dest);

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);

                if (beg != end)
                {
                    local_iterator_pair p =
                        dispatch(traits::get_id(sit), algo, policy,
                            std::true_type(), beg, end, traits::local(dest));

                    dest = output_traits::compose(sdest, p.out);
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                local_output_iterator_type out = traits::local(dest);

                if (beg != end)
                {
                    local_iterator_pair p = dispatch(traits::get_id(sit), algo,
                        policy, std::true_type(), beg, end, out);
                    out = p.out;
                }

                // handle all of the full partitions
                for ((void) ++sit, ++sdest; sit != send; (void) ++sit, ++sdest)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    out = traits::begin(sdest);

                    if (beg != end)
                    {
                        local_iterator_pair p = dispatch(traits::get_id(sit),
                            algo, policy, std::true_type(), beg, end, out);
                        out = p.out;
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    local_iterator_pair p =
                        dispatch(traits::get_id(sit), algo, policy,
                            std::true_type(), beg, end, traits::begin(sdest));
                    out = p.out;
                }

                dest = output_traits::compose(sdest, out);
            }

            using result_type = util::in_out_result<SegIter, SegOutIter>;

            return util::detail::algorithm_result<ExPolicy, result_type>::get(
                result_type{last, dest});
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename SegOutIter>
        static typename util::detail::algorithm_result<ExPolicy,
            util::in_out_result<SegIter, SegOutIter>>::type
        segmented_transfer(Algo&& algo, ExPolicy const& policy, std::false_type,
            SegIter first, SegIter last, SegOutIter dest)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;

            typedef hpx::traits::segmented_iterator_traits<SegOutIter>
                output_traits;
            typedef typename output_traits::segment_iterator
                segment_output_iterator;
            typedef typename output_traits::local_iterator
                local_output_iterator_type;

            typedef util::in_out_result<local_iterator_type,
                local_output_iterator_type>
                local_iterator_pair;

            typedef std::integral_constant<bool,
                !hpx::traits::is_forward_iterator<SegIter>::value>
                forced_seq;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            segment_output_iterator sdest = traits::segment(dest);

            std::vector<shared_future<local_iterator_pair>> segments;
            segments.reserve(std::distance(sit, send));

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);

                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit), algo,
                        policy, forced_seq(), beg, end, traits::local(dest)));
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                local_output_iterator_type out = traits::local(dest);

                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit), algo,
                        policy, forced_seq(), beg, end, out));
                }

                // handle all of the full partitions
                for ((void) ++sit, ++sdest; sit != send; (void) ++sit, ++sdest)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    out = traits::begin(sdest);

                    if (beg != end)
                    {
                        segments.push_back(dispatch_async(traits::get_id(sit),
                            algo, policy, forced_seq(), beg, end, out));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);

                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit), algo,
                        policy, forced_seq(), beg, end, traits::begin(sdest)));
                }
            }

            // NOLINTNEXTLINE(bugprone-use-after-move)
            HPX_ASSERT(!segments.empty());

            return util::detail::algorithm_result<ExPolicy,
                util::in_out_result<SegIter, SegOutIter>>::
                get(hpx::dataflow(
                    [=](std::vector<shared_future<local_iterator_pair>>&& r)
                        -> util::in_out_result<SegIter, SegOutIter> {
                        // handle any remote exceptions, will throw on error
                        std::list<std::exception_ptr> errors;
                        parallel::util::detail::handle_remote_exceptions<
                            ExPolicy>::call(r, errors);

                        local_iterator_pair p = r.back().get();
                        using result_type =
                            util::in_out_result<SegIter, SegOutIter>;

                        return result_type{output_traits::compose(sdest, p.in),
                            output_traits::compose(sdest, p.out)};
                    },
                    HPX_MOVE(segments)));
        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename Algo, typename ExPolicy, typename FwdIter1,
            typename Sent1, typename FwdIter2>
        typename util::detail::algorithm_result<ExPolicy,
            util::in_out_result<FwdIter1, FwdIter2>>::type
        transfer_(ExPolicy&& policy, FwdIter1 first, Sent1 last, FwdIter2 dest,
            std::true_type)
        {
            if (first == last)
            {
                using result_type = util::in_out_result<FwdIter1, FwdIter2>;

                return util::detail::algorithm_result<ExPolicy,
                    result_type>::get(result_type{last, dest});
            }

            typedef hpx::is_sequenced_execution_policy<ExPolicy> is_seq;
            return segmented_transfer(Algo(), HPX_FORWARD(ExPolicy, policy),
                is_seq(), first, last, dest);
        }

        // forward declare the non-segmented version of this algorithm
        template <typename Algo, typename ExPolicy, typename FwdIter1,
            typename Sent1, typename FwdIter2>
        typename util::detail::algorithm_result<ExPolicy,
            util::in_out_result<FwdIter1, FwdIter2>>::type
        transfer_(ExPolicy&& policy, FwdIter1 first, Sent1 last, FwdIter2 dest,
            std::false_type);

        /// \endcond
    }    // namespace detail
}}       // namespace hpx::parallel
