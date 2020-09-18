//  Copyright (c) 2017 Ajai V George
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/adjacent_difference.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iterator>
#include <list>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // segmented_adjacent_difference
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename FwdIter1,
            typename FwdIter2, typename Op>
        static typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
        segmented_adjacent_difference(Algo&& algo, ExPolicy const& policy,
            FwdIter1 first, FwdIter1 last, FwdIter2 dest, Op&& op,
            std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter1> traits1;
            typedef hpx::traits::segmented_iterator_traits<FwdIter2> traits2;
            typedef typename traits1::segment_iterator segment_iterator1;
            typedef typename traits1::local_iterator local_iterator_type1;
            typedef typename traits2::segment_iterator segment_iterator2;
            typedef typename traits2::local_iterator local_iterator_type2;

            typedef util::detail::algorithm_result<ExPolicy, FwdIter2> result;

            FwdIter1 ending, beginning;
            FwdIter2 end_dest = dest, curr;
            std::advance(end_dest, std::distance(first, last));

            segment_iterator1 sit = traits1::segment(first);
            segment_iterator1 send = traits1::segment(last);
            segment_iterator2 sdest = traits2::segment(dest);

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type1 beg = traits1::local(first);
                local_iterator_type1 end = traits1::end(sit);
                local_iterator_type2 ldest = traits2::begin(sdest);
                if (beg != end)
                {
                    local_iterator_type2 out = dispatch(traits2::get_id(sdest),
                        algo, policy, std::true_type(), beg, end, ldest, op);
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type1 beg = traits1::local(first);
                local_iterator_type1 end = traits1::end(sit);
                local_iterator_type2 ldest = traits2::begin(sdest);
                local_iterator_type2 out = traits2::local(dest);
                if (beg != end)
                {
                    out = dispatch(traits2::get_id(sdest), algo, policy,
                        std::true_type(), beg, end, ldest, op);
                }

                // handle all of the full partitions
                for (++sit, ++sdest; sit != send; ++sit, ++sdest)
                {
                    beg = traits1::begin(sit);
                    end = traits1::end(sit);
                    ldest = traits2::begin(sdest);
                    curr = traits2::compose(sdest, ldest);
                    if (beg != end)
                    {
                        out = dispatch(traits2::get_id(sdest), algo, policy,
                            std::true_type(), beg, end, ldest, op);

                        beginning = traits1::compose(sit, beg);
                        if (beginning != last)
                        {
                            if (curr != end_dest)
                                *curr = hpx::util::invoke(
                                    op, *beginning, *std::prev(beginning));
                        }
                    }
                }

                // handle the beginning of the last partition
                beg = traits1::begin(sit);
                end = traits1::local(last);
                ldest = traits2::begin(sdest);
                curr = traits2::compose(sdest, ldest);
                if (beg != end)
                {
                    out = dispatch(traits2::get_id(sdest), algo, policy,
                        std::true_type(), beg, end, ldest, op);
                    beginning = traits1::compose(sit, beg);
                    if (beginning != last)
                    {
                        if (curr != end_dest)
                            *curr = hpx::util::invoke(
                                op, *beginning, *std::prev(beginning));
                    }
                }
                dest = traits2::compose(sdest, out);
            }
            return result::get(std::move(dest));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename FwdIter1,
            typename FwdIter2, typename Op>
        static typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
        segmented_adjacent_difference(Algo&& algo, ExPolicy const& policy,
            FwdIter1 first, FwdIter1 last, FwdIter2 dest, Op&& op,
            std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter1> traits1;
            typedef hpx::traits::segmented_iterator_traits<FwdIter2> traits2;
            typedef typename traits1::segment_iterator segment_iterator1;
            typedef typename traits1::local_iterator local_iterator_type1;
            typedef typename traits2::segment_iterator segment_iterator2;
            typedef typename traits2::local_iterator local_iterator_type2;

            typedef util::detail::algorithm_result<ExPolicy, FwdIter2> result;

            typedef std::integral_constant<bool,
                !hpx::traits::is_forward_iterator<FwdIter1>::value>
                forced_seq;

            segment_iterator1 sit = traits1::segment(first);
            segment_iterator1 send = traits1::segment(last);
            segment_iterator2 sdest = traits2::segment(dest);

            typedef std::vector<future<local_iterator_type2>> segment_type;
            segment_type segments;
            segments.reserve(std::distance(sit, send));

            std::vector<FwdIter1> between_segments;
            between_segments.reserve(std::distance(sit, send));

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type1 beg = traits1::local(first);
                local_iterator_type1 end = traits1::local(last);
                local_iterator_type2 ldest = traits2::begin(sdest);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits2::get_id(sdest),
                        algo, policy, forced_seq(), beg, end, ldest, op));
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type1 beg = traits1::local(first);
                local_iterator_type1 end = traits1::end(sit);
                local_iterator_type2 ldest = traits2::begin(sdest);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits2::get_id(sdest),
                        algo, policy, forced_seq(), beg, end, ldest, op));
                }

                // handle all of the full partitions
                for (++sit, ++sdest; sit != send; ++sit, ++sdest)
                {
                    beg = traits1::begin(sit);
                    end = traits1::end(sit);
                    ldest = traits2::begin(sdest);
                    if (beg != end)
                    {
                        between_segments.push_back(traits1::compose(sit, beg));
                        segments.push_back(
                            dispatch_async(traits2::get_id(sdest), algo, policy,
                                forced_seq(), beg, end, ldest, op));
                    }
                }

                // handle the beginning of the last partition
                beg = traits1::begin(sit);
                end = traits1::local(last);
                ldest = traits2::begin(sdest);
                if (beg != end)
                {
                    between_segments.push_back(traits1::compose(sit, beg));
                    segments.push_back(dispatch_async(traits2::get_id(sdest),
                        algo, policy, forced_seq(), beg, end, ldest, op));
                }
            }

            return result::get(dataflow(
                [=](segment_type&& r) -> FwdIter2 {
                    // handle any remote exceptions, will throw on error
                    std::list<std::exception_ptr> errors;
                    parallel::util::detail::handle_remote_exceptions<
                        ExPolicy>::call(r, errors);
                    auto ft = r.back().get();
                    auto odest = traits2::compose(sdest, ft);
                    auto start = between_segments.begin();
                    while (start != between_segments.end())
                    {
                        FwdIter2 curr = dest;
                        std::advance(curr, std::distance(first, *start));
                        *curr = hpx::util::invoke(
                            op, *(*start), *std::prev(*start));
                        start = std::next(start);
                    }
                    return odest;
                },
                std::move(segments)));
        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Op>
        typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
        adjacent_difference_(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
            FwdIter2 dest, Op&& op, std::true_type)
        {
            typedef hpx::is_sequenced_execution_policy<ExPolicy> is_seq;
            typedef util::detail::algorithm_result<ExPolicy, FwdIter2> result;

            if (first == last)
            {
                return result::get(std::move(dest));
            }

            typedef hpx::traits::segmented_iterator_traits<FwdIter2>
                iterator_traits;

            return segmented_adjacent_difference(
                adjacent_difference<typename iterator_traits::local_iterator>(),
                std::forward<ExPolicy>(policy), first, last, dest,
                std::forward<Op>(op), is_seq());
        }

        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Op>
        typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
        adjacent_difference_(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
            FwdIter2 dest, Op&& op, std::false_type);
        /// \endcond
    }    // namespace detail
}}}      // namespace hpx::parallel::v1
