//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iterator>
#include <list>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel {

    ///////////////////////////////////////////////////////////////////////////
    // segmented_for_each
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename F, typename Proj>
        static typename util::detail::algorithm_result<ExPolicy, SegIter>::type
        segmented_for_each(Algo&& algo, ExPolicy const& policy, SegIter first,
            SegIter last, F&& f, Proj&& proj, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, SegIter> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    local_iterator_type out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f, proj);
                    last = traits::compose(send, out);
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                local_iterator_type out = traits::local(last);

                if (beg != end)
                {
                    out = dispatch(traits::get_id(sit), algo, policy,
                        std::true_type(), beg, end, f, proj);
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    out = traits::begin(send);

                    if (beg != end)
                    {
                        out = dispatch(traits::get_id(sit), algo, policy,
                            std::true_type(), beg, end, f, proj);
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    out = dispatch(traits::get_id(sit), algo, policy,
                        std::true_type(), beg, end, f, proj);
                }

                last = traits::compose(send, out);
            }

            return result::get(HPX_MOVE(last));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename F, typename Proj>
        static typename util::detail::algorithm_result<ExPolicy, SegIter>::type
        segmented_for_each(Algo&& algo, ExPolicy const& policy, SegIter first,
            SegIter last, F&& f, Proj&& proj, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, SegIter> result;

            typedef std::integral_constant<bool,
                !hpx::traits::is_forward_iterator<SegIter>::value>
                forced_seq;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<future<local_iterator_type>> segments;
            segments.reserve(std::distance(sit, send));

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit), algo,
                        policy, forced_seq(), beg, end, f, proj));
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit), algo,
                        policy, forced_seq(), beg, end, f, proj));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        segments.push_back(dispatch_async(traits::get_id(sit),
                            algo, policy, forced_seq(), beg, end, f, proj));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit), algo,
                        policy, forced_seq(), beg, end, f, proj));
                }
            }

            return result::get(dataflow(
                [=](std::vector<hpx::future<local_iterator_type>>&& r)
                    -> SegIter {
                    // handle any remote exceptions, will throw on error
                    std::list<std::exception_ptr> errors;
                    parallel::util::detail::handle_remote_exceptions<
                        ExPolicy>::call(r, errors);
                    return traits::compose(send, r.back().get());
                },
                HPX_MOVE(segments)));
        }
        /// \endcond
    }    // namespace detail
}}       // namespace hpx::parallel

// The segmented iterators we support all live in namespace hpx::segmented
namespace hpx { namespace segmented {

    // clang-format off
    template <typename InIter,
        typename F,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_iterator<InIter>::value &&
            hpx::traits::is_segmented_iterator<InIter>::value
        )>
    // clang-format on
    InIter tag_invoke(hpx::for_each_t, InIter first, InIter last, F&& f)
    {
        static_assert((hpx::traits::is_forward_iterator<InIter>::value),
            "Requires at least input iterator.");

        using iterator_traits = hpx::traits::segmented_iterator_traits<InIter>;

        if (first == last)
        {
            return first;
        }

        return hpx::parallel::detail::segmented_for_each(
            hpx::parallel::detail::for_each<
                typename iterator_traits::local_iterator>(),
            hpx::execution::seq, first, last, HPX_FORWARD(F, f),
            hpx::identity_v, std::true_type());
    }

    // clang-format off
    template <typename ExPolicy, typename SegIter,
        typename F,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<SegIter>::value &&
            hpx::traits::is_segmented_iterator<SegIter>::value
        )>
    // clang-format on
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        SegIter>::type
    tag_invoke(
        hpx::for_each_t, ExPolicy&& policy, SegIter first, SegIter last, F&& f)
    {
        static_assert((hpx::traits::is_forward_iterator<SegIter>::value),
            "Requires at least forward iterator.");

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

        if (first == last)
        {
            using result =
                hpx::parallel::util::detail::algorithm_result<ExPolicy,
                    SegIter>;
            return result::get(HPX_MOVE(first));
        }

        using iterator_traits = hpx::traits::segmented_iterator_traits<SegIter>;

        return segmented_for_each(
            hpx::parallel::detail::for_each<
                typename iterator_traits::local_iterator>(),
            HPX_FORWARD(ExPolicy, policy), first, last, HPX_FORWARD(F, f),
            hpx::identity_v, is_seq());
    }

    // clang-format off
    template <typename InIter, typename Size,
        typename F,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_iterator<InIter>::value &&
            hpx::traits::is_segmented_iterator<InIter>::value
        )>
    // clang-format on
    InIter tag_invoke(hpx::for_each_n_t, InIter first, Size count, F&& f)
    {
        static_assert((hpx::traits::is_input_iterator<InIter>::value),
            "Requires at least input iterator.");

        using iterator_traits = hpx::traits::segmented_iterator_traits<InIter>;

        if (hpx::parallel::detail::is_negative(count) || count == 0)
        {
            return first;
        }

        auto last = first;
        hpx::parallel::detail::advance(last, std::size_t(count));
        return hpx::parallel::detail::segmented_for_each(
            hpx::parallel::detail::for_each<
                typename iterator_traits::local_iterator>(),
            hpx::execution::seq, first, last, HPX_FORWARD(F, f),
            hpx::identity_v, std::true_type());
    }

    // clang-format off
    template <typename ExPolicy, typename SegIter, typename Size,
        typename F,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<SegIter>::value &&
            hpx::traits::is_segmented_iterator<SegIter>::value
        )>
    // clang-format on
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        SegIter>::type
    tag_invoke(
        hpx::for_each_n_t, ExPolicy&& policy, SegIter first, Size count, F&& f)
    {
        static_assert((hpx::traits::is_forward_iterator<SegIter>::value),
            "Requires at least input iterator.");

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

        if (hpx::parallel::detail::is_negative(count) || count == 0)
        {
            using result =
                hpx::parallel::util::detail::algorithm_result<ExPolicy,
                    SegIter>;
            return result::get(HPX_MOVE(first));
        }

        using iterator_traits = hpx::traits::segmented_iterator_traits<SegIter>;

        auto last = first;
        hpx::parallel::detail::advance(last, std::size_t(count));
        return segmented_for_each(
            hpx::parallel::detail::for_each<
                typename iterator_traits::local_iterator>(),
            HPX_FORWARD(ExPolicy, policy), first, last, HPX_FORWARD(F, f),
            hpx::identity_v, is_seq());
    }
}}    // namespace hpx::segmented
