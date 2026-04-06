//  Copyright (c) 2026 Mo'men Samir
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/algorithms.hpp>
#include <hpx/modules/executors.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/type_support.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iterator>
#include <list>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::parallel { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    // segmented_is_partitioned

    ///////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL

    // sequential remote implementation

    template <typename Algo, typename ExPolicy, typename SegIter, typename Pred,
        typename Proj>
    static util::detail::algorithm_result_t<ExPolicy, bool>
    segmented_is_partitioned(Algo&& algo, ExPolicy const& policy, SegIter first,
        SegIter last, Pred&& pred, Proj&& proj, std::true_type)
    {
        using traits = hpx::traits::segmented_iterator_traits<SegIter>;
        using segment_iterator = typename traits::segment_iterator;
        using local_iterator_type = typename traits::local_iterator;
        using result = util::detail::algorithm_result<ExPolicy, bool>;
        util::invoke_projected<Pred, Proj> pred_projected{pred, proj};

        segment_iterator sit = traits::segment(first);
        segment_iterator send = traits::segment(last);

        if (sit == send)
        {
            // All elements are on the same partition.
            local_iterator_type beg = traits::local(first);
            local_iterator_type end = traits::local(last);
            if (beg == end)
                return result::get(true);

            return result::get(dispatch(traits::get_id(sit), algo, policy,
                std::true_type(), beg, end, pred, proj));
        }

        // `false_found` becomes true as soon as any segment's last element
        // fails pred.  After that, every subsequent segment's first element
        // must also fail pred (otherwise the global sequence is not
        // partitioned).
        bool false_found = false;

        // Handle the remaining part of the first partition.
        {
            local_iterator_type beg = traits::local(first);
            local_iterator_type end = traits::end(sit);

            if (beg != end)
            {
                bool r = dispatch(traits::get_id(sit), algo, policy,
                    std::true_type(), beg, end, pred, proj);
                if (!r)
                    return result::get(false);

                // If the last element of this segment fails pred
                // all subsequent elements must also fail.
                SegIter seg_last = traits::compose(sit, std::prev(end));
                if (!HPX_INVOKE(pred_projected, *seg_last))
                    false_found = true;
            }
        }

        // Handle full middle partitions.
        for (++sit; sit != send; ++sit)
        {
            local_iterator_type beg = traits::begin(sit);
            local_iterator_type end = traits::end(sit);

            if (beg != end)
            {
                // Cross-segment boundary check
                // Once a false element is found in previous segments, all
                // subsequent elements must be false.
                if (false_found)
                {
                    SegIter seg_first = traits::compose(sit, beg);
                    if (HPX_INVOKE(pred_projected, *seg_first))
                        return result::get(false);
                }

                bool r = dispatch(traits::get_id(sit), algo, policy,
                    std::true_type(), beg, end, pred, proj);
                if (!r)
                    return result::get(false);

                // if local is_partitioned returns true for this segment check
                // the last element for false
                if (!false_found)
                {
                    SegIter seg_last = traits::compose(sit, std::prev(end));
                    if (!HPX_INVOKE(pred_projected, *seg_last))
                        false_found = true;
                }
            }
        }

        // Handle the beginning of the last partition.
        {
            local_iterator_type beg = traits::begin(sit);
            local_iterator_type end = traits::local(last);

            if (beg != end)
            {
                if (false_found)
                {
                    SegIter seg_first = traits::compose(sit, beg);
                    if (HPX_INVOKE(pred_projected, *seg_first))
                        return result::get(false);
                }

                bool r = dispatch(traits::get_id(sit), algo, policy,
                    std::true_type(), beg, end, pred, proj);
                if (!r)
                    return result::get(false);
            }
        }

        return result::get(true);
    }

    // parallel remote implementation

    template <typename Algo, typename ExPolicy, typename SegIter, typename Pred,
        typename Proj>
    static util::detail::algorithm_result_t<ExPolicy, bool>
    segmented_is_partitioned(Algo&& algo, ExPolicy const& policy, SegIter first,
        SegIter last, Pred&& pred, Proj&& proj, std::false_type)
    {
        using traits = hpx::traits::segmented_iterator_traits<SegIter>;
        using segment_iterator = typename traits::segment_iterator;
        using local_iterator_type = typename traits::local_iterator;
        using result = util::detail::algorithm_result<ExPolicy, bool>;

        using forced_seq =
            std::integral_constant<bool, !std::forward_iterator<SegIter>>;

        using segment_type = std::vector<hpx::future<bool>>;

        util::invoke_projected<Pred, Proj> pred_projected{pred, proj};

        segment_iterator sit = traits::segment(first);
        segment_iterator send = traits::segment(last);

        auto const size = std::distance(sit, send);

        segment_type segments;
        segments.reserve(size + 1);

        std::vector<SegIter> between_segments;
        between_segments.reserve(size);

        if (sit == send)
        {
            // All elements on the same partition.
            local_iterator_type beg = traits::local(first);
            local_iterator_type end = traits::local(last);
            if (beg != end)
            {
                segments.push_back(dispatch_async(traits::get_id(sit), algo,
                    policy, forced_seq(), beg, end, pred, proj));
            }
        }
        else
        {
            // Handle the remaining part of the first partition.
            local_iterator_type beg = traits::local(first);
            local_iterator_type end = traits::end(sit);

            if (beg != end)
            {
                segments.push_back(dispatch_async(traits::get_id(sit), algo,
                    policy, forced_seq(), beg, end, pred, proj));
            }

            // Handle full middle partitions.
            for (++sit; sit != send; ++sit)
            {
                beg = traits::begin(sit);
                end = traits::end(sit);

                if (beg != end)
                {
                    between_segments.push_back(traits::compose(sit, beg));
                    segments.push_back(dispatch_async(traits::get_id(sit), algo,
                        policy, forced_seq(), beg, end, pred, proj));
                }
            }

            // Handle the beginning of the last partition.
            beg = traits::begin(sit);
            end = traits::local(last);
            if (beg != end)
            {
                between_segments.push_back(traits::compose(sit, beg));
                segments.push_back(dispatch_async(traits::get_id(sit), algo,
                    policy, forced_seq(), beg, end, HPX_FORWARD(Pred, pred),
                    HPX_FORWARD(Proj, proj)));
            }
        }

        return result::get(dataflow(
            [=](segment_type&& r) mutable -> bool {
                std::list<std::exception_ptr> errors;
                parallel::util::detail::handle_remote_exceptions<
                    ExPolicy>::call(r, errors);
                std::vector<bool> res = hpx::unwrap(HPX_MOVE(r));

                // Every segment must be locally is_partitioned.
                for (bool b : res)
                {
                    if (!b)
                        return false;
                }

                // Cross-segment boundary check.
                // Check the last element of each segment, If its false
                // then the first element of the next segments must be false.
                bool false_found = false;
                for (auto const& seg_first : between_segments)
                {
                    SegIter prev_last = std::prev(seg_first);

                    if (!HPX_INVOKE(pred_projected, *prev_last))
                        false_found = true;

                    if (false_found && HPX_INVOKE(pred_projected, *seg_first))
                        return false;
                }

                return true;
            },
            HPX_MOVE(segments)));
    }

    /// \endcond

}}    // namespace hpx::parallel::detail

// The segmented iterators we support all live in namespace hpx::segmented
namespace hpx::segmented {

    template <typename InIter, typename Pred>
        requires(hpx::traits::is_iterator_v<InIter> &&
            hpx::traits::is_segmented_iterator_v<InIter>)
    bool tag_invoke(
        hpx::is_partitioned_t, InIter first, InIter last, Pred&& pred)
    {
        static_assert(std::forward_iterator<InIter>,
            "Requires at least forward iterator.");

        if (first == last)
            return true;

        using iterator_traits = hpx::traits::segmented_iterator_traits<InIter>;

        return hpx::parallel::detail::segmented_is_partitioned(
            hpx::parallel::detail::is_partitioned<
                typename iterator_traits::local_iterator,
                typename iterator_traits::local_iterator>(),
            hpx::execution::seq, first, last, HPX_FORWARD(Pred, pred),
            hpx::identity_v, std::true_type());
    }

    template <typename ExPolicy, typename SegIter, typename Pred>
        requires(hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_iterator_v<SegIter> &&
            hpx::traits::is_segmented_iterator_v<SegIter>)
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, bool> tag_invoke(
        hpx::is_partitioned_t, ExPolicy&& policy, SegIter first, SegIter last,
        Pred&& pred)
    {
        static_assert(std::forward_iterator<SegIter>,
            "Requires at least forward iterator.");

        if (first == last)
        {
            using result =
                hpx::parallel::util::detail::algorithm_result<ExPolicy, bool>;
            return result::get(true);
        }

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;
        using iterator_traits = hpx::traits::segmented_iterator_traits<SegIter>;

        return hpx::parallel::detail::segmented_is_partitioned(
            hpx::parallel::detail::is_partitioned<
                typename iterator_traits::local_iterator,
                typename iterator_traits::local_iterator>(),
            HPX_FORWARD(ExPolicy, policy), first, last, HPX_FORWARD(Pred, pred),
            hpx::identity_v, is_seq());
    }

}    // namespace hpx::segmented
