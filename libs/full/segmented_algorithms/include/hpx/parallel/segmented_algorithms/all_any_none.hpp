//  Copyright (c) 2017 Ajai V George
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/all_any_none.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>

#include <algorithm>
#include <exception>
#include <iterator>
#include <list>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel {
    ///////////////////////////////////////////////////////////////////////////
    // segmented_all_any_none
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename FwdIter,
            typename F, typename Proj>
        static typename util::detail::algorithm_result<ExPolicy, bool>::type
        segmented_none_of(Algo&& algo, ExPolicy const& policy, FwdIter first,
            FwdIter last, F&& f, Proj&& proj, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, bool> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);
            bool output = false;
            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    output = dispatch(traits::get_id(sit), algo, policy,
                        std::true_type(), beg, end, f, proj);
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);

                if (beg != end)
                {
                    output = dispatch(traits::get_id(sit), algo, policy,
                        std::true_type(), beg, end, f, proj);
                }

                // handle all of the full partitions
                for (++sit; sit != send && output; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);

                    if (beg != end)
                    {
                        output = dispatch(traits::get_id(sit), algo, policy,
                            std::true_type(), beg, end, f, proj);
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end && output)
                {
                    output = dispatch(traits::get_id(sit), algo, policy,
                        std::true_type(), beg, end, f, proj);
                }
            }

            return result::get(HPX_MOVE(output));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename FwdIter,
            typename F, typename Proj>
        static typename util::detail::algorithm_result<ExPolicy, bool>::type
        segmented_none_of(Algo&& algo, ExPolicy const& policy, FwdIter first,
            FwdIter last, F&& f, Proj&& proj, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, bool> result;

            typedef std::integral_constant<bool,
                !hpx::traits::is_forward_iterator<FwdIter>::value>
                forced_seq;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<shared_future<bool>> segments;
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
                [=](std::vector<shared_future<bool>>&& r) -> bool {
                    // handle any remote exceptions, will throw on error
                    std::list<std::exception_ptr> errors;
                    parallel::util::detail::handle_remote_exceptions<
                        ExPolicy>::call(r, errors);
                    std::vector<bool> res = hpx::unwrap(HPX_MOVE(r));
                    auto it = res.begin();
                    while (it != res.end())
                    {
                        if (*it == false)
                            return false;
                        it++;
                    }
                    return true;
                },
                HPX_MOVE(segments)));
        }

        ///////////////////////////////////////////////////////////////////////
        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename FwdIter,
            typename F, typename Proj>
        static typename util::detail::algorithm_result<ExPolicy, bool>::type
        segmented_any_of(Algo&& algo, ExPolicy const& policy, FwdIter first,
            FwdIter last, F&& f, Proj&& proj, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, bool> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);
            bool output = false;
            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    output = dispatch(traits::get_id(sit), algo, policy,
                        std::true_type(), beg, end, f, proj);
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);

                if (beg != end)
                {
                    output = dispatch(traits::get_id(sit), algo, policy,
                        std::true_type(), beg, end, f, proj);
                }

                // handle all of the full partitions
                for (++sit; sit != send && !output; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);

                    if (beg != end)
                    {
                        output = dispatch(traits::get_id(sit), algo, policy,
                            std::true_type(), beg, end, f, proj);
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end && !output)
                {
                    output = dispatch(traits::get_id(sit), algo, policy,
                        std::true_type(), beg, end, f, proj);
                }
            }

            return result::get(HPX_MOVE(output));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename FwdIter,
            typename F, typename Proj>
        static typename util::detail::algorithm_result<ExPolicy, bool>::type
        segmented_any_of(Algo&& algo, ExPolicy const& policy, FwdIter first,
            FwdIter last, F&& f, Proj&& proj, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, bool> result;

            typedef std::integral_constant<bool,
                !hpx::traits::is_forward_iterator<FwdIter>::value>
                forced_seq;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<shared_future<bool>> segments;
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
                [=](std::vector<shared_future<bool>>&& r) -> bool {
                    // handle any remote exceptions, will throw on error
                    std::list<std::exception_ptr> errors;
                    parallel::util::detail::handle_remote_exceptions<
                        ExPolicy>::call(r, errors);
                    std::vector<bool> res = hpx::unwrap(HPX_MOVE(r));
                    auto it = res.begin();
                    while (it != res.end())
                    {
                        if (*it == true)
                            return true;
                        it++;
                    }
                    return false;
                },
                HPX_MOVE(segments)));
        }

        ///////////////////////////////////////////////////////////////////////
        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename FwdIter,
            typename F, typename Proj>
        static typename util::detail::algorithm_result<ExPolicy, bool>::type
        segmented_all_of(Algo&& algo, ExPolicy const& policy, FwdIter first,
            FwdIter last, F&& f, Proj&& proj, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, bool> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);
            bool output = false;
            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    output = dispatch(traits::get_id(sit), algo, policy,
                        std::true_type(), beg, end, f, proj);
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);

                if (beg != end)
                {
                    output = dispatch(traits::get_id(sit), algo, policy,
                        std::true_type(), beg, end, f, proj);
                }

                // handle all of the full partitions
                for (++sit; sit != send && output; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);

                    if (beg != end)
                    {
                        output = dispatch(traits::get_id(sit), algo, policy,
                            std::true_type(), beg, end, f, proj);
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end && output)
                {
                    output = dispatch(traits::get_id(sit), algo, policy,
                        std::true_type(), beg, end, f, proj);
                }
            }

            return result::get(HPX_MOVE(output));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename FwdIter,
            typename F, typename Proj>
        static typename util::detail::algorithm_result<ExPolicy, bool>::type
        segmented_all_of(Algo&& algo, ExPolicy const& policy, FwdIter first,
            FwdIter last, F&& f, Proj&& proj, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, bool> result;

            typedef std::integral_constant<bool,
                !hpx::traits::is_forward_iterator<FwdIter>::value>
                forced_seq;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<shared_future<bool>> segments;
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
                [=](std::vector<shared_future<bool>>&& r) -> bool {
                    // handle any remote exceptions, will throw on error
                    std::list<std::exception_ptr> errors;
                    parallel::util::detail::handle_remote_exceptions<
                        ExPolicy>::call(r, errors);
                    std::vector<bool> res = hpx::unwrap(HPX_MOVE(r));
                    auto it = res.begin();
                    while (it != res.end())
                    {
                        if (*it == false)
                            return false;
                        it++;
                    }
                    return true;
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
    bool tag_invoke(hpx::none_of_t, InIter first, InIter last, F&& f)
    {
        static_assert(hpx::traits::is_input_iterator<InIter>::value,
            "Requires at least input iterator.");

        return hpx::parallel::detail::segmented_none_of(
            hpx::parallel::detail::none_of(), hpx::execution::seq, first, last,
            HPX_FORWARD(F, f), hpx::identity_v, std::true_type());
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
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy, bool>::type
    tag_invoke(
        hpx::none_of_t, ExPolicy&& policy, SegIter first, SegIter last, F&& f)
    {
        static_assert(hpx::traits::is_forward_iterator<SegIter>::value,
            "Requires at least forward iterator.");

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

        return hpx::parallel::detail::segmented_none_of(
            hpx::parallel::detail::none_of(), HPX_FORWARD(ExPolicy, policy),
            first, last, HPX_FORWARD(F, f), hpx::identity_v, is_seq());
    }

    // clang-format off
    template <typename InIter,
        typename F,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_iterator<InIter>::value &&
            hpx::traits::is_segmented_iterator<InIter>::value
        )>
    // clang-format on
    bool tag_invoke(hpx::any_of_t, InIter first, InIter last, F&& f)
    {
        static_assert(hpx::traits::is_input_iterator<InIter>::value,
            "Requires at least input iterator.");

        return hpx::parallel::detail::segmented_any_of(
            hpx::parallel::detail::any_of(), hpx::execution::seq, first, last,
            HPX_FORWARD(F, f), hpx::identity_v, std::true_type());
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
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy, bool>::type
    tag_invoke(
        hpx::any_of_t, ExPolicy&& policy, SegIter first, SegIter last, F&& f)
    {
        static_assert(hpx::traits::is_forward_iterator<SegIter>::value,
            "Requires at least forward iterator.");

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

        return hpx::parallel::detail::segmented_any_of(
            hpx::parallel::detail::any_of(), HPX_FORWARD(ExPolicy, policy),
            first, last, HPX_FORWARD(F, f), hpx::identity_v, is_seq());
    }

    // clang-format off
    template <typename InIter,
        typename F,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_iterator<InIter>::value &&
            hpx::traits::is_segmented_iterator<InIter>::value
        )>
    // clang-format on
    bool tag_invoke(hpx::all_of_t, InIter first, InIter last, F&& f)
    {
        static_assert(hpx::traits::is_input_iterator<InIter>::value,
            "Requires at least input iterator.");

        return hpx::parallel::detail::segmented_all_of(
            hpx::parallel::detail::all_of(), hpx::execution::seq, first, last,
            HPX_FORWARD(F, f), hpx::identity_v, std::true_type());
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
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy, bool>::type
    tag_invoke(
        hpx::all_of_t, ExPolicy&& policy, SegIter first, SegIter last, F&& f)
    {
        static_assert(hpx::traits::is_forward_iterator<SegIter>::value,
            "Requires at least forward iterator.");

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

        return hpx::parallel::detail::segmented_all_of(
            hpx::parallel::detail::all_of(), HPX_FORWARD(ExPolicy, policy),
            first, last, HPX_FORWARD(F, f), hpx::identity_v, is_seq());
    }
}}    // namespace hpx::segmented
