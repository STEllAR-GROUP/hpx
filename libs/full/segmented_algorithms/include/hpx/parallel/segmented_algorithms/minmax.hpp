//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/minmax.hpp>
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

    template <typename T>
    using minmax_element_result = hpx::parallel::util::min_max_result<T>;

    ///////////////////////////////////////////////////////////////////////////
    // segmented_minmax
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename F, typename Proj>
        static util::detail::algorithm_result_t<ExPolicy, SegIter>
        segmented_minormax(Algo&& algo, ExPolicy const& policy, SegIter first,
            SegIter last, F&& f, Proj&& proj, std::true_type)
        {
            using traits = hpx::traits::segmented_iterator_traits<SegIter>;
            using segment_iterator = typename traits::segment_iterator;
            using local_iterator_type = typename traits::local_iterator;
            using result = util::detail::algorithm_result<ExPolicy, SegIter>;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<SegIter> positions;
            positions.reserve(std::distance(sit, send) + 1);

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    local_iterator_type out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f, proj);

                    positions.push_back(traits::compose(send, out));
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);

                if (beg != end)
                {
                    local_iterator_type out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f, proj);

                    positions.push_back(traits::compose(sit, out));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);

                    if (beg != end)
                    {
                        local_iterator_type out = dispatch(traits::get_id(sit),
                            algo, policy, std::true_type(), beg, end, f, proj);

                        positions.push_back(traits::compose(sit, out));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    local_iterator_type out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f, proj);

                    positions.push_back(traits::compose(sit, out));
                }
            }

            return result::get(Algo::sequential_minmax_element_ind(policy,
                positions.begin(), positions.size(), HPX_FORWARD(F, f),
                HPX_FORWARD(Proj, proj)));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename F, typename Proj>
        static util::detail::algorithm_result_t<ExPolicy, SegIter>
        segmented_minormax(Algo&& algo, ExPolicy const& policy, SegIter first,
            SegIter last, F&& f, Proj&& proj, std::false_type)
        {
            using traits = hpx::traits::segmented_iterator_traits<SegIter>;
            using segment_iterator = typename traits::segment_iterator;
            using local_iterator_type = typename traits::local_iterator;
            using result = util::detail::algorithm_result<ExPolicy, SegIter>;

            using forced_seq = std::integral_constant<bool,
                !hpx::traits::is_random_access_iterator_v<SegIter>>;

            using hpx::execution::non_task;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<future<SegIter>> segments;
            segments.reserve(std::distance(sit, send) + 1);

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(hpx::make_future<SegIter>(
                        dispatch_async(traits::get_id(sit), algo,
                            policy(non_task), forced_seq(), beg, end, f, proj),
                        [sit](local_iterator_type const& out) -> SegIter {
                            return traits::compose(sit, out);
                        }));
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    segments.push_back(hpx::make_future<SegIter>(
                        dispatch_async(traits::get_id(sit), algo,
                            policy(non_task), forced_seq(), beg, end, f, proj),
                        [sit](local_iterator_type const& out) -> SegIter {
                            return traits::compose(sit, out);
                        }));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        segments.push_back(hpx::make_future<SegIter>(
                            dispatch_async(traits::get_id(sit), algo,
                                policy(non_task), forced_seq(), beg, end, f,
                                proj),
                            [sit](local_iterator_type const& out) -> SegIter {
                                return traits::compose(sit, out);
                            }));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(hpx::make_future<SegIter>(
                        dispatch_async(traits::get_id(sit), algo,
                            policy(non_task), forced_seq(), beg, end, f, proj),
                        [sit](local_iterator_type const& out) -> SegIter {
                            return traits::compose(sit, out);
                        }));
                }
            }

            return result::get(hpx::dataflow(
                hpx::launch::sync,
                [=](std::vector<hpx::future<SegIter>>&& r) -> SegIter {
                    // handle any remote exceptions, will throw on error
                    std::list<std::exception_ptr> errors;
                    parallel::util::detail::handle_remote_exceptions<
                        ExPolicy>::call(r, errors);

                    std::vector<SegIter> res = hpx::unwrap(HPX_MOVE(r));
                    return Algo::sequential_minmax_element_ind(
                        policy, res.begin(), res.size(), f, proj);
                },
                HPX_MOVE(segments)));
        }
        /// \endcond
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // segmented_minmax
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename F, typename Proj>
        static util::detail::algorithm_result_t<ExPolicy,
            minmax_element_result<SegIter>>
        segmented_minmax(Algo&& algo, ExPolicy const& policy, SegIter first,
            SegIter last, F&& f, Proj&& proj, std::true_type)
        {
            using traits = hpx::traits::segmented_iterator_traits<SegIter>;
            using segment_iterator = typename traits::segment_iterator;
            using local_iterator_type = typename traits::local_iterator;
            using result_type = minmax_element_result<SegIter>;
            using local_iterator_pair_type =
                minmax_element_result<local_iterator_type>;

            using result =
                util::detail::algorithm_result<ExPolicy, result_type>;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<result_type> positions;
            positions.reserve(std::distance(sit, send) + 1);

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    local_iterator_pair_type out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f, proj);

                    positions.emplace_back(
                        result_type{traits::compose(send, out.min),
                            traits::compose(send, out.max)});
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);

                if (beg != end)
                {
                    local_iterator_pair_type out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f, proj);

                    positions.emplace_back(
                        result_type{traits::compose(sit, out.min),
                            traits::compose(sit, out.max)});
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);

                    if (beg != end)
                    {
                        local_iterator_pair_type out =
                            dispatch(traits::get_id(sit), algo, policy,
                                std::true_type(), beg, end, f, proj);

                        positions.emplace_back(
                            result_type{traits::compose(sit, out.min),
                                traits::compose(sit, out.max)});
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    local_iterator_pair_type out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f, proj);

                    positions.emplace_back(
                        result_type{traits::compose(sit, out.min),
                            traits::compose(sit, out.max)});
                }
            }

            return result::get(Algo::sequential_minmax_element_ind(policy,
                positions.begin(), positions.size(), HPX_FORWARD(F, f),
                HPX_FORWARD(Proj, proj)));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename F, typename Proj>
        static util::detail::algorithm_result_t<ExPolicy,
            minmax_element_result<SegIter>>
        segmented_minmax(Algo&& algo, ExPolicy const& policy, SegIter first,
            SegIter last, F&& f, Proj&& proj, std::false_type)
        {
            using traits = hpx::traits::segmented_iterator_traits<SegIter>;
            using segment_iterator = typename traits::segment_iterator;
            using local_iterator_type = typename traits::local_iterator;

            using forced_seq = std::integral_constant<bool,
                !hpx::traits::is_random_access_iterator_v<SegIter>>;

            using result_type = minmax_element_result<SegIter>;
            using local_iterator_pair_type =
                minmax_element_result<local_iterator_type>;

            using result =
                util::detail::algorithm_result<ExPolicy, result_type>;

            using hpx::execution::non_task;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<future<result_type>> segments;
            segments.reserve(std::distance(sit, send) + 1);

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(hpx::make_future<result_type>(
                        dispatch_async(traits::get_id(sit), algo,
                            policy(non_task), forced_seq(), beg, end, f, proj),
                        [sit](local_iterator_pair_type out) -> result_type {
                            return {traits::compose(sit, out.min),
                                traits::compose(sit, out.max)};
                        }));
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    segments.push_back(hpx::make_future<result_type>(
                        dispatch_async(traits::get_id(sit), algo,
                            policy(non_task), forced_seq(), beg, end, f, proj),
                        [sit](local_iterator_pair_type const& out)
                            -> result_type {
                            return {traits::compose(sit, out.min),
                                traits::compose(sit, out.max)};
                        }));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        segments.push_back(hpx::make_future<result_type>(
                            dispatch_async(traits::get_id(sit), algo,
                                policy(non_task), forced_seq(), beg, end, f,
                                proj),
                            [sit](local_iterator_pair_type const& out)
                                -> result_type {
                                return {traits::compose(sit, out.min),
                                    traits::compose(sit, out.max)};
                            }));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(hpx::make_future<result_type>(
                        dispatch_async(traits::get_id(sit), algo,
                            policy(non_task), forced_seq(), beg, end, f, proj),
                        [sit](local_iterator_pair_type const& out)
                            -> result_type {
                            return {traits::compose(sit, out.min),
                                traits::compose(sit, out.max)};
                        }));
                }
            }

            return result::get(hpx::dataflow(
                hpx::launch::sync,
                [=](std::vector<hpx::future<result_type>>&& r) -> result_type {
                    // handle any remote exceptions, will throw on error
                    std::list<std::exception_ptr> errors;
                    parallel::util::detail::handle_remote_exceptions<
                        ExPolicy>::call(r, errors);

                    std::vector<result_type> res = hpx::unwrap(HPX_MOVE(r));
                    return Algo::sequential_minmax_element_ind(
                        policy, res.begin(), res.size(), f, proj);
                },
                HPX_MOVE(segments)));
        }
        /// \endcond
    }    // namespace detail
}}       // namespace hpx::parallel

// The segmented iterators we support all live in namespace hpx::segmented
namespace hpx { namespace segmented {

    template <typename T>
    using minmax_element_result = hpx::parallel::util::min_max_result<T>;

    // clang-format off
    template <typename SegIter,
        typename F,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_iterator_v<SegIter> &&
            hpx::traits::is_segmented_iterator_v<SegIter>
        )>
    // clang-format on
    SegIter tag_invoke(hpx::min_element_t, SegIter first, SegIter last, F&& f)
    {
        static_assert((hpx::traits::is_forward_iterator_v<SegIter>),
            "Requires at least forward iterator.");

        if (first == last || std::next(first) == last)
        {
            return first;
        }

        using iterator_traits = hpx::traits::segmented_iterator_traits<SegIter>;

        return hpx::parallel::detail::segmented_minormax(
            hpx::parallel::detail::min_element<
                typename iterator_traits::local_iterator>(),
            hpx::execution::seq, first, last, HPX_FORWARD(F, f),
            hpx::identity_v, std::true_type{});
    }

    // clang-format off
    template <typename ExPolicy, typename SegIter,
        typename F,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_iterator_v<SegIter> &&
            hpx::traits::is_segmented_iterator_v<SegIter>
        )>
    // clang-format on
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, SegIter>
    tag_invoke(hpx::min_element_t, ExPolicy&& policy, SegIter first,
        SegIter last, F&& f)
    {
        static_assert((hpx::traits::is_forward_iterator_v<SegIter>),
            "Requires at least forward iterator.");

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

        if (first == last || std::next(first) == last)
        {
            return hpx::parallel::util::detail::algorithm_result<ExPolicy,
                SegIter>::get(HPX_MOVE(first));
        }

        using iterator_traits = hpx::traits::segmented_iterator_traits<SegIter>;

        return hpx::parallel::detail::segmented_minormax(
            hpx::parallel::detail::min_element<
                typename iterator_traits::local_iterator>(),
            HPX_FORWARD(ExPolicy, policy), first, last, HPX_FORWARD(F, f),
            hpx::identity_v, is_seq());
    }

    // clang-format off
    template <typename SegIter,
        typename F,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_iterator_v<SegIter> &&
            hpx::traits::is_segmented_iterator_v<SegIter>
        )>
    // clang-format on
    SegIter tag_invoke(hpx::max_element_t, SegIter first, SegIter last, F&& f)
    {
        static_assert((hpx::traits::is_forward_iterator_v<SegIter>),
            "Requires at least forward iterator.");

        if (first == last || std::next(first) == last)
        {
            return first;
        }

        using iterator_traits = hpx::traits::segmented_iterator_traits<SegIter>;

        return hpx::parallel::detail::segmented_minormax(
            hpx::parallel::detail::max_element<
                typename iterator_traits::local_iterator>(),
            hpx::execution::seq, first, last, HPX_FORWARD(F, f),
            hpx::identity_v, std::true_type{});
    }

    // clang-format off
    template <typename ExPolicy, typename SegIter,
        typename F,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_iterator_v<SegIter> &&
            hpx::traits::is_segmented_iterator_v<SegIter>
        )>
    // clang-format on
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, SegIter>
    tag_invoke(hpx::max_element_t, ExPolicy&& policy, SegIter first,
        SegIter last, F&& f)
    {
        static_assert((hpx::traits::is_forward_iterator_v<SegIter>),
            "Requires at least forward iterator.");

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

        if (first == last || std::next(first) == last)
        {
            return hpx::parallel::util::detail::algorithm_result<ExPolicy,
                SegIter>::get(HPX_MOVE(first));
        }

        using iterator_traits = hpx::traits::segmented_iterator_traits<SegIter>;

        return hpx::parallel::detail::segmented_minormax(
            hpx::parallel::detail::max_element<
                typename iterator_traits::local_iterator>(),
            HPX_FORWARD(ExPolicy, policy), first, last, HPX_FORWARD(F, f),
            hpx::identity_v, is_seq());
    }

    // clang-format off
    template <typename SegIter,
        typename F,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_iterator_v<SegIter> &&
            hpx::traits::is_segmented_iterator_v<SegIter>
        )>
    // clang-format on
    minmax_element_result<SegIter> tag_invoke(
        hpx::minmax_element_t, SegIter first, SegIter last, F&& f)
    {
        static_assert((hpx::traits::is_forward_iterator_v<SegIter>),
            "Requires at least forward iterator.");

        if (first == last || std::next(first) == last)
        {
            return {first, first};
        }

        using iterator_traits = hpx::traits::segmented_iterator_traits<SegIter>;

        return hpx::parallel::detail::segmented_minmax(
            hpx::parallel::detail::minmax_element<
                typename iterator_traits::local_iterator>(),
            hpx::execution::seq, first, last, HPX_FORWARD(F, f),
            hpx::identity_v, std::true_type{});
    }

    // clang-format off
    template <typename ExPolicy, typename SegIter,
        typename F,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_iterator_v<SegIter> &&
            hpx::traits::is_segmented_iterator_v<SegIter>
        )>
    // clang-format on
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy,
        minmax_element_result<SegIter>>
    tag_invoke(hpx::minmax_element_t, ExPolicy&& policy, SegIter first,
        SegIter last, F&& f)
    {
        static_assert((hpx::traits::is_forward_iterator_v<SegIter>),
            "Requires at least forward iterator.");

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;
        using result_type = minmax_element_result<SegIter>;

        if (first == last || std::next(first) == last)
        {
            result_type result = {first, first};
            return hpx::parallel::util::detail::algorithm_result<ExPolicy,
                result_type>::get(HPX_MOVE(result));
        }

        using iterator_traits = hpx::traits::segmented_iterator_traits<SegIter>;

        return hpx::parallel::detail::segmented_minmax(
            hpx::parallel::detail::minmax_element<
                typename iterator_traits::local_iterator>(),
            HPX_FORWARD(ExPolicy, policy), first, last, HPX_FORWARD(F, f),
            hpx::identity_v, is_seq());
    }
}}    // namespace hpx::segmented
