//  Copyright (c) 2026 Abhishek Bansal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/algorithms.hpp>
#include <hpx/modules/executors.hpp>
#include <hpx/parallel/segmented_algorithms/for_each.hpp>

#include <algorithm>
#include <exception>
#include <iterator>
#include <list>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel {

    ///////////////////////////////////////////////////////////////////////////
    // segmented_replace
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        template <typename T>
        struct replace_function
        {
            replace_function(T old_val = T(), T new_val = T())
              : old_value_(old_val)
              , new_value_(new_val)
            {
            }

            T old_value_;
            T new_value_;

            void operator()(T& val) const
            {
                if (val == old_value_)
                {
                    val = new_value_;
                }
            }

            template <typename Archive>
            void serialize(Archive& ar, unsigned /* version */)
            {
                // clang-format off
                ar & old_value_;
                ar & new_value_;
                // clang-format on
            }
        };

        template <typename F, typename T>
        struct replace_if_function
        {
            replace_if_function(F pred = F(), T new_val = T())
              : pred_(pred)
              , new_value_(new_val)
            {
            }

            F pred_;
            T new_value_;

            void operator()(T& val) const
            {
                if (pred_(val))
                {
                    val = new_value_;
                }
            }

            template <typename Archive>
            void serialize(Archive& ar, unsigned /* version */)
            {
                // clang-format off
                ar & pred_;
                ar & new_value_;
                // clang-format on
            }
        };
    }    // namespace detail
    /// \endcond
}}    // namespace hpx::parallel

namespace hpx { namespace segmented {

    // segmented replace
    template <typename SegIter, typename T>
        requires(hpx::traits::is_iterator_v<SegIter> &&
            hpx::traits::is_segmented_iterator_v<SegIter>)
    SegIter tag_invoke(hpx::replace_t, SegIter first, SegIter last,
        T const& old_value, T const& new_value)
    {
        static_assert(hpx::traits::is_forward_iterator_v<SegIter>,
            "Requires at least forward iterator.");

        if (first == last)
        {
            return HPX_MOVE(first);
        }

        using iterator_traits = hpx::traits::segmented_iterator_traits<SegIter>;
        using value_type = typename std::iterator_traits<SegIter>::value_type;

        return hpx::parallel::detail::segmented_for_each(
            hpx::parallel::detail::for_each<
                typename iterator_traits::local_iterator>(),
            hpx::execution::seq, first, last,
            hpx::parallel::detail::replace_function<value_type>(
                old_value, new_value),
            hpx::identity_v, std::true_type{});
    }

    template <typename ExPolicy, typename SegIter, typename T>
        requires(hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_iterator_v<SegIter> &&
            hpx::traits::is_segmented_iterator_v<SegIter>)
    static typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        SegIter>::type
    tag_invoke(hpx::replace_t, ExPolicy&& policy, SegIter first, SegIter last,
        T const& old_value, T const& new_value)
    {
        static_assert(hpx::traits::is_forward_iterator_v<SegIter>,
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
        using value_type = typename std::iterator_traits<SegIter>::value_type;

        return segmented_for_each(
            hpx::parallel::detail::for_each<
                typename iterator_traits::local_iterator>(),
            HPX_FORWARD(ExPolicy, policy), first, last,
            hpx::parallel::detail::replace_function<value_type>(
                old_value, new_value),
            hpx::identity_v, is_seq());
    }

    // segmented replace_if
    template <typename SegIter, typename Pred, typename T>
        requires(hpx::traits::is_iterator_v<SegIter> &&
            hpx::traits::is_segmented_iterator_v<SegIter>)
    SegIter tag_invoke(hpx::replace_if_t, SegIter first, SegIter last,
        Pred pred, T const& new_value)
    {
        static_assert(hpx::traits::is_forward_iterator_v<SegIter>,
            "Requires at least forward iterator.");

        if (first == last)
        {
            return HPX_MOVE(first);
        }

        using iterator_traits = hpx::traits::segmented_iterator_traits<SegIter>;
        using value_type = typename std::iterator_traits<SegIter>::value_type;

        return hpx::parallel::detail::segmented_for_each(
            hpx::parallel::detail::for_each<
                typename iterator_traits::local_iterator>(),
            hpx::execution::seq, first, last,
            hpx::parallel::detail::replace_if_function<Pred, value_type>(
                pred, new_value),
            hpx::identity_v, std::true_type{});
    }

    template <typename ExPolicy, typename SegIter, typename Pred, typename T>
        requires(hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_iterator_v<SegIter> &&
            hpx::traits::is_segmented_iterator_v<SegIter>)
    static typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        SegIter>::type
    tag_invoke(hpx::replace_if_t, ExPolicy&& policy, SegIter first,
        SegIter last, Pred pred, T const& new_value)
    {
        static_assert(hpx::traits::is_forward_iterator_v<SegIter>,
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
        using value_type = typename std::iterator_traits<SegIter>::value_type;

        return segmented_for_each(
            hpx::parallel::detail::for_each<
                typename iterator_traits::local_iterator>(),
            HPX_FORWARD(ExPolicy, policy), first, last,
            hpx::parallel::detail::replace_if_function<Pred, value_type>(
                pred, new_value),
            hpx::identity_v, is_seq());
    }
}}    // namespace hpx::segmented
