//  Copyright (c) 2026 Mo'men Samir
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/algorithms.hpp>
#include <hpx/modules/executors.hpp>
#include <hpx/modules/type_support.hpp>
#include <hpx/parallel/segmented_algorithms/for_each.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::parallel {

    ///////////////////////////////////////////////////////////////////////////
    // segmented_replace, segmented_replace_if
    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        template <typename T>
        struct replace_function
        {
            replace_function(T old_value = T(), T new_value = T())
              : old_value_(old_value)
              , new_value_(new_value)
            {
            }

            T old_value_;
            T new_value_;

            void operator()(T& value) const
            {
                if (value == old_value_)
                {
                    value = new_value_;
                }
            }

            template <typename Archive>
            void serialize(Archive& ar, unsigned int /* version */)
            {
                // clang-format off
                ar & old_value_ & new_value_;
                // clang-format on
            }
        };

        template <typename Pred, typename T>
        struct replace_if_function
        {
            replace_if_function(Pred pred = Pred(), T new_value = T())
              : pred_(HPX_MOVE(pred))
              , new_value_(new_value)
            {
            }

            Pred pred_;
            T new_value_;

            void operator()(T& value) const
            {
                if (HPX_INVOKE(pred_, value))
                {
                    value = new_value_;
                }
            }

            template <typename Archive>
            void serialize(Archive& ar, unsigned int /* version */)
            {
                // clang-format off
                ar & pred_ & new_value_;
                // clang-format on
            }
        };

        /// \endcond
    }    // namespace detail
}    // namespace hpx::parallel

namespace hpx::segmented {

    ///////////////////////////////////////////////////////////////////////////
    // hpx::replace

    template <typename SegIter,
        typename T = typename std::iterator_traits<SegIter>::value_type>
        requires(hpx::traits::is_iterator_v<SegIter> &&
            hpx::traits::is_segmented_iterator_v<SegIter>)
    void tag_invoke(hpx::replace_t, SegIter first, SegIter last,
        T const& old_value, T const& new_value)
    {
        static_assert(std::forward_iterator<SegIter>,
            "Requires at least forward iterator.");

        if (first == last)
        {
            return;
        }

        using iterator_traits = hpx::traits::segmented_iterator_traits<SegIter>;
        using value_type = typename std::iterator_traits<SegIter>::value_type;

        hpx::parallel::detail::segmented_for_each(
            hpx::parallel::detail::for_each<
                typename iterator_traits::local_iterator>(),
            hpx::execution::seq, first, last,
            hpx::parallel::detail::replace_function<value_type>(
                old_value, new_value),
            hpx::identity_v, std::true_type{});
    }

    template <typename ExPolicy, typename SegIter,
        typename T = typename std::iterator_traits<SegIter>::value_type>
        requires(hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_iterator_v<SegIter> &&
            hpx::traits::is_segmented_iterator_v<SegIter>)
    typename parallel::util::detail::algorithm_result<ExPolicy, void>::type
    tag_invoke(hpx::replace_t, ExPolicy&& policy, SegIter first, SegIter last,
        T const& old_value, T const& new_value)
    {
        static_assert(std::forward_iterator<SegIter>,
            "Requires at least forward iterator.");

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;
        using result =
            hpx::parallel::util::detail::algorithm_result<ExPolicy, void>;
        using result_type = typename result::type;

        if (first == last)
        {
            return result::get();
        }

        using iterator_traits = hpx::traits::segmented_iterator_traits<SegIter>;
        using value_type = typename std::iterator_traits<SegIter>::value_type;

        return hpx::util::void_guard<result_type>(),
               hpx::parallel::detail::segmented_for_each(
                   hpx::parallel::detail::for_each<
                       typename iterator_traits::local_iterator>(),
                   HPX_FORWARD(ExPolicy, policy), first, last,
                   hpx::parallel::detail::replace_function<value_type>(
                       old_value, new_value),
                   hpx::identity_v, is_seq());
    }

    ///////////////////////////////////////////////////////////////////////////
    // hpx::replace_if

    template <typename SegIter, typename Pred,
        typename T = typename std::iterator_traits<SegIter>::value_type>
        requires(hpx::traits::is_iterator_v<SegIter> &&
            hpx::traits::is_segmented_iterator_v<SegIter> &&
            hpx::is_invocable_v<Pred,
                typename std::iterator_traits<SegIter>::value_type>)
    void tag_invoke(hpx::replace_if_t, SegIter first, SegIter last, Pred pred,
        T const& new_value)
    {
        static_assert(std::forward_iterator<SegIter>,
            "Requires at least forward iterator.");

        if (first == last)
        {
            return;
        }

        using iterator_traits = hpx::traits::segmented_iterator_traits<SegIter>;
        using value_type = typename std::iterator_traits<SegIter>::value_type;

        hpx::parallel::detail::segmented_for_each(
            hpx::parallel::detail::for_each<
                typename iterator_traits::local_iterator>(),
            hpx::execution::seq, first, last,
            hpx::parallel::detail::replace_if_function<Pred, value_type>(
                HPX_MOVE(pred), new_value),
            hpx::identity_v, std::true_type{});
    }

    template <typename ExPolicy, typename SegIter, typename Pred,
        typename T = typename std::iterator_traits<SegIter>::value_type>
        requires(hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_iterator_v<SegIter> &&
            hpx::traits::is_segmented_iterator_v<SegIter> &&
            hpx::is_invocable_v<Pred,
                typename std::iterator_traits<SegIter>::value_type>)
    typename parallel::util::detail::algorithm_result<ExPolicy, void>::type
    tag_invoke(hpx::replace_if_t, ExPolicy&& policy, SegIter first,
        SegIter last, Pred pred, T const& new_value)
    {
        static_assert(std::forward_iterator<SegIter>,
            "Requires at least forward iterator.");

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;
        using result =
            hpx::parallel::util::detail::algorithm_result<ExPolicy, void>;
        using result_type = typename result::type;

        if (first == last)
        {
            return result::get();
        }

        using iterator_traits = hpx::traits::segmented_iterator_traits<SegIter>;
        using value_type = typename std::iterator_traits<SegIter>::value_type;

        return hpx::util::void_guard<result_type>(),
               hpx::parallel::detail::segmented_for_each(
                   hpx::parallel::detail::for_each<
                       typename iterator_traits::local_iterator>(),
                   HPX_FORWARD(ExPolicy, policy), first, last,
                   hpx::parallel::detail::replace_if_function<Pred, value_type>(
                       HPX_MOVE(pred), new_value),
                   hpx::identity_v, is_seq());
    }

}    // namespace hpx::segmented
