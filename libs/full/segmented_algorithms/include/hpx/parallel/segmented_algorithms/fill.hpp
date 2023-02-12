//  Copyright (c) 2016 Minh-Khanh Do
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/fill.hpp>
#include <hpx/parallel/segmented_algorithms/for_each.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>

#include <algorithm>
#include <exception>
#include <iterator>
#include <list>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel {

    ///////////////////////////////////////////////////////////////////////////
    // segmented_fill
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        template <typename T>
        struct fill_function
        {
            fill_function(T val = T())
              : value_(val)
            {
            }

            T value_;

            void operator()(T& val) const
            {
                val = value_;
            }

            template <typename Archive>
            void serialize(Archive& ar, unsigned /* version */)
            {
                // clang-format off
                ar & value_;
                // clang-format on
            }
        };
    }    // namespace detail
         /// \endcond
}}       // namespace hpx::parallel

namespace hpx { namespace segmented {

    // clang-format off
    template<typename SegIter, typename T,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_iterator<SegIter>::value &&
            hpx::traits::is_segmented_iterator<SegIter>::value
        )>
    // clang-format on
    SegIter tag_invoke(hpx::fill_t, SegIter first, SegIter last, T const& value)
    {
        static_assert(hpx::traits::is_forward_iterator<SegIter>::value,
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
            hpx::parallel::detail::fill_function<value_type>(value),
            hpx::identity_v, std::true_type{});
    }

    // clang-format off
    template <typename ExPolicy, typename SegIter, typename T,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<SegIter>::value &&
            hpx::traits::is_segmented_iterator<SegIter>::value
        )>
    // clang-format on
    static typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        SegIter>::type
    tag_invoke(hpx::fill_t, ExPolicy&& policy, SegIter first, SegIter last,
        T const& value)
    {
        static_assert(hpx::traits::is_forward_iterator<SegIter>::value,
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
            hpx::parallel::detail::fill_function<value_type>(value),
            hpx::identity_v, is_seq());
    }
}}    // namespace hpx::segmented
