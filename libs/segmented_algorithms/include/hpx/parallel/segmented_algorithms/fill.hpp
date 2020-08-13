//  Copyright (c) 2016 Minh-Khanh Do
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/segmented_algorithms/for_each.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>

#include <algorithm>
#include <exception>
#include <iterator>
#include <list>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 {
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
            void serialize(Archive& ar, unsigned version)
            {
                ar& value_;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename InIter, typename T>
        static typename util::detail::algorithm_result<ExPolicy, InIter>::type
        fill_(ExPolicy&& policy, InIter first, InIter last, T const& value,
            std::true_type)
        {
            typedef typename util::detail::algorithm_result<ExPolicy>::type
                result_type;
            typedef
                typename std::iterator_traits<InIter>::value_type value_type;

            return for_each_(std::forward<ExPolicy>(policy), first, last,
                fill_function<value_type>(value), util::projection_identity(),
                std::true_type());
        }

        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename InIter, typename Sent, typename T>
        static typename util::detail::algorithm_result<ExPolicy, InIter>::type
        fill_(ExPolicy&& policy, InIter first, Sent last, T const& value,
            std::false_type);
    }    // namespace detail
         /// \endcond
}}}      // namespace hpx::parallel::v1
