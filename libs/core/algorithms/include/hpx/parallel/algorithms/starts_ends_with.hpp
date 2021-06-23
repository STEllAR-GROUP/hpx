//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/starts_ends_with.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/container_algorithms/equal.hpp>
#include <hpx/parallel/container_algorithms/mismatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/invoke_projected.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // starts_with
    namespace detail {
        /// \cond NOINTERNAL
        struct starts_with : public detail::algorithm<starts_with, bool>
        {
            starts_with()
              : starts_with::algorithm("starts_with")
            {
            }

            template <typename ExPolicy, typename Iter1, typename Sent1,
                typename Iter2, typename Sent2, typename Pred, typename Proj1,
                typename Proj2>
            static bool sequential(ExPolicy, Iter1 first1, Sent1 last1,
                Iter2 first2, Sent2 last2, Pred&& pred, Proj1&& proj1,
                Proj2&& proj2)
            {
                return ranges::mismatch(std::move(first1), std::move(last1),
                           std::move(first2), last2, std::forward<Pred>(pred),
                           std::forward<Proj1>(proj1),
                           std::forward<Proj2>(proj2))
                           .in2 == last2;
            }

            template <typename ExPolicy, typename FwdIter1, typename Sent1,
                typename FwdIter2, typename Sent2, typename Pred,
                typename Proj1, typename Proj2>
            static typename util::detail::algorithm_result<ExPolicy, bool>::type
            parallel(ExPolicy&& policy, FwdIter1 first1, Sent1 last1,
                FwdIter2 first2, Sent2 last2, Pred&& pred, Proj1&& proj1,
                Proj2&& proj2)
            {
                return util::detail::algorithm_result<ExPolicy, bool>::get(
                    parallel::util::get_second_element<FwdIter1, FwdIter2>(
                        ranges::mismatch(std::move(first1), std::move(last1),
                            std::move(first2), last2, std::forward<Pred>(pred),
                            std::forward<Proj1>(proj1),
                            std::forward<Proj2>(proj2))) == last2);
            }
        };
        /// \endcond
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // starts_with
    namespace detail {
        /// \cond NOINTERNAL
        struct ends_with : public detail::algorithm<ends_with, bool>
        {
            ends_with()
              : ends_with::algorithm("ends_with")
            {
            }

            template <typename ExPolicy, typename Iter1, typename Sent1,
                typename Iter2, typename Sent2, typename Pred, typename Proj1,
                typename Proj2>
            static bool sequential(ExPolicy, Iter1 first1, Sent1 last1,
                Iter2 first2, Sent2 last2, Pred&& pred, Proj1&& proj1,
                Proj2&& proj2)
            {
                const auto drop = detail::distance(first1, last1) -
                    detail::distance(first2, last2);

                if (drop < 0)
                    return false;

                return ranges::equal(std::next(std::move(first1), drop),
                    std::move(last1), std::move(first2), std::move(last2),
                    std::forward<Pred>(pred), std::forward<Proj1>(proj1),
                    std::forward<Proj2>(proj2));
            }

            template <typename ExPolicy, typename FwdIter1, typename Sent1,
                typename FwdIter2, typename Sent2, typename Pred,
                typename Proj1, typename Proj2>
            static typename util::detail::algorithm_result<ExPolicy, bool>::type
            parallel(ExPolicy&& policy, FwdIter1 first1, Sent1 last1,
                FwdIter2 first2, Sent2 last2, Pred&& pred, Proj1&& proj1,
                Proj2&& proj2)
            {
                const auto drop = detail::distance(first1, last1) -
                    detail::distance(first2, last2);

                if (drop < 0)
                {
                    return util::detail::algorithm_result<ExPolicy, bool>::get(
                        false);
                }

                return ranges::equal(std::forward<ExPolicy>(policy),
                    std::next(std::move(first1), drop), std::move(last1),
                    std::move(first2), std::move(last2),
                    std::forward<Pred>(pred), std::forward<Proj1>(proj1),
                    std::forward<Proj2>(proj2));
            }
        };
        /// \endcond
    }    // namespace detail
}}}      // namespace hpx::parallel::v1
