//  Copyright (c) 2015 Daniel Bourgeois
//  Copyright (c) 2017-2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/is_sorted.hpp
/// \page hpx::is_sorted, hpx::is_sorted_until
/// \headerfile hpx/algorithm.hpp

#pragma once

#include <hpx/config.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/execution/algorithms/detail/is_negative.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/algorithms/detail/is_sorted.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/invoke_projected.hpp>

#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx::parallel {

    ////////////////////////////////////////////////////////////////////////////
    // is_sorted
    namespace detail {

        /// \cond NOINTERNAL
        HPX_CXX_CORE_EXPORT template <typename FwdIter, typename Sent>
        struct is_sorted : public algorithm<is_sorted<FwdIter, Sent>, bool>
        {
            constexpr is_sorted() noexcept
              : algorithm<is_sorted, bool>("is_sorted")
            {
            }

            template <typename ExPolicy, typename FwdIter_, typename Sent_,
                typename Pred, typename Proj>
            static constexpr bool sequential(
                ExPolicy, FwdIter_ first, Sent_ last, Pred&& pred, Proj&& proj)
            {
                return is_sorted_sequential(first, last,
                    HPX_FORWARD(Pred, pred), HPX_FORWARD(Proj, proj));
            }

            template <typename ExPolicy, typename FwdIter_, typename Sent_,
                typename Pred, typename Proj>
            static decltype(auto) parallel(ExPolicy&& policy, FwdIter_ first,
                Sent_ last, Pred&& pred, Proj&& proj)
            {
                using difference_type =
                    hpx::traits::iter_difference_t<FwdIter_>;
                using result =
                    typename util::detail::algorithm_result<ExPolicy, bool>;
                constexpr bool has_scheduler_executor =
                    hpx::execution_policy_has_scheduler_executor_v<ExPolicy>;

                difference_type count = detail::distance(first, last);

                if constexpr (!has_scheduler_executor)
                {
                    if (count <= 1)
                        return result::get(true);
                }

                util::invoke_projected<Pred, Proj> pred_projected{
                    HPX_FORWARD(Pred, pred), HPX_FORWARD(Proj, proj)};
                hpx::parallel::util::cancellation_token<> tok;
                using intermediate_result_t = std::uint8_t;

                // Note: replacing the invoke() with HPX_INVOKE()
                // below makes gcc generate errors
                auto f1 =
                    [tok, last, pred_projected = HPX_MOVE(pred_projected)](
                        FwdIter_ part_begin, std::size_t part_size) mutable
                    -> intermediate_result_t {
                    FwdIter_ trail = part_begin++;
                    util::loop_n<std::decay_t<ExPolicy>>(part_begin,
                        part_size - 1,
                        [&trail, &tok, &pred_projected](
                            FwdIter_ it) mutable -> void {
                            if (hpx::invoke(pred_projected, *it, *trail++))
                            {
                                tok.cancel();
                            }
                        });

                    FwdIter_ i = trail++;

                    // trail now points one past the current grouping unless
                    // canceled

                    if (!tok.was_cancelled() && trail != last)
                    {
                        return !hpx::invoke(pred_projected, *trail, *i);
                    }

                    return !tok.was_cancelled();
                };

                auto f2 = [](auto&& results) {
                    return std::all_of(hpx::util::begin(results),
                        hpx::util::end(results), hpx::functional::unwrap{});
                };

                return util::partitioner<ExPolicy, bool,
                    intermediate_result_t>::call(HPX_FORWARD(ExPolicy, policy),
                    first, count, HPX_MOVE(f1), HPX_MOVE(f2));
            }
        };
        /// \endcond
    }    // namespace detail

    ////////////////////////////////////////////////////////////////////////////
    // is_sorted_until
    namespace detail {

        /// \cond NOINTERNAL
        HPX_CXX_CORE_EXPORT template <typename FwdIter, typename Sent>
        struct is_sorted_until
          : public algorithm<is_sorted_until<FwdIter, Sent>, FwdIter>
        {
            constexpr is_sorted_until() noexcept
              : algorithm<is_sorted_until, FwdIter>("is_sorted_until")
            {
            }

            template <typename ExPolicy, typename FwdIter_, typename Sent_,
                typename Pred, typename Proj>
            static constexpr FwdIter_ sequential(
                ExPolicy, FwdIter_ first, Sent_ last, Pred&& pred, Proj&& proj)
            {
                return is_sorted_until_sequential(first, last,
                    HPX_FORWARD(Pred, pred), HPX_FORWARD(Proj, proj));
            }

            template <typename ExPolicy, typename FwdIter_, typename Sent_,
                typename Pred, typename Proj>
            static decltype(auto) parallel(ExPolicy&& orgpolicy, FwdIter_ first,
                Sent_ last, Pred&& pred, Proj&& proj)
            {
                using reference = hpx::traits::iter_reference_t<FwdIter_>;
                using difference_type =
                    hpx::traits::iter_difference_t<FwdIter_>;
                using result =
                    typename util::detail::algorithm_result<ExPolicy, FwdIter_>;
                constexpr bool has_scheduler_executor =
                    hpx::execution_policy_has_scheduler_executor_v<ExPolicy>;

                difference_type count = detail::distance(first, last);
                if constexpr (!has_scheduler_executor)
                {
                    if (count <= 1)
                        return result::get(HPX_MOVE(last));
                }

                util::invoke_projected<Pred, Proj> pred_projected{
                    HPX_FORWARD(Pred, pred), HPX_FORWARD(Proj, proj)};

                decltype(auto) policy =
                    hpx::execution::experimental::adapt_placement_mode(
                        HPX_FORWARD(ExPolicy, orgpolicy),
                        hpx::threads::thread_placement_hint::breadth_first);

                using policy_type = std::decay_t<decltype(policy)>;

                hpx::parallel::util::cancellation_token<difference_type> tok(
                    count);

                // Note: replacing the invoke() with HPX_INVOKE() below makes
                // gcc generate errors
                auto f1 = [tok, last,
                              pred_projected = HPX_MOVE(pred_projected)](
                              FwdIter_ part_begin, std::size_t part_size,
                              std::size_t base_idx) mutable -> void {
                    std::size_t const cross_idx = base_idx + part_size;

                    FwdIter_ trail = part_begin++;
                    util::loop_idx_n<policy_type>(++base_idx, part_begin,
                        part_size - 1, tok,
                        [&trail, &tok, &pred_projected](
                            reference& v, std::size_t ind) -> void {
                            if (hpx::invoke(pred_projected, v, *trail++))
                            {
                                tok.cancel(ind);
                            }
                        });

                    FwdIter_ i = trail++;

                    // trail now points one past the current grouping unless
                    // canceled

                    if (!tok.was_cancelled(cross_idx) && trail != last)
                    {
                        if (HPX_INVOKE(pred_projected, *trail, *i))
                        {
                            tok.cancel(cross_idx);
                        }
                    }
                };

                auto f2 = [first, tok](auto&&... data) mutable -> FwdIter_ {
                    static_assert(sizeof...(data) < 2);

                    difference_type cancelled_at = tok.get_data();
                    if (cancelled_at != count)
                    {
                        return detail::advance(first, cancelled_at);
                    }
                    return detail::advance(first, count);
                };

                return util::partitioner<ExPolicy, FwdIter_>::call_with_index(
                    HPX_FORWARD(decltype(policy), policy), first, count, 1,
                    HPX_MOVE(f1), HPX_MOVE(f2));
            }
        };
        /// \endcond
    }    // namespace detail

    HPX_CXX_CORE_EXPORT inline constexpr struct is_sorted_t final
      : hpx::detail::tag_parallel_algorithm<is_sorted_t>
    {
    private:
        template <typename FwdIter, typename Pred = hpx::parallel::detail::less>
        // clang-format off
            requires (
                std::forward_iterator<FwdIter> &&
                hpx::is_invocable_v<Pred,
                    hpx::traits::iter_value_t<FwdIter>,
                    hpx::traits::iter_value_t<FwdIter>
                >
            )
        // clang-format on
        friend bool tag_fallback_invoke(
            hpx::is_sorted_t, FwdIter first, FwdIter last, Pred pred = Pred())
        {
            return hpx::parallel::detail::is_sorted<FwdIter, FwdIter>().call(
                hpx::execution::seq, first, last, HPX_MOVE(pred),
                hpx::identity_v);
        }

        template <typename ExPolicy, typename FwdIter,
            typename Pred = hpx::parallel::detail::less>
        // clang-format off
            requires (
                hpx::is_execution_policy_v<ExPolicy> &&
                std::forward_iterator<FwdIter> &&
                hpx::is_invocable_v<Pred,
                    hpx::traits::iter_value_t<FwdIter>,
                    hpx::traits::iter_value_t<FwdIter>
                >
            )
        // clang-format on
        friend decltype(auto) tag_fallback_invoke(hpx::is_sorted_t,
            ExPolicy&& policy, FwdIter first, FwdIter last, Pred pred = Pred())
        {
            return hpx::parallel::detail::is_sorted<FwdIter, FwdIter>().call(
                HPX_FORWARD(ExPolicy, policy), first, last, HPX_MOVE(pred),
                hpx::identity_v);
        }

        template <typename FwdIter, typename Pred, typename Proj>
        // clang-format off
            requires (
                std::forward_iterator<FwdIter> &&
                hpx::parallel::traits::is_projected_v<Proj, FwdIter> &&
                hpx::is_invocable_v<Pred,
                    hpx::util::invoke_result_t<Proj,
                        hpx::traits::iter_value_t<FwdIter>>,
                    hpx::util::invoke_result_t<Proj,
                        hpx::traits::iter_value_t<FwdIter>>
                >
            )
        // clang-format on
        friend bool tag_fallback_invoke(
            hpx::is_sorted_t, FwdIter first, FwdIter last, Pred pred, Proj proj)
        {
            return hpx::parallel::detail::is_sorted<FwdIter, FwdIter>().call(
                hpx::execution::seq, first, last, HPX_MOVE(pred),
                HPX_MOVE(proj));
        }

        template <typename ExPolicy, typename FwdIter, typename Pred,
            typename Proj>
        // clang-format off
            requires (
                hpx::is_execution_policy_v<ExPolicy> &&
                std::forward_iterator<FwdIter> &&
                hpx::parallel::traits::is_projected_v<Proj, FwdIter> &&
                hpx::is_invocable_v<Pred,
                    hpx::util::invoke_result_t<Proj,
                        hpx::traits::iter_value_t<FwdIter>>,
                    hpx::util::invoke_result_t<Proj,
                        hpx::traits::iter_value_t<FwdIter>>
                >
            )
        // clang-format on
        friend decltype(auto) tag_fallback_invoke(hpx::is_sorted_t,
            ExPolicy&& policy, FwdIter first, FwdIter last, Pred pred,
            Proj proj)
        {
            return hpx::parallel::detail::is_sorted<FwdIter, FwdIter>().call(
                HPX_FORWARD(ExPolicy, policy), first, last, HPX_MOVE(pred),
                HPX_MOVE(proj));
        }
    } is_sorted{};

    HPX_CXX_CORE_EXPORT inline constexpr struct is_sorted_until_t final
      : hpx::detail::tag_parallel_algorithm<is_sorted_until_t>
    {
    private:
        template <typename FwdIter, typename Pred = hpx::parallel::detail::less>
        // clang-format off
            requires (
                std::forward_iterator<FwdIter> &&
                hpx::is_invocable_v<Pred,
                    hpx::traits::iter_value_t<FwdIter>,
                    hpx::traits::iter_value_t<FwdIter>
                >
            )
        // clang-format on
        friend FwdIter tag_fallback_invoke(hpx::is_sorted_until_t,
            FwdIter first, FwdIter last, Pred pred = Pred())
        {
            return hpx::parallel::detail::is_sorted_until<FwdIter, FwdIter>()
                .call(hpx::execution::seq, first, last, HPX_MOVE(pred),
                    hpx::identity_v);
        }

        template <typename ExPolicy, typename FwdIter,
            typename Pred = hpx::parallel::detail::less>
        // clang-format off
            requires (
                hpx::is_execution_policy_v<ExPolicy> &&
                std::forward_iterator<FwdIter> &&
                hpx::is_invocable_v<Pred,
                    hpx::traits::iter_value_t<FwdIter>,
                    hpx::traits::iter_value_t<FwdIter>
                >
            )
        // clang-format on
        friend decltype(auto) tag_fallback_invoke(hpx::is_sorted_until_t,
            ExPolicy&& policy, FwdIter first, FwdIter last, Pred pred = Pred())
        {
            return hpx::parallel::detail::is_sorted_until<FwdIter, FwdIter>()
                .call(HPX_FORWARD(ExPolicy, policy), first, last,
                    HPX_MOVE(pred), hpx::identity_v);
        }

        template <typename FwdIter, typename Pred, typename Proj>
        // clang-format off
            requires (
                std::forward_iterator<FwdIter> &&
                hpx::parallel::traits::is_projected_v<Proj, FwdIter> &&
                hpx::is_invocable_v<Pred,
                    hpx::util::invoke_result_t<Proj,
                        hpx::traits::iter_value_t<FwdIter>>,
                    hpx::util::invoke_result_t<Proj,
                        hpx::traits::iter_value_t<FwdIter>>
                >
            )
        // clang-format on
        friend FwdIter tag_fallback_invoke(hpx::is_sorted_until_t,
            FwdIter first, FwdIter last, Pred pred, Proj proj)
        {
            return hpx::parallel::detail::is_sorted_until<FwdIter, FwdIter>()
                .call(hpx::execution::seq, first, last, HPX_MOVE(pred),
                    HPX_MOVE(proj));
        }

        template <typename ExPolicy, typename FwdIter, typename Pred,
            typename Proj>
        // clang-format off
            requires (
                hpx::is_execution_policy_v<ExPolicy> &&
                std::forward_iterator<FwdIter> &&
                hpx::parallel::traits::is_projected_v<Proj, FwdIter> &&
                hpx::is_invocable_v<Pred,
                    hpx::util::invoke_result_t<Proj,
                        hpx::traits::iter_value_t<FwdIter>>,
                    hpx::util::invoke_result_t<Proj,
                        hpx::traits::iter_value_t<FwdIter>>
                >
            )
        // clang-format on
        friend decltype(auto) tag_fallback_invoke(hpx::is_sorted_until_t,
            ExPolicy&& policy, FwdIter first, FwdIter last, Pred pred,
            Proj proj)
        {
            return hpx::parallel::detail::is_sorted_until<FwdIter, FwdIter>()
                .call(HPX_FORWARD(ExPolicy, policy), first, last,
                    HPX_MOVE(pred), HPX_MOVE(proj));
        }
    } is_sorted_until{};
}    // namespace hpx::parallel
