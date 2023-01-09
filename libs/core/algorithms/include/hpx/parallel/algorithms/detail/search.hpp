//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  This file is based on the following cppreference possible implementation:
//  https://en.cppreference.com/w/cpp/algorithm/ranges/search
#pragma once

#include <hpx/config.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/functional/detail/invoke.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/util/adapt_placement_mode.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/parallel/util/compare_projected.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/clear_container.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx::parallel::detail {
    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    // search
    template <typename FwdIter, typename Sent>
    struct search final : public algorithm<search<FwdIter, Sent>, FwdIter>
    {
        constexpr search() noexcept
          : algorithm<search, FwdIter>("search")
        {
        }

        template <typename ExPolicy, typename FwdIter2, typename Sent2,
            typename Pred, typename Proj1, typename Proj2>
        static constexpr FwdIter sequential(ExPolicy, FwdIter first, Sent last,
            FwdIter2 s_first, Sent2 s_last, Pred&& op, Proj1&& proj1,
            Proj2&& proj2)
        {
            for (;; ++first)
            {
                FwdIter it1 = first;
                for (FwdIter2 it2 = s_first;; (void) ++it1, ++it2)
                {
                    if (it2 == s_last)
                        return first;
                    if (it1 == last)
                        return it1;
                    if (!HPX_INVOKE(op, HPX_INVOKE(proj1, *it1),
                            HPX_INVOKE(proj2, *it2)))
                        break;
                }
            }
        }

        template <typename ExPolicy, typename FwdIter2, typename Sent2,
            typename Pred, typename Proj1, typename Proj2>
        static hpx::parallel::util::detail::algorithm_result_t<ExPolicy,
            FwdIter>
        parallel(ExPolicy&& orgpolicy, FwdIter first, Sent last,
            FwdIter2 s_first, Sent2 s_last, Pred&& op, Proj1&& proj1,
            Proj2&& proj2)
        {
            using reference = typename std::iterator_traits<FwdIter>::reference;

            using difference_type =
                typename std::iterator_traits<FwdIter>::difference_type;

            using s_difference_type =
                typename std::iterator_traits<FwdIter2>::difference_type;

            using result =
                hpx::parallel::util::detail::algorithm_result<ExPolicy,
                    FwdIter>;

            // Use of hpx::distance instead of std::distance to support
            // sentinels
            s_difference_type diff =
                hpx::parallel::detail::distance(s_first, s_last);
            if (diff <= 0)
                return result::get(HPX_MOVE(first));

            difference_type count =
                hpx::parallel::detail::distance(first, last);
            if (diff > count)
            {
                std::advance(
                    first, hpx::parallel::detail::distance(first, last) - 1);
                return result::get(HPX_MOVE(first));
            }

            decltype(auto) policy = parallel::util::adapt_placement_mode(
                HPX_FORWARD(ExPolicy, orgpolicy),
                hpx::threads::thread_placement_hint::breadth_first);

            using policy_type = std::decay_t<decltype(policy)>;

            using partitioner =
                hpx::parallel::util::partitioner<decltype(policy), FwdIter,
                    void>;

            hpx::parallel::util::cancellation_token<difference_type> tok(count);

            auto f1 = [diff, count, tok, s_first, op = HPX_FORWARD(Pred, op),
                          proj1 = HPX_FORWARD(Proj1, proj1),
                          proj2 = HPX_FORWARD(Proj2, proj2)](FwdIter it,
                          std::size_t part_size,
                          std::size_t base_idx) mutable -> void {
                FwdIter curr = it;

                hpx::parallel::util::loop_idx_n<policy_type>(base_idx, it,
                    part_size, tok,
                    [diff, count, s_first, &tok, &curr,
                        op = HPX_FORWARD(Pred, op),
                        proj1 = HPX_FORWARD(Proj1, proj1),
                        proj2 = HPX_FORWARD(Proj2, proj2)](
                        reference v, std::size_t i) -> void {
                        ++curr;
                        if (HPX_INVOKE(op, HPX_INVOKE(proj1, v),
                                HPX_INVOKE(proj2, *s_first)))
                        {
                            difference_type local_count = 1;
                            FwdIter2 needle = s_first;
                            FwdIter mid = curr;

                            for (difference_type len = 0;
                                 local_count != diff && len != count;
                                 ++local_count, ++len, ++mid)
                            {
                                if (!HPX_INVOKE(op, HPX_INVOKE(proj1, *mid),
                                        HPX_INVOKE(proj2, *++needle)))
                                    break;
                            }

                            if (local_count == diff)
                                tok.cancel(i);
                        }
                    });
            };

            auto f2 = [=](auto&& data) mutable -> FwdIter {
                // make sure iterators embedded in function object that is
                // attached to futures are invalidated
                util::detail::clear_container(data);
                difference_type search_res = tok.get_data();
                if (search_res != count)
                {
                    std::advance(first, search_res);
                }
                else
                {
                    std::advance(first,
                        hpx::parallel::detail::distance(first, last) - 1);
                }

                return HPX_MOVE(first);
            };

            return partitioner::call_with_index(
                HPX_FORWARD(decltype(policy), policy), first,
                count - (diff - 1), 1, HPX_MOVE(f1), HPX_MOVE(f2));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // search_n
    template <typename FwdIter, typename Sent>
    struct search_n final : public algorithm<search_n<FwdIter, Sent>, FwdIter>
    {
        constexpr search_n() noexcept
          : algorithm<search_n, FwdIter>("search_n")
        {
        }

        template <typename ExPolicy, typename FwdIter2, typename Pred,
            typename Proj1, typename Proj2>
        static FwdIter sequential(ExPolicy, FwdIter first, std::size_t count,
            FwdIter2 s_first, FwdIter2 s_last, Pred&& op, Proj1&& proj1,
            Proj2&& proj2)
        {
            return std::search(first, std::next(first, count), s_first, s_last,
                util::compare_projected<Pred&, Proj1&, Proj2&>(
                    op, proj1, proj2));
        }

        template <typename ExPolicy, typename FwdIter2, typename Pred,
            typename Proj1, typename Proj2>
        static util::detail::algorithm_result_t<ExPolicy, FwdIter> parallel(
            ExPolicy&& orgpolicy, FwdIter first, std::size_t count,
            FwdIter2 s_first, FwdIter2 s_last, Pred&& op, Proj1&& proj1,
            Proj2&& proj2)
        {
            typedef typename std::iterator_traits<FwdIter>::reference reference;
            typedef typename std::iterator_traits<FwdIter>::difference_type
                difference_type;
            typedef typename std::iterator_traits<FwdIter2>::difference_type
                s_difference_type;
            typedef util::detail::algorithm_result<ExPolicy, FwdIter> result;

            s_difference_type diff = std::distance(s_first, s_last);
            if (diff <= 0)
                return result::get(HPX_MOVE(first));

            if (diff > s_difference_type(count))
                return result::get(HPX_MOVE(first));

            decltype(auto) policy = parallel::util::adapt_placement_mode(
                HPX_FORWARD(ExPolicy, orgpolicy),
                hpx::threads::thread_placement_hint::breadth_first);

            using policy_type = std::decay_t<decltype(policy)>;

            using partitioner =
                util::partitioner<decltype(policy), FwdIter, void>;

            hpx::parallel::util::cancellation_token<difference_type> tok(count);

            auto f1 = [count, diff, tok, s_first, op = HPX_FORWARD(Pred, op),
                          proj1 = HPX_FORWARD(Proj1, proj1),
                          proj2 = HPX_FORWARD(Proj2, proj2)](FwdIter it,
                          std::size_t part_size,
                          std::size_t base_idx) mutable -> void {
                FwdIter curr = it;

                util::loop_idx_n<policy_type>(base_idx, it, part_size, tok,
                    [count, diff, s_first, &tok, &curr,
                        op = HPX_FORWARD(Pred, op),
                        proj1 = HPX_FORWARD(Proj1, proj1),
                        proj2 = HPX_FORWARD(Proj2, proj2)](
                        reference v, std::size_t i) -> void {
                        ++curr;
                        if (HPX_INVOKE(op, HPX_INVOKE(proj1, v),
                                HPX_INVOKE(proj2, *s_first)))
                        {
                            difference_type local_count = 1;
                            FwdIter2 needle = s_first;
                            FwdIter mid = curr;

                            for (difference_type len = 0; local_count != diff &&
                                 len != difference_type(count);
                                 ++local_count, ++len, ++mid)
                            {
                                if (!HPX_INVOKE(op, HPX_INVOKE(proj1, *mid),
                                        HPX_INVOKE(proj2, *++needle)))
                                    break;
                            }

                            if (local_count == diff)
                                tok.cancel(i);
                        }
                    });
            };

            auto f2 = [=](auto&& data) mutable -> FwdIter {
                // make sure iterators embedded in function object that is
                // attached to futures are invalidated
                util::detail::clear_container(data);
                difference_type search_res = tok.get_data();
                if (search_res != s_difference_type(count))
                    std::advance(first, search_res);

                return HPX_MOVE(first);
            };

            return partitioner::call_with_index(
                HPX_FORWARD(decltype(policy), policy), first,
                count - (diff - 1), 1, HPX_MOVE(f1), HPX_MOVE(f2));
        }
    };
    /// \endcond
}    // namespace hpx::parallel::detail
