//  Copyright (c) 2020-2022 STE||AR Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/concepts/concepts.hpp>
#include <hpx/datastructures/optional.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/reduce.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <iterator>
#include <numeric>

namespace hpx {
    inline constexpr struct fold_left_t final
      : hpx::detail::tag_parallel_algorithm<fold_left_t>
    {
    private:
        template <typename ExPolicy, typename FwdIter, typename T, typename F>
        friend T tag_fallback_invoke(fold_left_t, ExPolicy&& policy,
            FwdIter first, FwdIter last, T init, F f)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return hpx::reduce(HPX_FORWARD(ExPolicy, policy), first, last, init,
                HPX_FORWARD(F, f));
        }

        template <typename FwdIter, typename T, typename F>
        friend T tag_fallback_invoke(
            fold_left_t, FwdIter first, FwdIter last, T init, F f)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return hpx::reduce(
                hpx::execution::seq, first, last, init, HPX_FORWARD(F, f));
        }
    } fold_left{};
}    // namespace hpx

namespace hpx::parallel::detail {
    template <typename ExPolicy, typename FwdIter, typename F>
    auto fold_left_first_helper(
        ExPolicy&& policy, FwdIter first, FwdIter last, F&& f)
    {
        using T = ::hpx::traits::iter_value_t<FwdIter>;
        using U = decltype(hpx::fold_left(HPX_MOVE(first), last, T(*first), f));

        if (first == last)
            return hpx::optional<U>();

        T init = *first++;

        return hpx::optional<U>(hpx::fold_left(HPX_FORWARD(ExPolicy, policy),
            first, last, HPX_MOVE(init), HPX_MOVE(f)));
    }
}    // namespace hpx::parallel::detail

namespace hpx {
    inline constexpr struct fold_left_first_t final
      : hpx::detail::tag_parallel_algorithm<fold_left_first_t>
    {
    private:
        template <typename ExPolicy, typename FwdIter, typename F>
        friend auto tag_fallback_invoke(fold_left_first_t, ExPolicy&& policy,
            FwdIter first, FwdIter last, F f)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::fold_left_first_helper(
                HPX_FORWARD(ExPolicy, policy), first, last, HPX_MOVE(f));
        }

        template <typename FwdIter, typename F>
        friend auto tag_fallback_invoke(
            fold_left_first_t, FwdIter first, FwdIter last, F f)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::fold_left_first_helper(
                hpx::execution::seq, first, last, HPX_MOVE(f));
        }
    } fold_left_first{};
}    // namespace hpx

namespace hpx::parallel::detail {

    template <typename T_>
    struct fold_right : public algorithm<fold_right<T_>, T_>
    {
        constexpr fold_right() noexcept
          : algorithm<fold_right, T_>("fold_right")
        {
        }

        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename T, typename F>
        HPX_HOST_DEVICE static constexpr auto sequential(
            ExPolicy&& policy, FwdIter first, Sent last, T&& init, F&& f)
        {
            // last++ moves backward when its reverse iterator
            return hpx::fold_left(HPX_FORWARD(ExPolicy, policy),
                std::make_reverse_iterator(last),
                std::make_reverse_iterator(first), HPX_FORWARD(T, init),
                HPX_FORWARD(F, f));
        }

        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename T, typename F>
        static constexpr auto parallel(
            ExPolicy&& policy, FwdIter first, Sent last, T&& init, F&& f)
        {
            if (first == last)
            {
                return init;
            }

            auto ChunkReduce = [f = HPX_FORWARD(F, f),
                                   policy = HPX_FORWARD(ExPolicy, policy)](
                                   FwdIter it, std::size_t chunkSize) {
                FwdIter endIter = it;
                std::advance(endIter, --chunkSize);

                T init = *endIter;

                return sequential(HPX_FORWARD(ExPolicy, policy), it, endIter,
                    init, HPX_FORWARD(F, f));
            };

            auto RecursiveReduce = [f, policy, init](auto&& results) mutable {
                auto begin = hpx::util::begin(results);
                auto end = hpx::util::end(results);
                return sequential(HPX_FORWARD(ExPolicy, policy), begin, end,
                    init, HPX_FORWARD(F, f));
            };

            return util::partitioner<ExPolicy, T>::call(
                HPX_FORWARD(ExPolicy, policy), first,
                std::distance(first, last), HPX_MOVE(ChunkReduce),
                hpx::unwrapping(HPX_MOVE(RecursiveReduce)));
        }
    };
};    // namespace hpx::parallel::detail

namespace hpx {
    inline constexpr struct fold_right_t final
      : hpx::detail::tag_parallel_algorithm<fold_right_t>
    {
    private:
        template <typename ExPolicy, typename FwdIter, typename T, typename F>
        friend T tag_fallback_invoke(fold_right_t, ExPolicy&& policy,
            FwdIter first, FwdIter last, T init, F f)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::fold_right<T>().call(
                HPX_FORWARD(ExPolicy, policy), first, last, HPX_MOVE(init),
                HPX_MOVE(f));
        }

        template <typename FwdIter, typename T, typename F>
        friend T tag_fallback_invoke(
            fold_right_t, FwdIter first, FwdIter last, T init, F f)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::fold_right<T>().call(
                ::hpx::execution::seq, first, last, HPX_MOVE(init),
                HPX_MOVE(f));
        }
    } fold_right{};
}    // namespace hpx

namespace hpx::parallel::detail {
    template <typename ExPolicy, typename FwdIter, typename F>
    auto fold_right_first_helper(
        ExPolicy&& policy, FwdIter first, FwdIter last, F&& f)
    {
        using T = ::hpx::traits::iter_value_t<FwdIter>;
        using U =
            decltype(hpx::fold_right(HPX_MOVE(first), last, T(*first), f));

        if (first == last)
            return hpx::optional<U>();

        T init = *--last;

        return hpx::optional<U>(hpx::fold_right(HPX_FORWARD(ExPolicy, policy),
            first, last, HPX_MOVE(init), HPX_MOVE(f)));
    }
}    // namespace hpx::parallel::detail

namespace hpx {
    inline constexpr struct fold_right_first_t final
      : hpx::detail::tag_parallel_algorithm<fold_right_first_t>
    {
    private:
        template <typename ExPolicy, typename FwdIter, typename F>
        friend auto tag_fallback_invoke(fold_right_first_t, ExPolicy&& policy,
            FwdIter first, FwdIter last, F f)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::fold_right_first_helper(
                HPX_FORWARD(ExPolicy, policy), first, last, HPX_MOVE(f));
        }

        template <typename FwdIter, typename F>
        friend auto tag_fallback_invoke(
            fold_right_first_t, FwdIter first, FwdIter last, F f)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::fold_right_first_helper(
                hpx::execution::seq, first, last, HPX_MOVE(f));
        }
    } fold_right_first{};
}    // namespace hpx
