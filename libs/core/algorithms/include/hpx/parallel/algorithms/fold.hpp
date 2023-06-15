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
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <iterator>
#include <numeric>

namespace hpx::parallel::detail {

    template <typename T_>
    struct fold_left : public algorithm<fold_left<T_>, T_>
    {
        constexpr fold_left() noexcept
          : algorithm<fold_left, T_>("fold_left")
        {
        }

        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename T, typename F>
        HPX_HOST_DEVICE static constexpr T sequential(
            ExPolicy&&, FwdIter first, Sent last, T&& init, F&& f)
        {
            return util::loop_ind<ExPolicy>(
                first, last, [&init, &f](auto const& it) mutable {
                    init = HPX_INVOKE(f, HPX_MOVE(init), it);
                });
        }

        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename T, typename F>
        static constexpr auto parallel(
            ExPolicy&&, FwdIter first, Sent last, T&& init, F&& f)
        {
            return util::loop_ind<ExPolicy>(
                first, last, [&init, &f](auto const& it) mutable {
                    init = HPX_INVOKE(f, HPX_MOVE(init), it);
                });
        }
    };

}    // namespace hpx::parallel::detail

namespace hpx {
    inline constexpr struct fold_left_t final
      : hpx::detail::tag_parallel_algorithm<fold_left_t>
    {
    private:
        template <typename ExPolicy, typename FwdIter, typename T,
            typename F>    // TODO : add concept
        friend T tag_fallback_invoke(fold_left_t, ExPolicy&& policy,
            FwdIter first, FwdIter last, T init, F f)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::fold_left<T>().call(
                HPX_FORWARD(ExPolicy, policy), first, last,
                HPX_FORWARD(T, init), HPX_FORWARD(F, f));
        }

        template <typename FwdIter, typename T,
            typename F>    // TODO : add concept
        friend T tag_fallback_invoke(
            fold_left_t, FwdIter first, FwdIter last, T init, F f)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::fold_left<T>().call(
                ::hpx::execution::seq, first, last, HPX_FORWARD(T, init),
                HPX_FORWARD(F, f));
        }
    } fold_left{};
}    // namespace hpx

namespace hpx::parallel::detail {

    template <typename T_>
    struct fold_left_first : public algorithm<fold_left_first<T_>, T_>
    {
        constexpr fold_left_first() noexcept
          : algorithm<fold_left_first, T_>("fold_left_first")
        {
        }

        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename F>
        HPX_HOST_DEVICE static constexpr auto sequential(
            ExPolicy&&, FwdIter first, Sent last, F&& f)
        {
            using T = ::hpx::traits::iter_value_t<FwdIter>;
            using U =
                decltype(hpx::fold_left(HPX_MOVE(first), last, T(*first), f));

            if (first == last)
                return hpx::optional<U>();

            T init = *first;

            std::advance(first, 1);

            return hpx::optional<U>(hpx::parallel::detail::fold_left<T>().call(
                ::hpx::execution::seq, first, last, HPX_FORWARD(T, init),
                HPX_FORWARD(F, f)));
        }

        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename F>
        static constexpr auto parallel(
            ExPolicy&& policy, FwdIter first, Sent last, F&& f)
        {
            using T = ::hpx::traits::iter_value_t<FwdIter>;
            using U =
                decltype(hpx::fold_left(HPX_MOVE(first), last, T(*first), f));

            if (first == last)
                return hpx::optional<U>();

            T init = *first;

            std::advance(first, 1);

            return hpx::optional<U>(hpx::parallel::detail::fold_left<T>().call(
                HPX_FORWARD(ExPolicy, policy), first, last,
                HPX_FORWARD(T, init), HPX_FORWARD(F, f)));
        }
    };

}    // namespace hpx::parallel::detail

namespace hpx {
    inline constexpr struct fold_left_first_t final
      : hpx::detail::tag_parallel_algorithm<fold_left_first_t>
    {
    private:
        template <typename ExPolicy, typename FwdIter,
            typename F>    // TODO : add concept
        friend auto tag_fallback_invoke(fold_left_first_t, ExPolicy&& policy,
            FwdIter first, FwdIter last, F f)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            using U = decltype(hpx::fold_left(HPX_MOVE(first), last,
                ::hpx::traits::iter_value_t<FwdIter>(*first), f));

            return hpx::parallel::detail::fold_left_first<hpx::optional<U>>()
                .call(HPX_FORWARD(ExPolicy, policy), first, last,
                    HPX_FORWARD(F, f));
        }

        template <typename FwdIter, typename F>    // TODO : add concept
        friend auto tag_fallback_invoke(
            fold_left_first_t, FwdIter first, FwdIter last, F f)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            using U = decltype(hpx::fold_left(HPX_MOVE(first), last,
                ::hpx::traits::iter_value_t<FwdIter>(*first), f));

            return hpx::parallel::detail::fold_left_first<hpx::optional<U>>()
                .call(::hpx::execution::seq, first, last, HPX_FORWARD(F, f));
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
            ExPolicy&&, FwdIter first, Sent last, T&& init, F&& f)
        {
            using U = std::decay_t<std::invoke_result_t<F&,
                hpx::traits::iter_reference_t<FwdIter>, T>>;
            if (first == last)
                return U(HPX_MOVE(init));

            U accum = f(*--last, HPX_MOVE(init));
            while (first != last)
                accum = f(*--last, HPX_MOVE(accum));
            return accum;
        }

        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename T, typename F>
        static constexpr auto parallel(
            ExPolicy&& policy, FwdIter first, Sent last, T&& init, F&& f)
        {
            HPX_UNUSED(policy);
            HPX_UNUSED(first);
            HPX_UNUSED(last);
            HPX_UNUSED(init);
            HPX_UNUSED(f);

            // parallel version of fold_right has not been implemented
            exit(1);
            return f(first, init);
        }
    };

}    // namespace hpx::parallel::detail

namespace hpx {
    inline constexpr struct fold_right_t final
      : hpx::detail::tag_parallel_algorithm<fold_right_t>
    {
    private:
        template <typename ExPolicy, typename FwdIter, typename T,
            typename F>    // TODO : add concept
        friend T tag_fallback_invoke(fold_right_t, ExPolicy&& policy,
            FwdIter first, FwdIter last, T init, F f)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::fold_right<T>().call(
                HPX_FORWARD(ExPolicy, policy), first, last,
                HPX_FORWARD(T, init), HPX_FORWARD(F, f));
        }

        template <typename FwdIter, typename T,
            typename F>    // TODO : add concept
        friend T tag_fallback_invoke(
            fold_right_t, FwdIter first, FwdIter last, T init, F f)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::fold_right<T>().call(
                ::hpx::execution::seq, first, last, HPX_FORWARD(T, init),
                HPX_FORWARD(F, f));
        }
    } fold_right{};
}    // namespace hpx
