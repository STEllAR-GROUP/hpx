//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/functional/tag_fallback_invoke.hpp>
#include <hpx/type_support/pack.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <type_traits>
#include <utility>

namespace hpx { namespace execution { namespace experimental {
    namespace detail {
        template <typename R, typename Scheduler>
        struct on_receiver
        {
            std::decay_t<R> r;
            std::decay_t<Scheduler> scheduler;

            template <typename R_, typename Scheduler_>
            on_receiver(R_&& r, Scheduler_&& scheduler)
              : r(std::forward<R>(r))
              , scheduler(std::forward<Scheduler>(scheduler))
            {
            }

            template <typename E>
            auto set_error(E&& e) noexcept
                -> decltype(hpx::execution::experimental::set_error(
                                std::move(r), std::forward<E>(e)),
                    void())
            {
                hpx::execution::experimental::set_error(
                    std::move(r), std::forward<E>(e));
            }

            auto set_done() noexcept -> decltype(
                hpx::execution::experimental::set_done(std::move(r)), void())
            {
                hpx::execution::experimental::set_done(std::move(r));
            };

            template <typename... Ts>
            auto set_value(Ts&&... ts) noexcept
                -> decltype(hpx::execution::experimental::set_value(
                                std::move(r), std::forward<Ts>(ts)...),
                    void())
            {
                hpx::tuple<Ts...> ts_pack{std::forward<Ts>(ts)...};
                hpx::execution::experimental::execute(scheduler,
                    [r = std::move(r), ts_pack = std::move(ts_pack)]() mutable {
                        hpx::util::invoke_fused(
                            hpx::util::bind_front(
                                hpx::execution::experimental::set_value,
                                std::move(r)),
                            std::move(ts_pack));
                    });
            }
        };

        template <typename S, typename Scheduler>
        struct on_sender
        {
            std::decay_t<S> s;
            std::decay_t<Scheduler> scheduler;

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types =
                typename hpx::execution::experimental::sender_traits<
                    S>::template value_types<Tuple, Variant>;

            template <template <typename...> class Variant>
            using error_types =
                typename hpx::execution::experimental::sender_traits<
                    S>::template error_types<Variant>;

            static constexpr bool sends_done = false;

            template <typename R>
            auto connect(R&& r) &&
            {
                return hpx::execution::experimental::connect(std::move(s),
                    on_receiver<R, Scheduler>(
                        std::forward<R>(r), std::move(scheduler)));
            }

            template <typename R>
            auto connect(R&& r) &
            {
                return hpx::execution::experimental::connect(s,
                    on_receiver<R, Scheduler>(std::forward<R>(r), scheduler));
            }
        };
    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct on_t final
      : hpx::functional::tag_fallback<on_t>
    {
    private:
        template <typename S, typename Scheduler>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            on_t, S&& s, Scheduler&& scheduler)
        {
            return detail::on_sender<S, Scheduler>{
                std::forward<S>(s), std::forward<Scheduler>(scheduler)};
        }

        template <typename Scheduler>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            on_t, Scheduler&& scheduler)
        {
            return detail::partial_algorithm<on_t, Scheduler>{
                std::forward<Scheduler>(scheduler)};
        }
    } on{};
}}}    // namespace hpx::execution::experimental
