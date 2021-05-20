//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/member_pack.hpp>
#include <hpx/execution_base/detail/try_catch_exception_ptr.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <exception>
#include <stdexcept>
#include <utility>

namespace hpx { namespace execution { namespace experimental {
    namespace detail {
        template <typename Is, typename... Ts>
        struct just_sender;

        template <typename std::size_t... Is, typename... Ts>
        struct just_sender<hpx::util::index_pack<Is...>, Ts...>
        {
            hpx::util::member_pack_for<std::decay_t<Ts>...> ts;

            template <typename... Ts_>
            explicit constexpr just_sender(Ts_&&... ts)
              : ts(std::piecewise_construct, std::forward<Ts_>(ts)...)
            {
            }

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types = Variant<Tuple<Ts...>>;

            template <template <typename...> class Variant>
            using error_types = Variant<std::exception_ptr>;

            static constexpr bool sends_done = false;

            template <typename R>
            struct operation_state
            {
                std::decay_t<R> r;
                hpx::util::member_pack_for<std::decay_t<Ts>...> ts;

                template <typename R_>
                operation_state(
                    R_&& r, hpx::util::member_pack_for<std::decay_t<Ts>...> ts)
                  : r(std::forward<R_>(r))
                  , ts(std::move(ts))
                {
                }

                operation_state(operation_state&&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                void start() & noexcept
                {
                    hpx::detail::try_catch_exception_ptr(
                        [&]() {
                            hpx::execution::experimental::set_value(
                                std::move(r),
                                std::move(ts).template get<Is>()...);
                        },
                        [&](std::exception_ptr ep) {
                            hpx::execution::experimental::set_error(
                                std::move(r), std::move(ep));
                        });
                }
            };

            template <typename R>
            auto connect(R&& r) &&
            {
                return operation_state<R>{std::forward<R>(r), std::move(ts)};
            }

            template <typename R>
            auto connect(R&& r) &
            {
                return operation_state<R>{std::forward<R>(r), ts};
            }
        };
    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct just_t final
      : hpx::functional::tag_fallback<just_t>
    {
    private:
        template <typename... Ts>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            just_t, Ts&&... ts)
        {
            return detail::just_sender<
                typename hpx::util::make_index_pack<sizeof...(Ts)>::type,
                Ts...>{std::forward<Ts>(ts)...};
        }
    } just{};
}}}    // namespace hpx::execution::experimental
