//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/member_pack.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/tag_fallback_invoke.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <utility>

namespace hpx { namespace execution { namespace experimental {
    namespace detail {
        template <typename Is, typename... Ts>
        struct just_sender;

        template <typename std::size_t... Is, typename... Ts>
        struct just_sender<hpx::util::index_pack<Is...>, Ts...>
        {
            // TODO: Are references allowed?
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
            using error_types = Variant<>;

            static constexpr bool sends_done = false;

            template <typename R>
            struct operation_state
            {
                std::decay_t<R> r;
                hpx::util::member_pack_for<std::decay_t<Ts>...> ts;

                void start() noexcept
                {
                    hpx::execution::experimental::set_value(
                        std::move(r), std::move(ts).template get<Is>()...);
                }
            };

            template <typename R>
            auto connect(R&& r)
            {
                return operation_state<R>{std::forward<R>(r), std::move(ts)};
            }
        };
    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct just_t final
      : hpx::functional::tag_fallback<just_t>
    {
    private:
        template <typename... Ts>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            just_t, Ts&&... ts)
        {
            return detail::just_sender<
                typename hpx::util::make_index_pack<sizeof...(Ts)>::type,
                Ts...>{std::forward<Ts>(ts)...};
        }
    } just{};
}}}    // namespace hpx::execution::experimental
