//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/member_pack.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
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

            constexpr just_sender() = default;

            template <typename T,
                typename = std::enable_if_t<
                    !std::is_same_v<std::decay_t<T>, just_sender>>>
            explicit constexpr just_sender(T&& t)
              : ts(std::piecewise_construct, std::forward<T>(t))
            {
            }

            template <typename T0, typename T1, typename... Ts_>
            explicit constexpr just_sender(T0&& t0, T1&& t1, Ts_&&... ts)
              : ts(std::piecewise_construct, std::forward<T0>(t0),
                    std::forward<T1>(t1), std::forward<Ts_>(ts)...)
            {
            }

            just_sender(just_sender&&) = default;
            just_sender(just_sender const&) = default;
            just_sender& operator=(just_sender&&) = default;
            just_sender& operator=(just_sender const&) = default;

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types = Variant<Tuple<std::decay_t<Ts>...>>;

            template <template <typename...> class Variant>
            using error_types = Variant<std::exception_ptr>;

            static constexpr bool sends_done = false;

            template <typename Receiver>
            struct operation_state
            {
                HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
                hpx::util::member_pack_for<std::decay_t<Ts>...> ts;

                template <typename Receiver_>
                operation_state(Receiver_&& receiver,
                    hpx::util::member_pack_for<std::decay_t<Ts>...> ts)
                  : receiver(std::forward<Receiver_>(receiver))
                  , ts(std::move(ts))
                {
                }

                operation_state(operation_state&&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                friend void tag_dispatch(start_t, operation_state& os) noexcept
                {
                    hpx::detail::try_catch_exception_ptr(
                        [&]() {
                            hpx::execution::experimental::set_value(
                                std::move(os.receiver),
                                std::move(os.ts).template get<Is>()...);
                        },
                        [&](std::exception_ptr ep) {
                            hpx::execution::experimental::set_error(
                                std::move(os.receiver), std::move(ep));
                        });
                }
            };

            template <typename Receiver>
            friend auto tag_dispatch(
                connect_t, just_sender&& s, Receiver&& receiver)
            {
                return operation_state<Receiver>{
                    std::forward<Receiver>(receiver), std::move(s.ts)};
            }

            template <typename Receiver>
            friend auto tag_dispatch(
                connect_t, just_sender& s, Receiver&& receiver)
            {
                return operation_state<Receiver>{
                    std::forward<Receiver>(receiver), s.ts};
            }
        };
    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct just_t final
    {
        template <typename... Ts>
        constexpr HPX_FORCEINLINE auto operator()(Ts&&... ts) const
        {
            return detail::just_sender<
                typename hpx::util::make_index_pack<sizeof...(Ts)>::type,
                Ts...>{std::forward<Ts>(ts)...};
        }
    } just{};
}}}    // namespace hpx::execution::experimental
