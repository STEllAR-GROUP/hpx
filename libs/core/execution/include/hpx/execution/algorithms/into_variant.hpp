//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_STDEXEC)
#include <hpx/modules/execution_base.hpp>
#else

#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/modules/concepts.hpp>
#include <hpx/modules/datastructures.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/tag_invoke.hpp>
#include <hpx/modules/type_support.hpp>

#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace hpx::execution::experimental {

    namespace detail {

        template <typename Receiver>
        struct into_variant_receiver
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

            template <typename... Ts>
            void set_value_helper(Ts&&... ts) && noexcept
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        if constexpr (sizeof...(Ts) == 0)
                        {
                            hpx::execution::experimental::set_value(
                                HPX_MOVE(receiver),
                                std::variant<std::tuple<>>());
                        }
                        else if constexpr (sizeof...(Ts) == 1)
                        {
                            hpx::execution::experimental::set_value(
                                HPX_MOVE(receiver),
                                std::variant<std::tuple<std::decay_t<Ts>...>>(
                                    std::make_tuple(HPX_FORWARD(Ts, ts)...)));
                        }
                        else
                        {
                            hpx::execution::experimental::set_value(
                                HPX_MOVE(receiver),
                                std::variant<std::tuple<std::decay_t<Ts>...>>(
                                    std::make_tuple(HPX_FORWARD(Ts, ts)...)));
                        }
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(receiver), HPX_MOVE(ep));
                    });
            }

            template <typename... Ts>
            friend void tag_invoke(
                set_value_t, into_variant_receiver&& r, Ts&&... ts) noexcept
            {
                HPX_MOVE(r).set_value_helper(HPX_FORWARD(Ts, ts)...);
            }

            template <typename Error>
            friend void tag_invoke(
                set_error_t, into_variant_receiver&& r, Error&& error) noexcept
            {
                hpx::execution::experimental::set_error(
                    HPX_MOVE(r.receiver), HPX_FORWARD(Error, error));
            }

            friend void tag_invoke(
                set_stopped_t, into_variant_receiver&& r) noexcept
            {
                hpx::execution::experimental::set_stopped(HPX_MOVE(r.receiver));
            }

            friend auto tag_invoke(
                get_env_t, into_variant_receiver const& r) noexcept
                -> env_of_t<std::decay_t<Receiver>>
            {
                return hpx::execution::experimental::get_env(r.receiver);
            }
        };

        template <typename Sender>
        struct into_variant_sender
        {
            using is_sender = void;

            HPX_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;

            template <typename... Ts>
            struct wrap_in_tuple
            {
                using type = hpx::tuple<std::decay_t<Ts>...>;
            };

            template <typename Env>
            struct generate_completion_signatures
            {
                template <template <typename...> typename Tuple,
                    template <typename...> typename Variant>
                using value_types = Variant<hpx::util::detail::transform_t<
                    value_types_of_t<Sender, Env, std::tuple, Variant>,
                    wrap_in_tuple>>;

                template <template <typename...> typename Variant>
                using error_types = hpx::util::detail::unique_concat_t<
                    error_types_of_t<Sender, Env, Variant>,
                    Variant<std::exception_ptr>>;

                static constexpr bool sends_stopped =
                    sends_stopped_of_v<Sender, Env>;
            };

            template <typename Env>
            friend auto tag_invoke(get_completion_signatures_t,
                into_variant_sender const&, Env) noexcept
                -> generate_completion_signatures<Env>;

            template <typename CPO,
                HPX_CONCEPT_REQUIRES_(
                    meta::value<meta::one_of<std::decay_t<CPO>, set_value_t>>&&
                        detail::has_completion_scheduler_v<CPO, Sender>)>
            friend constexpr auto tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<CPO>
                    tag,
                into_variant_sender const& s)
            {
                return tag(s.sender);
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, into_variant_sender&& s, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(HPX_MOVE(s.sender),
                    into_variant_receiver<Receiver>{
                        HPX_FORWARD(Receiver, receiver)});
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, into_variant_sender& s, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(s.sender,
                    into_variant_receiver<Receiver>{
                        HPX_FORWARD(Receiver, receiver)});
            }
        };
    }    // namespace detail

    inline constexpr struct into_variant_t final
      : hpx::functional::detail::tag_fallback<into_variant_t>
    {
    private:
        template <typename Sender, HPX_CONCEPT_REQUIRES_(is_sender_v<Sender>)>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            into_variant_t, Sender&& sender)
        {
            return detail::into_variant_sender<Sender>{
                HPX_FORWARD(Sender, sender)};
        }
    } into_variant{};
}    // namespace hpx::execution::experimental

#endif
