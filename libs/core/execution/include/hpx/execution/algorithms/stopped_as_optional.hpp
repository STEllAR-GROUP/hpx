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
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    namespace detail {

        template <typename Receiver>
        struct stopped_as_optional_receiver
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
                                hpx::optional<meta::pack<>>());
                        }
                        else if constexpr (sizeof...(Ts) == 1)
                        {
                            hpx::execution::experimental::set_value(
                                HPX_MOVE(receiver),
                                hpx::optional<std::decay_t<Ts>...>(
                                    HPX_FORWARD(Ts, ts)...));
                        }
                        else
                        {
                            hpx::execution::experimental::set_value(
                                HPX_MOVE(receiver),
                                hpx::optional<hpx::tuple<std::decay_t<Ts>...>>(
                                    hpx::make_tuple(HPX_FORWARD(Ts, ts)...)));
                        }
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(receiver), HPX_MOVE(ep));
                    });
            }

            template <typename... Ts>
            friend void tag_invoke(set_value_t,
                stopped_as_optional_receiver&& r, Ts&&... ts) noexcept
            {
                HPX_MOVE(r).set_value_helper(HPX_FORWARD(Ts, ts)...);
            }

            template <typename Error>
            friend void tag_invoke(set_error_t,
                stopped_as_optional_receiver&& r, Error&& error) noexcept
            {
                hpx::execution::experimental::set_error(
                    HPX_MOVE(r.receiver), HPX_FORWARD(Error, error));
            }

            friend void tag_invoke(
                set_stopped_t, stopped_as_optional_receiver&& r) noexcept
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        hpx::execution::experimental::set_value(
                            HPX_MOVE(r.receiver), hpx::nullopt);
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(r.receiver), HPX_MOVE(ep));
                    });
            }

            friend auto tag_invoke(
                get_env_t, stopped_as_optional_receiver const& r) noexcept
                -> env_of_t<std::decay_t<Receiver>>
            {
                return hpx::execution::experimental::get_env(r.receiver);
            }
        };

        template <typename Sender>
        struct stopped_as_optional_sender
        {
            using is_sender = void;

            HPX_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;

            template <typename... Ts>
            struct wrap_in_optional
            {
                using type = std::conditional_t<sizeof...(Ts) == 0,
                    hpx::optional<meta::pack<>>,
                    std::conditional_t<sizeof...(Ts) == 1,
                        hpx::optional<std::decay_t<Ts>...>,
                        hpx::optional<hpx::tuple<std::decay_t<Ts>...>>>>;
            };

            template <typename Env>
            struct generate_completion_signatures
            {
                template <template <typename...> typename Tuple,
                    template <typename...> typename Variant>
                using value_types = hpx::util::detail::transform_t<
                    value_types_of_t<Sender, Env, meta::pack, Variant>,
                    wrap_in_optional>;

                template <template <typename...> typename Variant>
                using error_types = hpx::util::detail::unique_concat_t<
                    error_types_of_t<Sender, Env, Variant>,
                    Variant<std::exception_ptr>>;

                static constexpr bool sends_stopped = false;
            };

            template <typename Env>
            friend auto tag_invoke(get_completion_signatures_t,
                stopped_as_optional_sender const&, Env) noexcept
                -> generate_completion_signatures<Env>;

            template <typename CPO,
                HPX_CONCEPT_REQUIRES_(meta::value<
                    meta::one_of<std::decay_t<CPO>, set_value_t, set_error_t>>&&
                        detail::has_completion_scheduler_v<CPO, Sender>)>
            friend constexpr auto tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<CPO>
                    tag,
                stopped_as_optional_sender const& s)
            {
                return tag(s.sender);
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, stopped_as_optional_sender&& s, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(HPX_MOVE(s.sender),
                    stopped_as_optional_receiver<Receiver>{
                        HPX_FORWARD(Receiver, receiver)});
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, stopped_as_optional_sender& s, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(s.sender,
                    stopped_as_optional_receiver<Receiver>{
                        HPX_FORWARD(Receiver, receiver)});
            }
        };
    }    // namespace detail

    inline constexpr struct stopped_as_optional_t final
      : hpx::functional::detail::tag_fallback<stopped_as_optional_t>
    {
    private:
        template <typename Sender, HPX_CONCEPT_REQUIRES_(is_sender_v<Sender>)>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            stopped_as_optional_t, Sender&& sender)
        {
            return detail::stopped_as_optional_sender<Sender>{
                HPX_FORWARD(Sender, sender)};
        }
    } stopped_as_optional{};
}    // namespace hpx::execution::experimental

#endif
