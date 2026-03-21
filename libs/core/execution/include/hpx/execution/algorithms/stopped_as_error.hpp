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
#include <hpx/modules/errors.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/tag_invoke.hpp>
#include <hpx/modules/type_support.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    namespace detail {

        template <typename Receiver, typename Error>
        struct stopped_as_error_receiver
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Error> error;

            template <typename... Ts>
            friend void tag_invoke(
                set_value_t, stopped_as_error_receiver&& r, Ts&&... ts) noexcept
            {
                hpx::execution::experimental::set_value(
                    HPX_MOVE(r.receiver), HPX_FORWARD(Ts, ts)...);
            }

            template <typename E>
            friend void tag_invoke(
                set_error_t, stopped_as_error_receiver&& r, E&& e) noexcept
            {
                hpx::execution::experimental::set_error(
                    HPX_MOVE(r.receiver), HPX_FORWARD(E, e));
            }

            friend void tag_invoke(
                set_stopped_t, stopped_as_error_receiver&& r) noexcept
            {
                hpx::execution::experimental::set_error(
                    HPX_MOVE(r.receiver), HPX_MOVE(r.error));
            }

            friend auto tag_invoke(
                get_env_t, stopped_as_error_receiver const& r) noexcept
                -> env_of_t<std::decay_t<Receiver>>
            {
                return hpx::execution::experimental::get_env(r.receiver);
            }
        };

        template <typename Sender, typename Error>
        struct stopped_as_error_sender
        {
            using is_sender = void;

            HPX_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Error> error;

            template <typename Env>
            struct generate_completion_signatures
            {
                template <template <typename...> typename Tuple,
                    template <typename...> typename Variant>
                using value_types =
                    value_types_of_t<Sender, Env, Tuple, Variant>;

                template <template <typename...> typename Variant>
                using error_types = hpx::util::detail::unique_concat_t<
                    error_types_of_t<Sender, Env, Variant>, Variant<Error>>;

                static constexpr bool sends_stopped = false;
            };

            template <typename Env>
            friend auto tag_invoke(get_completion_signatures_t,
                stopped_as_error_sender const&, Env) noexcept
                -> generate_completion_signatures<Env>;

            template <typename CPO,
                HPX_CONCEPT_REQUIRES_(
                    meta::value<meta::one_of<std::decay_t<CPO>, set_value_t>>&&
                        detail::has_completion_scheduler_v<CPO, Sender>)>
            friend constexpr auto tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<CPO>
                    tag,
                stopped_as_error_sender const& s)
            {
                return tag(s.sender);
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, stopped_as_error_sender&& s, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(HPX_MOVE(s.sender),
                    stopped_as_error_receiver<Receiver, Error>{
                        HPX_FORWARD(Receiver, receiver), HPX_MOVE(s.error)});
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, stopped_as_error_sender& s, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(s.sender,
                    stopped_as_error_receiver<Receiver, Error>{
                        HPX_FORWARD(Receiver, receiver), s.error});
            }
        };
    }    // namespace detail

    inline constexpr struct stopped_as_error_t final
      : hpx::functional::detail::tag_priority<stopped_as_error_t>
    {
    private:
        template <typename Sender, typename Error,
            HPX_CONCEPT_REQUIRES_(is_sender_v<Sender>)>
        friend constexpr HPX_FORCEINLINE auto tag_invoke(
            stopped_as_error_t, Sender&& sender, Error&& error)
        {
            return detail::stopped_as_error_sender<Sender, Error>{
                HPX_FORWARD(Sender, sender), HPX_FORWARD(Error, error)};
        }

        template <typename Error>
        friend constexpr HPX_FORCEINLINE auto tag_invoke(
            stopped_as_error_t, Error&& error)
        {
            return detail::partial_algorithm<stopped_as_error_t, Error>{
                HPX_FORWARD(Error, error)};
        }
    } stopped_as_error{};
}    // namespace hpx::execution::experimental

#endif
