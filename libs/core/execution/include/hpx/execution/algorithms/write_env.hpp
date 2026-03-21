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
#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/tag_invoke.hpp>
#include <hpx/modules/type_support.hpp>

#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    namespace detail {

        template <typename Receiver, typename Env>
        struct write_env_receiver
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Env> env;

            template <typename... Ts>
            friend void tag_invoke(
                set_value_t, write_env_receiver&& r, Ts&&... ts) noexcept
            {
                hpx::execution::experimental::set_value(
                    HPX_MOVE(r.receiver), HPX_FORWARD(Ts, ts)...);
            }

            template <typename Error>
            friend void tag_invoke(
                set_error_t, write_env_receiver&& r, Error&& error) noexcept
            {
                hpx::execution::experimental::set_error(
                    HPX_MOVE(r.receiver), HPX_FORWARD(Error, error));
            }

            friend void tag_invoke(
                set_stopped_t, write_env_receiver&& r) noexcept
            {
                hpx::execution::experimental::set_stopped(HPX_MOVE(r.receiver));
            }

            friend auto tag_invoke(get_env_t,
                write_env_receiver const& r) noexcept -> std::decay_t<Env>
            {
                return r.env;
            }
        };

        template <typename Sender, typename Env>
        struct write_env_sender
        {
            using is_sender = void;

            HPX_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Env> env;

            template <typename Env2>
            struct generate_completion_signatures
            {
                template <template <typename...> typename Tuple,
                    template <typename...> typename Variant>
                using value_types =
                    value_types_of_t<Sender, Env, Tuple, Variant>;

                template <template <typename...> typename Variant>
                using error_types = error_types_of_t<Sender, Env, Variant>;

                static constexpr bool sends_stopped =
                    sends_stopped_of_v<Sender, Env>;
            };

            template <typename Env2>
            friend auto tag_invoke(get_completion_signatures_t,
                write_env_sender const&, Env2) noexcept
                -> generate_completion_signatures<Env2>;

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, write_env_sender&& s, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(HPX_MOVE(s.sender),
                    write_env_receiver<Receiver, Env>{
                        HPX_FORWARD(Receiver, receiver), HPX_MOVE(s.env)});
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, write_env_sender& s, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(s.sender,
                    write_env_receiver<Receiver, Env>{
                        HPX_FORWARD(Receiver, receiver), s.env});
            }
        };
    }    // namespace detail

    inline constexpr struct write_env_t final
      : hpx::functional::detail::tag_priority<write_env_t>
    {
    private:
        template <typename Sender, typename Env,
            HPX_CONCEPT_REQUIRES_(is_sender_v<Sender>)>
        friend constexpr HPX_FORCEINLINE auto tag_invoke(
            write_env_t, Sender&& sender, Env&& env)
        {
            return detail::write_env_sender<Sender, Env>{
                HPX_FORWARD(Sender, sender), HPX_FORWARD(Env, env)};
        }

        template <typename Env>
        friend constexpr HPX_FORCEINLINE auto tag_invoke(write_env_t, Env&& env)
        {
            return detail::partial_algorithm<write_env_t, Env>{
                HPX_FORWARD(Env, env)};
        }
    } write_env{};
}    // namespace hpx::execution::experimental

#endif
