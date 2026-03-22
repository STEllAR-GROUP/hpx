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

#include <hpx/execution_base/receiver_inlining.hpp>
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

        template <typename Receiver, typename Query>
        struct read_env_operation_state
          : inlinable_operation_state<read_env_operation_state<Receiver, Query>,
                Receiver>
        {
            using base_t = inlinable_operation_state<
                read_env_operation_state<Receiver, Query>, Receiver>;

            HPX_NO_UNIQUE_ADDRESS std::decay_t<Query> query;

            read_env_operation_state(Receiver&& r, Query&& q)
              : base_t(HPX_FORWARD(Receiver, r))
              , query(HPX_FORWARD(Query, q))
            {
            }

            friend void tag_invoke(
                start_t, read_env_operation_state& os) noexcept
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        auto env = hpx::execution::experimental::get_env(
                            os.get_receiver());
                        auto&& result = os.query(env);
                        hpx::execution::experimental::set_value(
                            HPX_MOVE(os.get_receiver()), HPX_MOVE(result));
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(os.get_receiver()), HPX_MOVE(ep));
                    });
            }
        };

        template <typename Query>
        struct read_env_sender
        {
            using is_sender = void;

            HPX_NO_UNIQUE_ADDRESS std::decay_t<Query> query;

            template <typename Env>
            struct generate_completion_signatures
            {
                using query_result_type =
                    decltype(std::declval<Query>()(std::declval<Env>()));

                template <template <typename...> typename Tuple,
                    template <typename...> typename Variant>
                using value_types = Variant<Tuple<query_result_type>>;

                template <template <typename...> typename Variant>
                using error_types = Variant<std::exception_ptr>;

                static constexpr bool sends_stopped = false;
            };

            template <typename Env>
            friend auto tag_invoke(get_completion_signatures_t,
                read_env_sender const&, Env) noexcept
                -> generate_completion_signatures<Env>;

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, read_env_sender&& s, Receiver&& receiver)
            {
                return read_env_operation_state<Receiver, Query>(
                    HPX_FORWARD(Receiver, receiver), HPX_MOVE(s.query));
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, read_env_sender const& s, Receiver&& receiver)
            {
                return read_env_operation_state<Receiver, Query>(
                    HPX_FORWARD(Receiver, receiver), s.query);
            }
        };
    }    // namespace detail

    inline constexpr struct read_env_t final
      : hpx::functional::detail::tag_fallback<read_env_t>
    {
    private:
        template <typename Query>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            read_env_t, Query&& query)
        {
            return detail::read_env_sender<Query>{HPX_FORWARD(Query, query)};
        }
    } read_env{};
}    // namespace hpx::execution::experimental

#endif
