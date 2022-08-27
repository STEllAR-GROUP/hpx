//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/get_env.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/functional/traits/is_invocable.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <exception>
#include <stdexcept>
#include <utility>

namespace hpx::execution::experimental {

    namespace detail {

        template <typename Tag>
        struct read_sender
        {
            constexpr read_sender() = default;

            read_sender(read_sender&&) = default;
            read_sender(read_sender const&) = default;
            read_sender& operator=(read_sender&&) = default;
            read_sender& operator=(read_sender const&) = default;

            template <typename Receiver>
            struct operation_state
            {
                HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

                template <typename Receiver_>
                explicit operation_state(Receiver_&& receiver) noexcept
                  : receiver(HPX_FORWARD(Receiver_, receiver))
                {
                }

                operation_state(operation_state&&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                friend void tag_invoke(start_t, operation_state& os) noexcept
                {
                    hpx::detail::try_catch_exception_ptr(
                        [&]() {
                            auto env = hpx::execution::experimental::get_env(
                                os.receiver);
                            hpx::execution::experimental::set_value(
                                HPX_MOVE(os.receiver), Tag()(HPX_MOVE(env)));
                        },
                        [&](std::exception_ptr ep) {
                            hpx::execution::experimental::set_error(
                                HPX_MOVE(os.receiver), HPX_MOVE(ep));
                        });
                }
            };

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, read_sender&&, Receiver&& receiver) noexcept
            {
                return operation_state<Receiver>{
                    HPX_FORWARD(Receiver, receiver)};
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, read_sender&, Receiver&& receiver) noexcept
            {
                return operation_state<Receiver>{
                    HPX_FORWARD(Receiver, receiver)};
            }

            // clang-format off
            template <typename Env>
            friend auto tag_invoke(get_completion_signatures_t, read_sender, Env)
            {
                if constexpr (hpx::is_invocable_v<Tag, Env>)
                {
                    using result_type =
                        completion_signatures<
                            set_value_t(hpx::util::invoke_result<Tag, Env>),
                            set_error_t(std::exception_ptr)>;
                    return result_type{};
                }
                else
                {
                    return dependent_completion_signatures<Env>{};
                }
            }
            // clang-format on
        };
    }    // namespace detail

    inline constexpr struct read_t final : hpx::functional::tag<read_t>
    {
    private:
        template <typename Tag>
        friend constexpr auto tag_invoke(read_t, Tag) noexcept
        {
            return detail::read_sender<Tag>{};
        }
    } read{};

}    // namespace hpx::execution::experimental
