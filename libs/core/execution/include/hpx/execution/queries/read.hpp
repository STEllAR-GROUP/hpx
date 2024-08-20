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

            template <typename Tag1>
            friend auto tag_invoke(
                get_completion_signatures_t, read_sender<Tag1>, no_env)
                -> dependent_completion_signatures<no_env>;

            // clang-format off
            template <typename Env>
            friend auto tag_invoke(get_completion_signatures_t, read_sender, Env)
            {
                if constexpr (hpx::is_nothrow_invocable_v<Tag, Env>)
                {
                    using result_type =
                        completion_signatures<
                            set_value_t(hpx::util::invoke_result<Tag, Env>)>;
                    return result_type{};
                }
                else if constexpr (hpx::is_invocable_v<Tag, Env>)
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

    // execution::read is used to create a sender that retrieves a value from
    // the receiver's associated environment and sends it back to the receiver
    // through the value channel.
    //
    // execution::read is a customization point object of an unspecified class
    // type
    //
    // Returns a sender that reaches into a receiver's environment and pulls out
    // the current value associated with the customization point denoted by Tag.
    // It then sends the value read back to the receiver through the value
    // channel. For instance, get_scheduler() (with no arguments) is a sender
    // that asks the receiver for the currently suggested scheduler and passes
    // it to the receiver's set_value completion-signal.
    //
    // This can be useful when scheduling nested dependent work. The following
    // sender pulls the current scheduler into the value channel and then
    // schedules more work onto it. E.g.
    //
    //  execution::sender auto task =
    //      execution::get_scheduler()
    //        | execution::let_value(
    //              [](auto sched) {
    //                  return execution::on(sched, some nested work here);
    //              });
    //
    //  this_thread::sync_wait(std::move(task));    // wait for it to finish
    //
    // This code uses the fact that sync_wait associates a scheduler with the
    // receiver that it connects with task. get_scheduler() reads that scheduler
    // out of the receiver, and passes it to let_value's receiver's set_value
    // function, which in turn passes it to the lambda. That lambda returns a
    // new sender that uses the scheduler to schedule some nested work onto
    // sync_wait's scheduler.
    //
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
