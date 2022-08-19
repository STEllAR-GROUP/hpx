//  Copyright (c) 2020 Thomas Heller
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/constexpr.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/coroutine_utils.hpp>
#include <hpx/execution_base/get_env.hpp>
#include <hpx/execution_base/operation_state.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/functional/traits/is_invocable.hpp>
#include <hpx/type_support/equality.hpp>
#include <hpx/type_support/meta.hpp>

#include <cstddef>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace execution { namespace experimental {
#if defined(DOXYGEN)
    /// connect is a customization point object.
    /// For some subexpression `s` and `r`, let `S` be the type such that `decltype((s))`
    /// is `S` and let `R` be the type such that `decltype((r))` is `R`. The result of
    /// the expression `hpx::execution::experimental::connect(s, r)` is then equivalent to:
    ///     * `s.connect(r)`, if that expression is valid and returns a type
    ///       satisfying the `operation_state`
    ///       (\see hpx::execution::experimental::traits::is_operation_state)
    ///       and if `S` satisfies the `sender` concept.
    ///     * `s.connect(r)`, if that expression is valid and returns a type
    ///       satisfying the `operation_state`
    ///       (\see hpx::execution::experimental::traits::is_operation_state)
    ///       and if `S` satisfies the `sender` concept.
    ///       Overload resolution is performed in a context that include the declaration
    ///       `void connect();`
    ///     * Otherwise, the expression is ill-formed.
    ///
    /// The customization is implemented in terms of
    /// `hpx::functional::tag_invoke`.
    template <typename S, typename R>
    void connect(S&& s, R&& r);

    /// The name schedule denotes a customization point object. For some
    /// subexpression s, let S be decltype((s)). The expression schedule(s) is
    /// expression-equivalent to:
    ///
    ///     * s.schedule(), if that expression is valid and its type models
    ///       sender.
    ///     * Otherwise, schedule(s), if that expression is valid and its type
    ///       models sender with overload resolution performed in a context that
    ///       includes the declaration
    ///
    ///           void schedule();
    ///
    ///       and that does not include a declaration of schedule.
    ///
    ///      * Otherwise, schedule(s) is ill-formed.
    ///
    /// The customization is implemented in terms of
    /// `hpx::functional::tag_invoke`.

#endif

    struct is_debug_env_t
    {
        template <typename Env,
            typename = std::enable_if_t<
                hpx::functional::is_tag_invocable_v<is_debug_env_t, Env>>>
        void operator()(Env&&) const noexcept;
    };

    // execution::connect is used to connect a sender with a receiver, producing
    // an operation state object that represents the work that needs to be
    // performed to satisfy the receiver contract of the receiver with values
    // that are the result of the operations described by the sender.
    //
    // execution::connect is a customization point which connects senders with
    // receivers, resulting in an operation state that will ensure that the
    // receiver contract of the receiver passed to connect will be fulfilled.
    //
    //      execution::sender auto snd = some input sender;
    //      execution::receiver auto rcv = some receiver;
    //      execution::operation_state auto state = execution::connect(snd, rcv);
    //
    //      execution::start(state);
    //
    //      // at this point, it is guaranteed that the work represented by state
    //      // has been submitted to an execution context, and that execution
    //      // context will eventually fulfill receiver contract of rcv
    //
    //      // operation states are not movable, and therefore this operation
    //      // state object must be kept alive until the operation finishes
    //
    HPX_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE struct connect_t
      : hpx::functional::tag<connect_t>
    {
        template <typename Sender, typename Receiver>
        static inline constexpr bool is_connectable_with_tag_invoke_v =
            is_sender_v<Sender, env_of_t<Receiver>>&&
                is_receiver_from_v<Receiver, Sender>&&
                    hpx::functional::is_tag_invocable_v<connect_t, Sender,
                        Receiver>;

        template <typename Sender, typename Receiver>
        static constexpr bool nothrow_connect() noexcept
        {
            if constexpr (is_connectable_with_tag_invoke_v<Sender, Receiver>)
            {
                return hpx::functional::is_nothrow_tag_invocable_v<connect_t,
                    Sender, Receiver>;
            }
            else
            {
                return false;
            }
        }

#if defined(HPX_HAVE_CXX20_COROUTINES)

        template <typename Sender, typename Receiver,
            typename = std::enable_if_t<
                is_connectable_with_tag_invoke_v<Sender, Receiver> ||
                hpx::is_invocable_v<connect_awaitable_t, Sender, Receiver> ||
                hpx::functional::is_tag_invocable_v<is_debug_env_t,
                    env_of_t<Receiver>>>>
        auto operator()(Sender&& sndr, Receiver&& rcvr) const
            noexcept(nothrow_connect<Sender, Receiver>())
        {
            if constexpr (is_connectable_with_tag_invoke_v<Sender, Receiver>)
            {
                // hpx::util::invoke_result_t<connect_t, std::decay_t<Sender>,
                // std::decay_t<Receiver>> is same as connect_result_t<S,R>
                std::decay_t<hpx::util::invoke_result_t<connect_t,
                    std::decay_t<Sender>, std::decay_t<Receiver>>>
                    operation_state;

                static_assert(
                    is_operation_state_v<hpx::functional::tag_invoke_result_t<
                        connect_t, Sender, Receiver>>,
                    "execution::connect(sender, receiver) must return a type "
                    "that "
                    "satisfies the operation_state concept");
                return tag_invoke(connect_t{}, HPX_FORWARD(Sender, sndr),
                    HPX_FORWARD(Receiver, rcvr));
            }
            else if constexpr (hpx::is_invocable_v<connect_awaitable_t, Sender,
                                   Receiver>)
            {
                return connect_awaitable(
                    HPX_FORWARD(Sender, sndr), HPX_FORWARD(Receiver, rcvr));
            }
            else
            {
                // This should generate an instantiate backtrace that contains useful
                // debugging information.
                return hpx::functional::tag_invoke(*this,
                    HPX_FORWARD(Sender, sndr), HPX_FORWARD(Receiver, rcvr));
            }
        }
#endif    // HPX_HAVE_CXX20_COROUTINES

    } connect{};

    namespace detail {
        template <typename S, typename R, typename Enable = void>
        struct connect_result_helper
        {
            struct dummy_operation_state
            {
            };
            using type = dummy_operation_state;
        };

        template <typename S, typename R>
        struct connect_result_helper<S, R,
            std::enable_if_t<hpx::is_invocable<connect_t, S, R>::value>>
          : hpx::util::invoke_result<connect_t, S, R>
        {
        };
    }    // namespace detail

    namespace detail {
        template <typename F, typename E>
        struct as_receiver
        {
            F f;

            void set_value() noexcept(noexcept(HPX_INVOKE(f, )))
            {
                HPX_INVOKE(f, );
            }

            template <typename E_>
            [[noreturn]] void set_error(E_&&) noexcept
            {
                std::terminate();
            }

            void set_stopped() noexcept {}
        };
    }    // namespace detail

    // Returns a sender describing the start of a task graph on the provided
    // scheduler.
    //
    // A scheduler is a lightweight handle that represents a strategy for
    // scheduling work onto an execution context. Since execution contexts don't
    // necessarily manifest in C++ code, it's not possible to program directly
    // against their API. A scheduler is a solution to that problem: the
    // scheduler concept is defined by a single sender algorithm, schedule,
    // which returns a sender that will complete on an execution context
    // determined by the scheduler. Logic that you want to run on that context
    // can be placed in the receiver's completion-signalling method.
    //
    //      // snd is a sender describing the creation of a new execution
    //      // resource on the execution context associated with sch
    //      execution::scheduler auto sch = thread_pool.scheduler();
    //      execution::sender auto snd = execution::schedule(sch);
    //
    // Note that a particular scheduler type may provide other kinds of
    // scheduling operations which are supported by its associated execution
    // context. It is not limited to scheduling purely using the
    // execution::schedule API.
    //
    HPX_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE
    struct schedule_t : hpx::functional::tag<schedule_t>
    {
    } schedule{};

    template <typename Scheduler, typename Enable = void>
    struct is_scheduler : std::false_type
    {
    };

    // different versions of clang-format disagree
    // clang-format off
    template <typename Scheduler>
    struct is_scheduler<Scheduler,
        std::enable_if_t<hpx::is_invocable_v<schedule_t, Scheduler> &&
            std::is_copy_constructible_v<Scheduler> &&
            hpx::traits::is_equality_comparable_v<Scheduler>>> : std::true_type
    {
    };
    // clang-format on

    template <typename Scheduler>
    inline constexpr bool is_scheduler_v = is_scheduler<Scheduler>::value;

    template <typename S>
    using schedule_result_t = hpx::util::invoke_result_t<schedule_t, S>;

    template <typename S, typename R>
    using connect_result_t = hpx::util::invoke_result_t<connect_t, S, R>;

    namespace detail {
        // Dummy type used in place of a scheduler if none is given
        struct no_scheduler
        {
        };
    }    // namespace detail
}}}      // namespace hpx::execution::experimental
