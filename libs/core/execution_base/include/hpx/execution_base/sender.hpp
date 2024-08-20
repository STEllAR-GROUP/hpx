//  Copyright (c) 2020 Thomas Heller
//  Copyright (c) 2022-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/constexpr.hpp>
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
#include <type_traits>

namespace hpx::execution::experimental {

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
    struct connect_t;

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
            [[noreturn]] static void set_error(E_&&) noexcept
            {
                std::terminate();
            }

            static constexpr void set_stopped() noexcept {}
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
    // snd is a sender describing the creation of a new execution resource on
    // the execution context associated with sch
    //
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
        std::enable_if_t<
            hpx::is_invocable_v<schedule_t, std::decay_t<Scheduler>> &&
            std::is_copy_constructible_v<std::decay_t<Scheduler>> &&
            hpx::traits::is_equality_comparable_v<std::decay_t<Scheduler>>>>
      : std::true_type
    {
    };
    // clang-format on

    template <typename Scheduler>
    inline constexpr bool is_scheduler_v = is_scheduler<Scheduler>::value;

    template <typename S>
    using schedule_result_t = hpx::util::invoke_result_t<schedule_t, S>;

    namespace detail {

        // Dummy type used in place of a scheduler if none is given
        struct no_scheduler
        {
        };
    }    // namespace detail
}    // namespace hpx::execution::experimental
