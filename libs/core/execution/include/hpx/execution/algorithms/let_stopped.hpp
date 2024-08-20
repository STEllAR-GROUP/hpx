//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/datastructures/optional.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/datastructures/variant.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution/algorithms/detail/inject_scheduler.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution/algorithms/run_loop.hpp>
#include <hpx/execution_base/completion_scheduler.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/detail/tag_priority_invoke.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/type_support/detail/with_result_of.hpp>
#include <hpx/type_support/meta.hpp>
#include <hpx/type_support/pack.hpp>

#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    namespace detail {

        template <typename PredecessorSender, typename F,
            typename Scheduler = no_scheduler>
        struct let_stopped_sender
        {
            using is_sender = void;
            using predecessor_sender_t = std::decay_t<PredecessorSender>;

            HPX_NO_UNIQUE_ADDRESS predecessor_sender_t predecessor_sender;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Scheduler> scheduler;

            template <typename Env = empty_env>
            struct generate_completion_signatures
            {
                struct successor_sender_type_helper
                {
                    using type = hpx::util::invoke_result_t<F>;

                    static_assert(is_sender_v<type>,
                        "let_stopped expects the invocable sender factory to "
                        "return a sender");
                };

                using sender_type = meta::type<successor_sender_type_helper>;

                // Type of the potential values returned from the predecessor
                // sender
                template <template <typename...> class Tuple = meta::pack,
                    template <typename...> class Variant = meta::pack>
                using predecessor_value_types =
                    value_types_of_t<predecessor_sender_t, Env, Tuple, Variant>;

                template <template <typename...> class Variant = meta::pack>
                using predecessor_error_types =
                    error_types_of_t<predecessor_sender_t, Env, Variant>;

                // Type of the potential values returned from the predecessor
                // sender
                template <template <typename...> class Tuple = meta::pack,
                    template <typename...> class Variant = meta::pack>
                using successor_value_types =
                    value_types_of_t<sender_type, Env, Tuple, Variant>;

                // Types of the potential senders returned from the sender
                // factory F

                // clang-format off
                template <template <typename...> typename Tuple = meta::pack,
                    template <typename...> typename Variant = meta::pack>
                using value_types =
                    meta::apply<
                        meta::push_back<meta::unique<meta::func<Variant>>>,
                        meta::pack<
                            value_types_of_t<sender_type, Env, Tuple, meta::pack>>,
                        predecessor_value_types<Tuple>>;

                template <template <typename...> typename Variant>
                using error_types =
                    meta::apply<
                        meta::push_back<meta::unique<meta::func<Variant>>>,
                        meta::pack<
                            error_types_of_t<sender_type, Env, meta::pack>>,
                        predecessor_error_types<>,
                        meta::pack<std::exception_ptr>>;

                static constexpr bool sends_stopped =
                    sends_stopped_of_v<sender_type, Env>;
                // clang-format on
            };

            template <typename Env>
            friend auto tag_invoke(get_completion_signatures_t,
                let_stopped_sender const&, Env) noexcept
                -> generate_completion_signatures<Env>;

            // clang-format off
            template <typename CPO, typename Scheduler_ = Scheduler,
                HPX_CONCEPT_REQUIRES_(
                   !hpx::execution::experimental::is_scheduler_v<Scheduler_> &&
                    hpx::execution::experimental::detail::is_receiver_cpo_v<CPO> &&
                    hpx::execution::experimental::detail::has_completion_scheduler_v<
                        CPO, predecessor_sender_t>
                )>
            // clang-format on
            friend constexpr auto tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<CPO>
                    tag,
                let_stopped_sender const& sender)
            {
                return tag(sender.predecessor_sender);
            }

            // clang-format off
            template <typename CPO, typename Scheduler_ = Scheduler,
                HPX_CONCEPT_REQUIRES_(
                    hpx::execution::experimental::is_scheduler_v<Scheduler_> &&
                    hpx::execution::experimental::detail::is_receiver_cpo_v<CPO>
                )>
            // clang-format on
            friend constexpr auto tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<CPO>,
                let_stopped_sender const& sender)
            {
                return sender.scheduler;
            }

            // TODO: add forwarding_sender_query

            template <typename Receiver>
            struct operation_state
            {
                struct let_stopped_predecessor_receiver;

                // Type of the operation state returned when connecting the
                // predecessor sender to the let_stopped_predecessor_receiver
                using predecessor_operation_state_type =
                    std::decay_t<connect_result_t<PredecessorSender&&,
                        let_stopped_predecessor_receiver>>;

                // Operation state from connecting predecessor sender to
                // let_stopped_predecessor_receiver
                predecessor_operation_state_type predecessor_op_state;

                struct let_stopped_predecessor_receiver
                {
                    HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
                    HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;
                    operation_state& op_state;

                    template <typename Receiver_, typename F_>
                    let_stopped_predecessor_receiver(
                        Receiver_&& receiver, F_&& f, operation_state& op_state)
                      : receiver(HPX_FORWARD(Receiver_, receiver))
                      , f(HPX_FORWARD(F_, f))
                      , op_state(op_state)
                    {
                    }

                    template <typename Error>
                    friend void tag_invoke(set_error_t,
                        let_stopped_predecessor_receiver&& r,
                        Error&& error) noexcept
                    {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(r.receiver), HPX_FORWARD(Error, error));
                    }

                    friend void tag_invoke(set_stopped_t,
                        let_stopped_predecessor_receiver&& r) noexcept
                    {
                        hpx::detail::try_catch_exception_ptr(
                            [&]() {
#if defined(HPX_HAVE_CXX17_COPY_ELISION)
                                // with_result_of is used to emplace the
                                // operation state returned from connect without
                                // any intermediate copy construction (the
                                // operation state is not required to be
                                // copyable nor movable).
                                r.op_state.successor_op_state.emplace(
                                    hpx::util::detail::with_result_of([&]() {
                                        return hpx::execution::experimental::
                                            connect(HPX_INVOKE(HPX_MOVE(r.f), ),
                                                HPX_MOVE(r.receiver));
                                    }));
#else
                                // earlier versions of MSVC don't get copy
                                // elision quite right, the operation state must
                                // be constructed explicitly directly in place
                                r.op_state.successor_op_state.emplace_f(
                                    hpx::execution::experimental::connect,
                                    HPX_INVOKE(HPX_MOVE(r.f), ),
                                    HPX_MOVE(r.receiver));
#endif
                                hpx::execution::experimental::start(
                                    *r.op_state.successor_op_state);
                            },
                            [&](std::exception_ptr ep) {
                                hpx::execution::experimental::set_error(
                                    HPX_MOVE(r.receiver), HPX_MOVE(ep));
                            });
                    };

                    template <typename... Ts,
                        typename = std::enable_if_t<hpx::is_invocable_v<
                            hpx::execution::experimental::set_value_t,
                            Receiver&&, Ts...>>>
                    friend void tag_invoke(set_value_t,
                        let_stopped_predecessor_receiver&& r,
                        Ts&&... ts) noexcept
                    {
                        hpx::execution::experimental::set_value(
                            HPX_MOVE(r.receiver), HPX_FORWARD(Ts, ts)...);
                    }
                };

                // Type of the potential operation state returned when
                // connecting a successor_sender_type to the receiver connected
                // to the let_stpped_sender
                using successor_operation_state_type =
                    connect_result_t<hpx::util::invoke_result_t<F>, Receiver>;

                // Potential operation states returned when connecting a sender
                // in successor_sender_types to the receiver connected to the
                // let_stopped_sender
                hpx::optional<successor_operation_state_type>
                    successor_op_state;

                template <typename PredecessorSender_, typename Receiver_,
                    typename F_>
                operation_state(PredecessorSender_&& predecessor_sender,
                    Receiver_&& receiver, F_&& f)
                  : predecessor_op_state{hpx::execution::experimental::connect(
                        HPX_FORWARD(PredecessorSender_, predecessor_sender),
                        let_stopped_predecessor_receiver(
                            HPX_FORWARD(Receiver_, receiver),
                            HPX_FORWARD(F_, f), *this))}
                {
                }

                operation_state(operation_state&&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                friend void tag_invoke(start_t, operation_state& os) noexcept
                {
                    hpx::execution::experimental::start(
                        os.predecessor_op_state);
                }
            };

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, let_stopped_sender&& s, Receiver&& receiver)
            {
                return operation_state<Receiver>(HPX_MOVE(s.predecessor_sender),
                    HPX_FORWARD(Receiver, receiver), HPX_MOVE(s.f));
            }
        };
    }    // namespace detail

    // let_stopped is very similar to then: when it is started, it invokes the
    // provided function with the values sent by the input sender as arguments.
    // However, where the sender returned from then sends exactly what that
    // function ends up returning - let_stopped requires that the function
    // return a sender, and the sender returned by let_stopped sends the values
    // sent by the sender returned from the callback. This is similar to the
    // notion of "future unwrapping" in future/promise-based frameworks.
    //
    // let_stopped is guaranteed to not begin executing function until the
    // returned sender is started.
    inline constexpr struct let_stopped_t final
      : hpx::functional::detail::tag_priority<let_stopped_t>
    {
    private:
        // clang-format off
        template <typename PredecessorSender, typename F,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<PredecessorSender> &&
                experimental::detail::is_completion_scheduler_tag_invocable_v<
                    hpx::execution::experimental::set_value_t,
                    PredecessorSender, let_stopped_t, F
                >
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_override_invoke(
            let_stopped_t, PredecessorSender&& predecessor_sender, F&& f)
        {
            auto scheduler =
                hpx::execution::experimental::get_completion_scheduler<
                    hpx::execution::experimental::set_value_t>(
                    predecessor_sender);

            return hpx::functional::tag_invoke(let_stopped_t{},
                HPX_MOVE(scheduler),
                HPX_FORWARD(PredecessorSender, predecessor_sender),
                HPX_FORWARD(F, f));
        }

        // clang-format off
        template <typename PredecessorSender, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_sender_v<PredecessorSender>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_invoke(let_stopped_t,
            hpx::execution::experimental::run_loop_scheduler const& sched,
            PredecessorSender&& predecessor_sender, F&& f)
        {
            return detail::let_stopped_sender<PredecessorSender, F,
                hpx::execution::experimental::run_loop_scheduler>{
                HPX_FORWARD(PredecessorSender, predecessor_sender),
                HPX_FORWARD(F, f), sched};
        }

        // clang-format off
        template <typename PredecessorSender, typename F,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<PredecessorSender>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            let_stopped_t, PredecessorSender&& predecessor_sender, F&& f)
        {
            return detail::let_stopped_sender<PredecessorSender, F>{
                HPX_FORWARD(PredecessorSender, predecessor_sender),
                HPX_FORWARD(F, f), detail::no_scheduler{}};
        }

        // clang-format off
        template <typename F, typename Scheduler,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_scheduler_v<Scheduler>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            let_stopped_t, Scheduler&& scheduler, F&& f)
        {
            return hpx::execution::experimental::detail::inject_scheduler<
                let_stopped_t, Scheduler, F>{
                HPX_FORWARD(Scheduler, scheduler), HPX_FORWARD(F, f)};
        }

        template <typename F>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            let_stopped_t, F&& f)
        {
            return detail::partial_algorithm<let_stopped_t, F>{
                HPX_FORWARD(F, f)};
        }
    } let_stopped{};
}    // namespace hpx::execution::experimental
