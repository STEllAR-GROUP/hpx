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
#include <hpx/datastructures/tuple.hpp>
#include <hpx/datastructures/variant.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution/algorithms/detail/inject_scheduler.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution/algorithms/run_loop.hpp>
#include <hpx/execution_base/completion_scheduler.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/get_env.hpp>
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
        struct let_value_sender
        {
            using is_sender = void;
            using predecessor_sender_t = std::decay_t<PredecessorSender>;

            HPX_NO_UNIQUE_ADDRESS predecessor_sender_t predecessor_sender;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Scheduler> scheduler;

            template <typename Env = empty_env>
            struct generate_completion_signatures
            {
                template <typename Pack, typename Enable = void>
                struct successor_sender_types_helper;

                template <typename... Ts>
                struct successor_sender_types_helper<meta::pack<Ts...>,
                    std::enable_if_t<hpx::is_invocable_v<F,
                        std::add_lvalue_reference_t<std::decay_t<Ts>>...>>>
                {
                    using type = hpx::util::invoke_result_t<F,
                        std::add_lvalue_reference_t<std::decay_t<Ts>>...>;

                    static_assert(is_sender_v<type>,
                        "let_value expects the invocable sender factory to "
                        "return a sender");
                };

                template <typename... Ts>
                using successor_sender_types = meta::type<
                    successor_sender_types_helper<meta::pack<Ts...>>>;

                // Type of the potential values returned from the predecessor
                // sender
                template <template <typename...> class Tuple = meta::pack,
                    template <typename...> class Variant = meta::pack>
                using predecessor_value_types =
                    value_types_of_t<predecessor_sender_t, Env, Tuple, Variant>;

                // Types of the potential senders returned from the sender
                // factory F

                // clang-format off
                template <template <typename...> typename Tuple = meta::pack,
                    template <typename...> typename Variant = meta::pack>
                using sender_types =
                    meta::apply<
                        meta::transform<meta::uncurry<
                            meta::func<successor_sender_types>>,
                            meta::func<Variant>>,
                        predecessor_value_types<Tuple>>;

                template <template <typename...> typename Tuple,
                    template <typename...> typename Variant>
                using value_types =
                    meta::apply<
                        meta::transform<
                            meta::bind_back<
                                meta::defer<detail::value_types_of>, Env,
                                meta::func<Tuple>>,
                            meta::uncurry<meta::unique<meta::func<Variant>>>>,
                        sender_types<>>;

                template <template <typename...> typename Variant>
                using error_types =
                    meta::invoke<
                        meta::push_back<meta::uncurry<
                            meta::unique<meta::func<Variant>>>>,
                        meta::apply<
                            meta::transform<meta::bind_back<
                                meta::defer<detail::error_types_of>, Env>>,
                            sender_types<>>,
                        meta::pack<std::exception_ptr>>;

                static constexpr bool sends_stopped = meta::value<
                    meta::apply<
                        meta::transform<
                            meta::bind_back1_func<
                                detail::sends_stopped_of, Env>,
                            meta::right_fold<
                                detail::sends_stopped_of<
                                    predecessor_sender_t, Env>,
                                meta::func2<meta::or_>>>,
                        sender_types<>>>;
                // clang-format on
            };

            template <typename Env>
            friend auto tag_invoke(get_completion_signatures_t,
                let_value_sender const&, Env) noexcept
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
                let_value_sender const& sender)
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
                let_value_sender const& sender)
            {
                return sender.scheduler;
            }

            // TODO: add forwarding_sender_query

            template <typename Receiver>
            struct operation_state
            {
                struct let_value_predecessor_receiver;

                // Type of the operation state returned when connecting the
                // predecessor sender to the let_value_predecessor_receiver
                using predecessor_operation_state_type =
                    std::decay_t<connect_result_t<PredecessorSender&&,
                        let_value_predecessor_receiver>>;

                // Type of the potential operation states returned when
                // connecting a sender in successor_sender_types to the receiver
                // connected to the let_value_sender
                template <typename Sender>
                struct successor_operation_state_types_helper
                {
                    using type = connect_result_t<Sender, Receiver>;
                };

                template <template <typename...> class Tuple,
                    template <typename...> class Variant>
                using successor_operation_state_types =
                    hpx::util::detail::transform_t<
                        typename generate_completion_signatures<>::
                            template sender_types<Tuple, Variant>,
                        successor_operation_state_types_helper>;

                // Operation state from connecting predecessor sender to
                // let_value_predecessor_receiver
                predecessor_operation_state_type predecessor_op_state;

                using predecessor_ts_type = hpx::util::detail::prepend_t<
                    typename generate_completion_signatures<>::
                        template predecessor_value_types<decayed_tuple,
                            hpx::variant>,
                    hpx::monostate>;

                // Potential values returned from the predecessor sender
                predecessor_ts_type predecessor_ts;

                // Potential operation states returned when connecting a sender
                // in successor_sender_types to the receiver connected to the
                // let_value_sender
                hpx::util::detail::prepend_t<
                    successor_operation_state_types<hpx::tuple, hpx::variant>,
                    hpx::monostate>
                    successor_op_state;

                struct let_value_predecessor_receiver
                {
                    HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
                    HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;
                    operation_state& op_state;

                    template <typename Receiver_, typename F_>
                    let_value_predecessor_receiver(
                        Receiver_&& receiver, F_&& f, operation_state& op_state)
                      : receiver(HPX_FORWARD(Receiver_, receiver))
                      , f(HPX_FORWARD(F_, f))
                      , op_state(op_state)
                    {
                    }

                    template <typename Error>
                    friend void tag_invoke(set_error_t,
                        let_value_predecessor_receiver&& r,
                        Error&& error) noexcept
                    {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(r.receiver), HPX_FORWARD(Error, error));
                    }

                    friend void tag_invoke(set_stopped_t,
                        let_value_predecessor_receiver&& r) noexcept
                    {
                        hpx::execution::experimental::set_stopped(
                            HPX_MOVE(r.receiver));
                    };

                    struct start_visitor
                    {
                        [[noreturn]] void operator()(hpx::monostate) const
                        {
                            HPX_UNREACHABLE;
                        }

                        template <typename OperationState_,
                            typename = std::enable_if_t<!std::is_same_v<
                                std::decay_t<OperationState_>, hpx::monostate>>>
                        void operator()(OperationState_& op_state) const
                        {
                            hpx::execution::experimental::start(op_state);
                        }
                    };

                    struct set_value_visitor
                    {
                        HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
                        HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;
                        operation_state& op_state;

                        [[noreturn]] void operator()(hpx::monostate) const
                        {
                            HPX_UNREACHABLE;
                        }

                        template <typename T,
                            typename = std::enable_if_t<!std::is_same_v<
                                std::decay_t<T>, hpx::monostate>>>
                        void operator()(T& t)
                        {
                            using operation_state_type =
                                decltype(hpx::execution::experimental::connect(
                                    hpx::invoke_fused(HPX_MOVE(f), t),
                                    std::declval<Receiver>()));

#if defined(HPX_HAVE_CXX17_COPY_ELISION)
                            // with_result_of is used to emplace the operation state
                            // returned from connect without any intermediate copy
                            // construction (the operation state is not required to be
                            // copyable nor movable).
                            op_state.successor_op_state
                                .template emplace<operation_state_type>(
                                    hpx::util::detail::with_result_of([&]() {
                                        return hpx::execution::experimental::
                                            connect(hpx::invoke_fused(
                                                        HPX_MOVE(f), t),
                                                HPX_MOVE(receiver));
                                    }));
#else
                            // earlier versions of MSVC don't get copy elision
                            // quite right, the operation state must be
                            // constructed explicitly directly in place
                            op_state.successor_op_state
                                .template emplace_f<operation_state_type>(
                                    hpx::execution::experimental::connect,
                                    hpx::invoke_fused(HPX_MOVE(f), t),
                                    HPX_MOVE(receiver));
#endif
                            hpx::visit(
                                start_visitor{}, op_state.successor_op_state);
                        }
                    };

                    // This typedef is duplicated from the parent struct. The
                    // parent typedef is not instantiated early enough for use
                    // here.
                    using predecessor_ts_type = hpx::util::detail::prepend_t<
                        typename generate_completion_signatures<>::
                            template predecessor_value_types<decayed_tuple,
                                hpx::variant>,
                        hpx::monostate>;

                    template <typename... Ts>
                    void set_value(Ts&&... ts)
                    {
                        hpx::detail::try_catch_exception_ptr(
                            [&]() {
                                op_state.predecessor_ts
                                    .template emplace<hpx::tuple<Ts...>>(
                                        HPX_FORWARD(Ts, ts)...);
                                hpx::visit(set_value_visitor{HPX_MOVE(receiver),
                                               HPX_MOVE(f), op_state},
                                    op_state.predecessor_ts);
                            },
                            [&](std::exception_ptr ep) {
                                hpx::execution::experimental::set_error(
                                    HPX_MOVE(receiver), HPX_MOVE(ep));
                            });
                    }

                    template <typename... Ts>
                    friend auto tag_invoke(set_value_t,
                        let_value_predecessor_receiver&& r, Ts&&... ts) noexcept
                        -> decltype(std::declval<predecessor_ts_type>()
                                        .template emplace<
                                            hpx::tuple<std::decay_t<Ts>...>>(
                                            HPX_FORWARD(Ts, ts)...),
                            void())
                    {
                        // set_value is in a member function only because of a
                        // compiler bug in GCC 7. When the body of set_value is
                        // inlined here compilation fails with an internal
                        // compiler error.
                        r.set_value(HPX_FORWARD(Ts, ts)...);
                    }

                    // Pass through the get_env receiver query
                    friend auto tag_invoke(
                        get_env_t, let_value_predecessor_receiver const& r)
                        -> env_of_t<std::decay_t<Receiver>>
                    {
                        return hpx::execution::experimental::get_env(
                            r.receiver);
                    }
                };

                template <typename PredecessorSender_, typename Receiver_,
                    typename F_>
                operation_state(PredecessorSender_&& predecessor_sender,
                    Receiver_&& receiver, F_&& f)
                  : predecessor_op_state{hpx::execution::experimental::connect(
                        HPX_FORWARD(PredecessorSender_, predecessor_sender),
                        let_value_predecessor_receiver(
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
                connect_t, let_value_sender&& s, Receiver&& receiver)
            {
                return operation_state<Receiver>(HPX_MOVE(s.predecessor_sender),
                    HPX_FORWARD(Receiver, receiver), HPX_MOVE(s.f));
            }
        };
    }    // namespace detail

    // let_value is very similar to then: when it is started, it invokes the
    // provided function with the values sent by the input sender as arguments.
    // However, where the sender returned from then sends exactly what that
    // function ends up returning - let_value requires that the function return
    // a sender, and the sender returned by let_value sends the values sent by
    // the sender returned from the callback. This is similar to the notion of
    // "future unwrapping" in future/promise-based frameworks.
    //
    // let_value is guaranteed to not begin executing function until the
    // returned sender is started.
    inline constexpr struct let_value_t final
      : hpx::functional::detail::tag_priority<let_value_t>
    {
    private:
        // clang-format off
        template <typename PredecessorSender, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_sender_v<PredecessorSender> &&
                experimental::detail::is_completion_scheduler_tag_invocable_v<
                    hpx::execution::experimental::set_value_t,
                    PredecessorSender, let_value_t, F
                >
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_override_invoke(
            let_value_t, PredecessorSender&& predecessor_sender, F&& f)
        {
            auto scheduler =
                hpx::execution::experimental::get_completion_scheduler<
                    hpx::execution::experimental::set_value_t>(
                    predecessor_sender);

            return hpx::functional::tag_invoke(let_value_t{},
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
        friend constexpr HPX_FORCEINLINE auto tag_invoke(let_value_t,
            hpx::execution::experimental::run_loop_scheduler const& sched,
            PredecessorSender&& predecessor_sender, F&& f)
        {
            return detail::let_value_sender<PredecessorSender, F,
                hpx::execution::experimental::run_loop_scheduler>{
                HPX_FORWARD(PredecessorSender, predecessor_sender),
                HPX_FORWARD(F, f), sched};
        }

        // clang-format off
        template <typename PredecessorSender, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_sender_v<PredecessorSender>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            let_value_t, PredecessorSender&& predecessor_sender, F&& f)
        {
            return detail::let_value_sender<PredecessorSender, F>{
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
            let_value_t, Scheduler&& scheduler, F&& f)
        {
            return hpx::execution::experimental::detail::inject_scheduler<
                let_value_t, Scheduler, F>{
                HPX_FORWARD(Scheduler, scheduler), HPX_FORWARD(F, f)};
        }

        template <typename F>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            let_value_t, F&& f)
        {
            return detail::partial_algorithm<let_value_t, F>{HPX_FORWARD(F, f)};
        }
    } let_value{};
}    // namespace hpx::execution::experimental
