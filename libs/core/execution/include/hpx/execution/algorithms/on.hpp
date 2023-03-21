//  Copyright (c) 2023 Shreyas Atre
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/queries/get_scheduler.hpp>
#include <hpx/execution_base/completion_scheduler.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/get_env.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/type_support/meta.hpp>

#include <utility>

namespace hpx::execution::experimental {
    // 10.8.5.3. execution::on [exec.on]
    // 1. execution::on is used to adapt a sender into a sender that will
    //    start the input sender on an execution agent belonging to a
    //    specific execution context.
    // 2. Let replace-scheduler(e, sch) be an expression denoting an object
    //    e' such that execution::getScheduler(e) returns a copy of sch,
    //    and tag_invoke(tag, e', args...) is expression-equivalent to
    //    tag(e, args...) for all arguments args... and for all tag whose
    //    type satisfies forwarding-env-query and is not
    //    execution::getScheduler_t.
    // 3. The name execution::on denotes a customization point object. For
    //    some subexpressions sch and s, let Sch be decltype((sch)) and S
    //    be decltype((s)). If Sch does not satisfy execution::scheduler,
    //    or S does not satisfy execution::sender, execution::on is
    //    ill-formed. Otherwise, the expression execution::on(sch, s) is
    //    expression-equivalent to:
    //
    //     1. tag_invoke(execution::on, sch, s), if that expression is
    //        valid. If the function selected above does not return a
    //        sender which starts s on an execution agent of the associated
    //        execution context of sch when started, the behavior of
    //        calling execution::on(sch, s) is undefined. Mandates: The
    //        type of the tag_invoke expression above satisfies
    //        execution::sender.
    //     2. Otherwise, constructs a sender s1. When s1 is connected with
    //        some receiver out_r, it:
    //         1. Constructs a receiver r such that:
    //             1. When execution::set_value(r) is called, it calls
    //                execution::connect(s, r2), where r2 is as specified
    //                below, which results in op_state3. It calls
    //                execution::start(op_state3). If any of these throws
    //                an exception, it calls execution::set_error on out_r,
    //                passing current_exception() as the second argument.
    //             2. execution::set_error(r, e) is expression-equivalent
    //                to execution::set_error(out_r, e).
    //             3. execution::set_stopped(r) is expression-equivalent to
    //                execution::set_stopped(out_r).
    //             4. execution::get_env(r) is expression-equivalent to
    //                execution::get_env(out_r).
    //         2. Calls execution::schedule(sch), which results in s2. It
    //            then calls execution::connect(s2, r), resulting in
    //            op_state2.
    //         3. op_state2 is wrapped by a new operation state, op_state1,
    //            that is returned to the caller.
    //         4. r2 is a receiver that wraps a reference to out_r and
    //            forwards all receiver completion-signals to it. In
    //            addition, execution::get_env(r2) returns
    //            replace-scheduler(e, sch).
    //         5. When execution::start is called on op_state1, it calls
    //            execution::start on op_state2.
    //         6. The lifetime of op_state2, once constructed, lasts until
    //            either op_state3 is constructed or op_state1 is
    //            destroyed, whichever comes first. The lifetime of
    //            op_state3, once constructed, lasts until op_state1 is
    //            destroyed.
    //
    //     3. Given subexpressions s1 and e, where s1 is a sender returned
    //        from on or a copy of such, let S1 be decltype((s1)). Let E'
    //        be decltype((replace-scheduler(e, sch))). Then the type of
    //        tag_invoke(get_completion_signatures, s1, e) shall be:
    //        make_completion_signatures<
    //          copy_cvref_t<S1, S>,E',
    //          make_completion_signatures<
    //          schedule_result_t<Sch>,E,
    //          completion_signatures<set_error_t(exception_ptr)>,
    //          no-value-completions>>;
    //
    // where no-value-completions<As...> names the type
    // completion_signatures<> for any set of types As...

    namespace detail {
        template <typename SchedulerId, typename SenderId, typename ReceiverId>
        struct on_operation;

        template <typename SchedulerId, typename SenderId, typename ReceiverId>
        struct Receiver_ref
        {
            using Scheduler = hpx::meta::type<SchedulerId>;
            using Sender = hpx::meta::type<SenderId>;
            using on_receiver = hpx::meta::type<ReceiverId>;

            struct type
            {
                using id = Receiver_ref;
                hpx::meta::type<
                    on_operation<SchedulerId, SenderId, ReceiverId>>* op_state;

                on_receiver&& base() && noexcept
                {
                    return HPX_FORWARD(on_receiver, op_state->rcvr);
                }

                const on_receiver& base() const& noexcept
                {
                    return op_state->rcvr;
                }

                friend auto tag_invoke(get_env_t, type&& r)
                    -> env_of_t<std::decay_t<on_receiver>>
                {
                    return hpx::execution::experimental::get_env(r.base());
                }
            };
        };

        template <typename SchedulerId, typename SenderId, typename ReceiverId>
        struct on_receiver
        {
            using Scheduler = hpx::meta::type<SchedulerId>;
            using Sender = hpx::meta::type<SenderId>;
            using Receiver = hpx::meta::type<ReceiverId>;

            struct type
            {
                using id = on_receiver;
                using Receiver_ref_t = hpx::meta::type<
                    Receiver_ref<SchedulerId, SenderId, ReceiverId>>;
                hpx::meta::type<
                    on_operation<SchedulerId, SenderId, ReceiverId>>* op_state;

                Receiver&& base() && noexcept
                {
                    return HPX_FORWARD(Receiver, op_state->rcvr);
                }

                const Receiver& base() const& noexcept
                {
                    return op_state->rcvr;
                }

                void set_value() && noexcept
                {
                    // cache this locally since *this is going bye-bye.
                    auto* op_state_local = op_state;
                    try
                    {
                        // This line will invalidate *this:
                        start(op_state_local->data.template emplace<1>(
                            [op_state_local] {
                                return connect(
                                    HPX_FORWARD(Sender, op_state_local->sndr),
                                    Receiver_ref_t{{}, op_state_local});
                            }));
                    }
                    catch (...)
                    {
                        set_error(HPX_FORWARD(Receiver, op_state_local->rcvr),
                            std::current_exception());
                    }
                }
            };
        };

        template <typename SchedulerId, typename SenderId, typename ReceiverId>
        struct on_operation
        {
            using Scheduler = hpx::meta::type<SchedulerId>;
            using Sender = hpx::meta::type<SenderId>;
            using Receiver = hpx::meta::type<ReceiverId>;

            struct type
            {
                using id = on_operation;
                using Receiver_t = hpx::meta::type<
                    on_receiver<SchedulerId, SenderId, ReceiverId>>;
                using Receiver_ref_t = hpx::meta::type<
                    Receiver_ref<SchedulerId, SenderId, ReceiverId>>;

                friend void tag_invoke(start_t, type& self) noexcept
                {
                    start(std::get<0>(self.data));
                }

                template <typename Sender2, typename Receiver2>
                type(Scheduler sched, Sender2&& sndr, Receiver2&& rcvr)
                  : scheduler(HPX_FORWARD(Scheduler, sched))
                  , sndr(HPX_FORWARD(Sender2, sndr))
                  , rcvr(HPX_FORWARD(Receiver2, rcvr))
                  , data{std::in_place_index<0>, [this] {
                             return connect(
                                 schedule(scheduler), Receiver_t{{}, this});
                         }}
                {
                }

                Scheduler scheduler;
                Sender sndr;
                Receiver rcvr;
                std::variant<
                    connect_result_t<schedule_result_t<Scheduler>, Receiver_t>,
                    connect_result_t<Sender, Receiver_ref_t>>
                    data;
            };
        };

        template <typename SchedulerId, typename SenderId>
        struct on_sender
        {
            using Scheduler = hpx::meta::type<SchedulerId>;
            using Sender = hpx::meta::type<SenderId>;

            struct type
            {
                using id = on_sender;
                using is_sender = void;

                template <typename ReceiverId>
                using Receiver_ref_t = hpx::meta::type<
                    Receiver_ref<SchedulerId, SenderId, ReceiverId>>;
                template <typename ReceiverId>
                using Receiver_t = hpx::meta::type<
                    on_receiver<SchedulerId, SenderId, ReceiverId>>;
                template <typename ReceiverId>
                using on_operation_t = hpx::meta::type<
                    on_operation<SchedulerId, SenderId, ReceiverId>>;

                Scheduler scheduler;
                Sender sndr;

                template <typename Self, typename on_receiver,
                    typename = std::enable_if_t<is_receiver_v<on_receiver>>,
                    typename = std::enable_if_t<
                        is_sender_to_v<schedule_result_t<Scheduler>,
                            Receiver_t<hpx::meta::get_id_t<on_receiver>>>>>
                friend auto tag_invoke(connect_t, Self&& self, on_receiver rcvr)
                    -> on_operation_t<hpx::meta::get_id_t<on_receiver>>
                {
                    return {(HPX_FORWARD(Self, self)).scheduler,
                        (HPX_FORWARD(Self, self)).sndr, (on_receiver &&) rcvr};
                }

                friend auto tag_invoke(get_env_t, const type& self) noexcept(
                    hpx::is_nothrow_invocable_v<get_env_t, const Sender&>)
                    -> hpx::util::invoke_result_t<get_env_t, const Sender&>
                {
                    return get_env(self.sndr);
                }

                template <typename Self, typename Env>
                friend auto tag_invoke(get_completion_signatures_t, Self&&, Env)
                    -> make_completion_signatures<schedule_result_t<Scheduler>,
                        Env,
                        make_completion_signatures<Self,
                            make_env_t<get_scheduler_t, Scheduler, Env>,
                            completion_signatures<set_error_t(
                                std::exception_ptr)>>>;
            };
        };
    }    // namespace detail

    inline constexpr struct on_t : hpx::functional::detail::tag_fallback<on_t>
    {
        template <typename Scheduler, typename Sender,
            typename = std::enable_if_t<is_scheduler_v<Scheduler>>,
            typename = std::enable_if_t<is_sender_v<Sender>>>
        friend auto tag_fallback_invoke(on_t, Scheduler&& sched, Sender&& sndr)
            -> hpx::meta::type<
                detail::on_sender<hpx::meta::get_id_t<std::decay_t<Scheduler>>,
                    hpx::meta::get_id_t<std::decay_t<Sender>>>>
        {
            // connect-based customization will remove the need for this check
            // using has_customizations =
            //     hpx::util::invoke_result_t<has_algorithm_customizations_t, Scheduler>;
            // static_assert(!has_customizations{},
            //     "For now the default stdexec::on implementation doesn't "
            //     "support scheduling "
            //     "onto schedulers that customize algorithms.");
            return {HPX_FORWARD(Scheduler, sched), HPX_FORWARD(Sender, sndr)};
        }
    } on{};

}    // namespace hpx::execution::experimental
