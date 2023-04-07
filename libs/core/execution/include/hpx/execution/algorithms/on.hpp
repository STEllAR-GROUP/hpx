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

    namespace adaptors {

        HPX_HAS_XXX_TRAIT_DEF(set_value)
        HPX_HAS_XXX_TRAIT_DEF(set_error)
        HPX_HAS_XXX_TRAIT_DEF(set_stopped)
        HPX_HAS_XXX_TRAIT_DEF(get_env)

        // A derived-to-base cast that works even when the base is not
        // accessible from derived.
        template <typename T, typename U,
            typename = std::enable_if_t<std::is_same_v<std::decay_t<T>, T>>>
        hpx::meta::copy_cvref_t<U&&, T> c_cast(U&& u) noexcept
        {
            static_assert(std::is_reference_v<hpx::meta::copy_cvref_t<U&&, T>>);
            static_assert(std::is_base_of_v<T, std::remove_reference_t<U>>);
            return (hpx::meta::copy_cvref_t<U&&, T>) HPX_FORWARD(U, u);
        }

        namespace no {
            struct nope
            {
            };

            struct receiver : nope
            {
            };

            void tag_invoke(set_error_t, receiver, std::exception_ptr) noexcept;
            void tag_invoke(set_stopped_t, receiver) noexcept;
            empty_env tag_invoke(get_env_t, receiver) noexcept;
        }    // namespace no

        using not_a_receiver = no::receiver;

        template <typename Base, typename = void>
        struct adaptor
        {
            struct type
            {
                type() = default;

                template <typename T1,
                    typename = std::enable_if_t<
                        hpx::meta::is_constructible_from_v<Base, T1>>>
                explicit type(T1&& _base)
                  : base_((T1 &&) _base)
                {
                }

            private:
                HPX_NO_UNIQUE_ADDRESS Base base_;

            protected:
                Base& base() & noexcept
                {
                    return base_;
                }

                const Base& base() const& noexcept
                {
                    return base_;
                }

                Base&& base() && noexcept
                {
                    return HPX_FORWARD(Base, base_);
                }
            };
        };

        template <typename Base>
        struct adaptor<Base,
            std::enable_if_t<hpx::meta::is_derived_from_v<no::nope, Base>>>
        {
            struct type : no::nope
            {
            };
        };

        template <typename Base>
        using adaptor_base = typename adaptor<Base>::type;

        template <typename...>
        inline constexpr bool __true = true;

        template <typename _Cp>
        inline constexpr bool __class = __true<int _Cp::*> &&
            (!std::is_same_v<const _Cp, _Cp>);

        template <typename Derived, typename Base,
            typename = std::enable_if_t<__class<Derived>>>
        struct receiver_adaptor
        {
            class type : adaptor_base<Base>
            {
                friend Derived;

                template <typename Self, typename... Ts>
                static auto call_set_value(Self&& self, Ts&&... ts) noexcept(
                    noexcept(((Self &&) self).set_value((Ts &&) ts...)))
                    -> decltype(((Self &&) self).set_value((Ts &&) ts...))
                {
                    return ((Self &&) self).set_value((Ts &&) ts...);
                }
                using set_value = void;

                template <typename Self, typename... Ts>
                static auto call_set_error(Self&& self, Ts&&... ts) noexcept(
                    noexcept(((Self &&) self).set_error((Ts &&) ts...)))
                    -> decltype(((Self &&) self).set_error((Ts &&) ts...))
                {
                    return ((Self &&) self).set_error((Ts &&) ts...);
                }
                using set_error = void;

                template <typename Self, typename... Ts>
                static auto call_set_stopped(Self&& self, Ts&&... ts) noexcept(
                    noexcept(((Self &&) self).set_stopped((Ts &&) ts...)))
                    -> decltype(((Self &&) self).set_stopped((Ts &&) ts...))
                {
                    return ((Self &&) self).set_stopped((Ts &&) ts...);
                }
                using set_stopped = void;

                template <typename Self, typename... Ts>
                static auto call_get_env(Self&& self, Ts&&... ts) noexcept(
                    noexcept(((Self &&) self).get_env((Ts &&) ts...)))
                    -> decltype(((Self &&) self).get_env((Ts &&) ts...))
                {
                    return ((Self &&) self).get_env((Ts &&) ts...);
                }
                using get_env = void;

                static inline constexpr bool has_base_v =
                    !hpx::meta::is_derived_from_v<Base, no::nope>;

                template <typename T>
                using base_from_derived_t = decltype(std::declval<T>().base());

                using get_base_t = hpx::meta::if_<
                    std::integral_constant<bool, has_base_v>,
                    hpx::meta::bind_back1_func<hpx::meta::copy_cvref_t, Base>,
                    hpx::meta::func1<base_from_derived_t>>;

                template <typename D>
                using base_t = hpx::meta::invoke1<get_base_t, D&&>;

                template <typename D>
                static base_t<D> get_base(D&& self) noexcept
                {
                    if constexpr (has_base_v)
                    {
                        return c_cast<type>((D &&) self).base();
                    }
                    else
                    {
                        return ((D &&) self).base();
                    }
                }

                template <typename SetValue,
                    typename std::enable_if_t<
                        std::is_same_v<std::decay_t<SetValue>, set_value_t> &&
                        has_set_value_v<Derived>>,
                    typename... As>
                friend auto tag_invoke(
                    SetValue, Derived&& self, As&&... as) noexcept
                    -> decltype(call_set_value(
                        (Derived &&) self, (As &&) as...))
                {
                    static_assert(noexcept(
                        call_set_value((Derived &&) self, (As &&) as...)));
                    call_set_value((Derived &&) self, (As &&) as...);
                }

                template <typename SetValue, typename D = Derived,
                    typename... As,
                    typename = std::enable_if_t<
                        std::is_same_v<std::decay_t<SetValue>, set_value_t> &&
                        hpx::util::all_of_v<std::integral_constant<bool,
                            ((!has_set_value_v<D>) &&hpx::functional::
                                    is_tag_invocable_v<set_value_t, base_t<D>,
                                        As>)>...>>>
                friend void tag_invoke(
                    SetValue, Derived&& self, As&&... as) noexcept
                {
                    hpx::execution::experimental::set_value(
                        get_base((D &&) self), (As &&) as...);
                }

                template <typename SetError, typename Error,
                    typename = std::enable_if_t<
                        std::is_same_v<std::decay_t<SetError>, set_error_t> &&
                        has_set_error_v<Derived>>>
                friend auto tag_invoke(
                    SetError, Derived&& self, Error&& err) noexcept
                    -> decltype(call_set_error(
                        (Derived &&) self, (Error &&) err))
                {
                    static_assert(noexcept(
                        call_set_error((Derived &&) self, (Error &&) err)));
                    call_set_error((Derived &&) self, (Error &&) err);
                }

                template <typename SetError, typename Error,
                    typename D = Derived,
                    typename = std::enable_if_t<
                        std::is_same_v<std::decay_t<SetError>, set_error_t> &&
                        (!has_set_error_v<D>) &&hpx::functional::
                            is_tag_invocable_v<set_error_t, base_t<D>>>>
                friend void tag_invoke(
                    SetError, Derived&& self, Error&& err) noexcept
                {
                    hpx::execution::experimental::set_error(
                        get_base((Derived &&) self), (Error &&) err);
                }

                template <typename SetStopped, typename D = Derived,
                    typename = std::enable_if_t<
                        std::is_same_v<std::decay_t<SetStopped>,
                            set_stopped_t> &&
                        has_set_stopped_v<D>>>
                friend auto tag_invoke(SetStopped, Derived&& self) noexcept
                    -> decltype(call_set_stopped((D &&) self))
                {
                    static_assert(
                        noexcept(call_set_stopped((Derived &&) self)));
                    call_set_stopped((Derived &&) self);
                }

                template <typename SetStopped, typename D = Derived,
                    typename = std::enable_if_t<
                        std::is_same_v<std::decay_t<SetStopped>,
                            set_stopped_t> &&
                        (!has_set_stopped_v<D>) &&hpx::functional::
                            is_tag_invocable_v<set_stopped_t, base_t<D>>>>
                friend void tag_invoke(SetStopped, Derived&& self) noexcept
                {
                    hpx::execution::experimental::set_stopped(
                        get_base((Derived &&) self));
                }

                // Pass through the get_env receiver query
                template <typename GetEnv, typename D = Derived,
                    typename = std::enable_if_t<
                        std::is_same_v<std::decay_t<GetEnv>, get_env_t> &&
                        has_get_env_v<D>>>
                friend auto tag_invoke(GetEnv, const Derived& self)
                    -> decltype(call_get_env((const D&) self))
                {
                    return call_get_env(self);
                }

                template <typename GetEnv, typename D = Derived,
                    typename = std::enable_if_t<
                        std::is_same_v<std::decay_t<GetEnv>, get_env_t> &&
                        (!has_get_env_v<D>)>>
                friend auto tag_invoke(GetEnv, const Derived& self)
                    -> hpx::util::invoke_result_t<get_env_t, base_t<const D&>>
                {
                    return hpx::execution::experimental::get_env(
                        get_base(self));
                }

            public:
                type() = default;
                using adaptor_base<Base>::adaptor_base;

                using is_receiver = void;
            };
        };
    }    // namespace adaptors

    template <typename Derived, typename Base = adaptors::not_a_receiver,
        typename =
            std::enable_if_t<is_receiver_v<Base> && adaptors::__class<Derived>>>
    using receiver_adaptor =
        typename adaptors::receiver_adaptor<Derived, Base>::type;

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

        template <typename Fn,
            typename =
                std::enable_if_t<std::is_nothrow_move_constructible_v<Fn>>>
        struct conv
        {
            Fn fn;
            using type = hpx::util::invoke_result_t<Fn>;

            operator type() && noexcept(hpx::is_nothrow_invocable_v<Fn>)
            {
                return HPX_FORWARD(Fn, fn)();
            }

            type operator()() && noexcept(hpx::is_nothrow_invocable_v<Fn>)
            {
                return HPX_FORWARD(Fn, fn)();
            }
        };

        template <typename Fn>
        conv(Fn) -> conv<Fn>;

        template <typename SchedulerId, typename SenderId, typename ReceiverId>
        struct on_operation;

        template <typename SchedulerId, typename SenderId, typename ReceiverId>
        struct on_receiver_ref
        {
            using Scheduler = hpx::meta::type<SchedulerId>;
            using Sender = hpx::meta::type<SenderId>;
            using on_receiver_t = hpx::meta::type<ReceiverId>;

            struct type : receiver_adaptor<type>
            {
                using id = on_receiver_ref;
                hpx::meta::type<
                    on_operation<SchedulerId, SenderId, ReceiverId>>* op_state;

                on_receiver_t&& base() && noexcept
                {
                    return HPX_FORWARD(on_receiver_t, op_state->rcvr);
                }

                const on_receiver_t& base() const& noexcept
                {
                    return op_state->rcvr;
                }

                auto get_env() const -> make_env_t<get_scheduler_t, Scheduler,
                    env_of_t<std::decay_t<on_receiver_t>>>
                {
                    return hpx::execution::experimental::get_env(this->base());
                }
            };
        };

        template <typename SchedulerId, typename SenderId, typename ReceiverId>
        struct on_receiver
        {
            using Scheduler = hpx::meta::type<SchedulerId>;
            using Sender = hpx::meta::type<SenderId>;
            using Receiver = hpx::meta::type<ReceiverId>;

            struct type : receiver_adaptor<type>
            {
                using id = on_receiver;
                using on_receiver_ref_t = hpx::meta::type<
                    on_receiver_ref<SchedulerId, SenderId, ReceiverId>>;
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
                            conv{[op_state_local] {
                                return connect(
                                    HPX_FORWARD(Sender, op_state_local->sndr),
                                    on_receiver_ref_t{{}, op_state_local});
                            }}));
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
                using on_receiver_t = hpx::meta::type<
                    on_receiver<SchedulerId, SenderId, ReceiverId>>;
                using on_receiver_ref_t = hpx::meta::type<
                    on_receiver_ref<SchedulerId, SenderId, ReceiverId>>;

                friend void tag_invoke(start_t, type& self) noexcept
                {
                    start(std::get<0>(self.data));
                }

                template <typename Sender2, typename Receiver2>
                type(Scheduler sched, Sender2&& sndr, Receiver2&& rcvr)
                  : scheduler(HPX_FORWARD(Scheduler, sched))
                  , sndr(HPX_FORWARD(Sender2, sndr))
                  , rcvr(HPX_FORWARD(Receiver2, rcvr))
                  , data{std::in_place_index<0>, conv{[this] {
                             return connect(
                                 schedule(scheduler), on_receiver_t{{}, this});
                         }}}
                {
                }

                type(type&&) = delete;

                Scheduler scheduler;
                Sender sndr;
                Receiver rcvr;
                std::variant<connect_result_t<schedule_result_t<Scheduler>,
                                 on_receiver_t>,
                    connect_result_t<Sender, on_receiver_ref_t>>
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
                using on_receiver_ref_t = hpx::meta::type<
                    on_receiver_ref<SchedulerId, SenderId, ReceiverId>>;
                template <typename ReceiverId>
                using on_receiver_t = hpx::meta::type<
                    on_receiver<SchedulerId, SenderId, ReceiverId>>;
                template <typename ReceiverId>
                using on_operation_t = hpx::meta::type<
                    on_operation<SchedulerId, SenderId, ReceiverId>>;

                Scheduler scheduler;
                Sender sndr;

                friend auto tag_invoke(get_env_t, const type& self) noexcept(
                    hpx::is_nothrow_invocable_v<get_env_t, const Sender&>)
                    -> hpx::util::invoke_result_t<get_env_t, const Sender&>
                {
                    return get_env(self.sndr);
                }

                template <typename...>
                using value_types = completion_signatures<>;

                // template <typename Sender, typename Env>
                // struct generate_completion_signatures
                // {
                //     template <typename...>
                //     using value_types =
                //         detail::value_types_of<schedule_result_t<Scheduler>,
                //             Env>;

                //     template <template <typename...> typename Variant>
                //     using error_types =
                //         completion_signatures<set_error_t(std::exception_ptr)>;

                //     static constexpr bool sends_stopped = false;
                // };

                template <typename Self, typename Env,
                    std::enable_if_t<std::is_same_v<std::decay_t<Self>, type>>>
                friend auto tag_invoke(get_completion_signatures_t, Self&&, Env)
                    // -> generate_completion_signatures<
                    //     schedule_result_t<Scheduler>, Env>;
                    -> make_completion_signatures<schedule_result_t<Scheduler>,
                        Env,
                        make_completion_signatures<
                            hpx::meta::copy_cvref_t<Self, Sender>,
                            make_env_t<get_scheduler_t, Scheduler, Env>,
                            completion_signatures<set_error_t(
                                std::exception_ptr)>>,
                        value_types>;

                // clang-format off
                template <typename Self, typename ReceiverOn,
                    HPX_CONCEPT_REQUIRES_(
                        is_receiver_v<ReceiverOn>
                        && std::is_same_v<std::decay_t<Self>, type>
                        && hpx::meta::is_constructible_from_v<Sender,
                                hpx::meta::copy_cvref_t<Self, Sender>>
                        && is_sender_to_v<schedule_result_t<Scheduler>,
                            on_receiver_t<hpx::meta::get_id_t<ReceiverOn>>>
                        && is_sender_to_v<Sender, on_receiver_ref_t<
                                        hpx::meta::get_id_t<ReceiverOn>>>
                    )>
                // clang-format on
                friend auto tag_invoke(connect_t, Self&& self, ReceiverOn rcvr)
                    -> on_operation_t<hpx::meta::get_id_t<ReceiverOn>>
                {
                    return {(HPX_FORWARD(Self, self)).scheduler,
                        (HPX_FORWARD(Self, self)).sndr,
                        HPX_FORWARD(ReceiverOn, rcvr)};
                }
            };
        };

    }    // namespace detail

    inline constexpr struct on_t : hpx::functional::detail::tag_fallback<on_t>
    {
        template <typename Scheduler, typename Sender,
            typename = std::enable_if_t<is_scheduler_v<Scheduler> &&
                is_sender_v<Sender>>>
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
