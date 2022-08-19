//  Copyright (c) 2022 Shreyas Atre
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/assert.hpp>
#include <hpx/datastructures/variant.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/get_env.hpp>
#include <hpx/execution_base/traits/coroutine_traits.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/type_support/meta.hpp>

#include <exception>
#include <system_error>
#include <utility>

#if defined(HPX_HAVE_CXX20_COROUTINES)

namespace hpx::execution::experimental {

    template <typename Sender, typename Receiver>
    inline constexpr bool has_nothrow_connect = noexcept(
        connect(std::declval<Sender>(), std::declval<Receiver>()));

    // 4.18. Cancellation of a sender can unwind a stack of coroutines
    // As described in the section "All awaitables are senders", the sender
    // customization points recognize awaitables and adapt them transparently
    // to model the sender concept. When connect-ing an awaitable and a
    // receiver, the adaptation layer awaits the awaitable within a coroutine
    // that implements unhandled_stopped in its promise type. The effect of
    // this is that an "uncatchable" stopped exception propagates seamlessly
    // out of awaitables, causing execution::set_stopped to be called on the
    // receiver.
    // Obviously, unhandled_stopped is a library extension of the
    // coroutine promise interface. Many promise types will not implement
    // unhandled_stopped. When an uncatchable stopped exception tries to
    // propagate through such a coroutine, it is treated as an unhandled
    // exception and terminate is called. The solution, as described above, is
    // to use a sender adaptor to handle the stopped exception before awaiting
    // it. It goes without saying that any future Standard Library coroutine
    // types ought to implement unhandled_stopped. The author of Add lazy
    // coroutine (coroutine task) type, which proposes a standard coroutine
    // task type, is in agreement.
    template <typename Promise, typename = void>
    inline constexpr bool has_unhandled_stopped = false;

    template <typename Promise>
    inline constexpr bool has_unhandled_stopped<Promise,
        std::void_t<decltype(std::declval<Promise>().unhandled_stopped())>> =
        true;

    template <typename Promise, typename = void>
    inline constexpr bool has_convertible_unhandled_stopped = false;

    template <typename Promise>
    inline constexpr bool has_convertible_unhandled_stopped<Promise,
        std::enable_if_t<std::is_convertible_v<
            decltype(std::declval<Promise>().unhandled_stopped()),
            hpx::coro::coroutine_handle<>>>> = true;

    // execution::as_awaitable [execution.coro_utils.as_awaitable]
    // as_awaitable is used to transform an object into one that is
    // awaitable within a particular coroutine.
    //
    // as_awaitable is a customization point object.
    // For some subexpressions e and p where p is an lvalue,
    // E names the type decltype((e)) and P names the type decltype((p)),
    // as_awaitable(e, p) is expression-equivalent to the following:
    //
    //      1. tag_invoke(as_awaitable, e, p)
    //         if that expression is well-formed.
    //         -- Mandates: is-awaitable<A> is true,
    //          where A is the type of the tag_invoke expression above.
    //      2. Otherwise, e if is-awaitable<E> is true.
    //      3. Otherwise, sender-awaitable{e, p} if awaitable-sender<E, P>
    //      is true.
    //      4. Otherwise, e.
    struct as_awaitable_t;

    namespace detail {
        template <typename T, typename U>
        inline constexpr bool decays_to = std::is_same_v<std::decay_t<T>, U>&&
            std::is_same_v<std::decay_t<U>, T>;

        struct void_type
        {
        };

        template <typename Value>
        using value_or_void_t =
            hpx::meta::if_<std::is_same<Value, void>, void_type, Value>;

        template <typename Value>
        using expected_t = hpx::variant<hpx::monostate, value_or_void_t<Value>,
            std::exception_ptr>;

        template <typename Value>
        struct receiver_base : hpx::functional::tag<receiver_base<Value>>
        {
            template <typename... Us,
                HPX_CONCEPT_REQUIRES_(
                    std::is_constructible_v<value_or_void_t<Value>, Us...>)>
            friend void tag_invoke(
                set_value_t, receiver_base&& self, Us&&... us) noexcept
            try
            {
                self.result->template emplace<1>(HPX_FORWARD(Us, us)...);
                self.continuation.resume();
            }
            catch (...)
            {
                set_error(
                    HPX_FORWARD(receiver_base, self), std::current_exception());
            }

            template <typename Error>
            friend void tag_invoke(
                set_error_t, receiver_base&& self, Error&& err) noexcept
            {
                if constexpr (decays_to<Error, std::exception_ptr>)
                    self.result->template emplace<2>(HPX_FORWARD(Error, err));
                else if constexpr (decays_to<Error, std::error_code>)
                    self.result->template emplace<2>(
                        make_exception_ptr(system_error(err)));
                else
                    self.result->template emplace<2>(
                        make_exception_ptr(HPX_FORWARD(Error, err)));
                self.continuation.resume();
            }

            expected_t<Value>* result;
            hpx::coro::coroutine_handle<> continuation;
        };

        template <typename PromiseId, typename Value>
        struct sender_awaitable_base
        {
            using Promise = hpx::meta::hidden<PromiseId>;
            struct receiver : receiver_base<Value>
            {
                friend void tag_invoke(set_stopped_t, receiver&& self) noexcept
                {
                    auto continuation =
                        hpx::coro::coroutine_handle<Promise>::from_address(
                            self.continuation.address());
                    hpx::coro::coroutine_handle<> stopped_continuation =
                        continuation.promise().unhandled_stopped();
                    stopped_continuation.resume();
                }

                // Forward get_env query to the coroutine promise
                friend auto tag_invoke(get_env_t, const receiver& self)
                    -> env_of_t<Promise>
                {
                    auto continuation =
                        hpx::coro::coroutine_handle<Promise>::from_address(
                            self.continuation.address());
                    return get_env(continuation.promise());
                }
            };

            bool await_ready() const noexcept
            {
                return false;
            }

            Value await_resume()
            {
                switch (result.index())
                {
                case 0:    // receiver contract not satisfied
                    HPX_ASSERT_MSG(0, "_Should never get here");
                    break;
                case 1:    // set_value
                    if constexpr (!std::is_void_v<Value>)
                        return HPX_FORWARD(Value, std::get<1>(result));
                    else
                        return;
                case 2:    // set_error
                    std::rethrow_exception(std::get<2>(result));
                }
                std::terminate();
            }

        protected:
            expected_t<Value> result;
        };

        template <typename PromiseId, typename SenderId>
        struct sender_awaitable
          : sender_awaitable_base<PromiseId,
                single_sender_value_t<hpx::meta::hidden<SenderId>,
                    env_of_t<hpx::meta::hidden<PromiseId>>>>
        {
        private:
            using Promise = hpx::meta::hidden<PromiseId>;
            using Sender = hpx::meta::hidden<SenderId>;
            using Env = env_of_t<Promise>;
            using Value = single_sender_value_t<Sender, Env>;
            using Base = sender_awaitable_base<PromiseId, Value>;
            using receiver = typename Base::receiver;
            connect_result_t<Sender, receiver> op_state_;

        public:
            sender_awaitable(Sender&& sender,
                hpx::coro::coroutine_handle<Promise>
                    hcoro) noexcept(has_nothrow_connect<Sender, receiver>)
              : op_state_(connect(HPX_FORWARD(Sender, sender),
                    receiver{{&this->result, hcoro}}))
            {
            }

            void await_suspend(hpx::coro::coroutine_handle<Promise>) noexcept
            {
                start(op_state_);
            }
        };
        template <typename Promise, typename Sender>
        using sender_awaitable_t = sender_awaitable<hpx::meta::hidden<Promise>,
            hpx::meta::hidden<Sender>>;

        template <typename T, typename Promise>
        inline constexpr bool is_custom_tag_invoke_awaiter_v =
            hpx::functional::is_tag_invocable_v<as_awaitable_t, T, Promise&>&&
                is_awaitable_v<hpx::functional::tag_invoke_result_t<
                                   as_awaitable_t, T, Promise&>,
                    Promise>;

        template <typename Sender, typename Promise>
        using receiver =
            typename sender_awaitable_base<hpx::meta::hidden<Promise>,
                single_sender_value_t<Sender, env_of_t<Promise>>>::receiver;

        template <typename Sender, typename Env = no_env>
        inline constexpr bool is_single_typed_sender_v =
            is_sender_v<Sender, Env>&& hpx::meta::value<
                hpx::meta::is_valid<single_sender_value_t, Sender, Env>>;

        template <typename Sender, typename Promise>
        inline constexpr bool is_awaitable_sender_v =
            is_single_typed_sender_v<Sender, env_of_t<Promise>>&&
                is_sender_to_v<Sender, receiver<Sender, Promise>>&&
                    has_unhandled_stopped<Promise>;

    }    // namespace detail

    struct as_awaitable_t
    {
        template <typename T, typename Promise>
        static constexpr bool is_noexcept() noexcept
        {
            if constexpr (detail::is_custom_tag_invoke_awaiter_v<T, Promise>)
            {
                return hpx::functional::is_nothrow_tag_invocable_v<
                    as_awaitable_t, T, Promise&>;
            }
            else if constexpr (is_awaitable_v<T, Promise>)
            {
                return true;
            }
            else if constexpr (detail::is_awaitable_sender_v<T, Promise>)
            {
                using Sender = detail::sender_awaitable_t<Promise, T>;
                return std::is_nothrow_constructible_v<Sender, T,
                    hpx::coro::coroutine_handle<Promise>>;
            }
            else
            {
                return true;
            }
        }

        template <typename T, typename Promise>
        decltype(auto) operator()(T&& t, Promise& promise) const
            noexcept(is_noexcept<T, Promise>())
        {
            if constexpr (detail::is_custom_tag_invoke_awaiter_v<T, Promise>)
            {
                return tag_invoke(*this, HPX_FORWARD(T, t), promise);
            }
            else if constexpr (is_awaitable_v<T, Promise>)
            {
                return HPX_FORWARD(T, t);
            }
            else if constexpr (detail::is_awaitable_sender_v<T, Promise>)
            {
                auto hcoro =
                    hpx::coro::coroutine_handle<Promise>::from_promise(promise);
                return detail::sender_awaitable_t<Promise, T>{
                    HPX_FORWARD(T, t), hcoro};
            }
            else
            {
                return HPX_FORWARD(T, t);
            }
        }
    };

    inline constexpr as_awaitable_t as_awaitable;

    namespace detail {
        struct with_awaitable_senders_base
        {
            template <typename OtherPromise>
            void set_continuation(
                hpx::coro::coroutine_handle<OtherPromise> hcoro) noexcept
            {
                static_assert(!std::is_void_v<OtherPromise>);
                continuation_handle = hcoro;
                if constexpr (has_unhandled_stopped<OtherPromise>)
                {
                    stopped_callback = [](void* address) noexcept
                        -> hpx::coro::coroutine_handle<> {
                        // This causes the rest of the coroutine (the part after the co_await
                        // of the sender) to be skipped and invokes the calling coroutine's
                        // stopped handler.
                        return hpx::coro::coroutine_handle<
                            OtherPromise>::from_address(address)
                            .promise()
                            .unhandled_stopped();
                    };
                }
                // If OtherPromise doesn't implement unhandled_stopped(), then if a "stopped" unwind
                // reaches this point, it's considered an unhandled exception and terminate()
                // is called.
            }

            hpx::coro::coroutine_handle<> continuation() const noexcept
            {
                return continuation_handle;
            }

            hpx::coro::coroutine_handle<> unhandled_stopped() noexcept
            {
                return (*stopped_callback)(continuation_handle.address());
            }

        private:
            hpx::coro::coroutine_handle<> continuation_handle{};
            hpx::coro::coroutine_handle<> (*stopped_callback)(void*) noexcept =
                [](void*) noexcept -> hpx::coro::coroutine_handle<> {
                std::terminate();
            };
        };
    }    // namespace detail

    template <typename A, typename B>
    inline constexpr bool is_derived_from = std::is_base_of_v<B, A>&&
        std::is_convertible_v<const volatile A*, const volatile B*>;

    // with_awaitable_senders, when used as the base class of a coroutine
    // promise type, makes senders awaitable in that coroutine type. In
    // addition, it provides a default implementation of unhandled_stopped()
    // such that if a sender completes by calling execution::set_stopped, it
    // is treated as if an uncatchable "stopped" exception were thrown from
    // the await-expression. In practice, the coroutine is never resumed, and
    // the unhandled_stopped of the coroutine caller’s promise type is called.
    //
    template <typename Promise>
    struct with_awaitable_senders : detail::with_awaitable_senders_base
    {
        template <typename Value>
        auto await_transform(Value&& val)
            -> hpx::functional::tag_invoke_result_t<as_awaitable_t, Value,
                Promise&>
        {
            static_assert(is_derived_from<Promise, with_awaitable_senders>);
            return as_awaitable(
                HPX_FORWARD(Value, val), static_cast<Promise&>(*this));
        }
    };

    struct promise_base
    {
        hpx::coro::suspend_always initial_suspend() noexcept
        {
            return {};
        }
        [[noreturn]] hpx::coro::suspend_always final_suspend() noexcept
        {
            std::terminate();
        }
        [[noreturn]] void unhandled_exception() noexcept
        {
            std::terminate();
        }
        [[noreturn]] void return_void() noexcept
        {
            std::terminate();
        }
        template <typename Fun>
        auto yield_value(Fun&& fun) noexcept
        {
            struct awaiter
            {
                Fun&& fun;
                bool await_ready() noexcept
                {
                    return false;
                }
                void await_suspend(hpx::coro::coroutine_handle<>) noexcept(
                    std::is_nothrow_invocable_v<Fun>)
                {
                    // If this throws, the runtime catches the exception,
                    // resumes the connect_awaitable coroutine, and immediately
                    // rethrows the exception. The end result is that an
                    // exception_ptr to the exception gets passed to set_error.
                    (HPX_FORWARD(Fun, fun))();
                }
                [[noreturn]] void await_resume() noexcept
                {
                    std::terminate();
                }
            };
            return awaiter{HPX_FORWARD(Fun, fun)};
        }
    };

    struct operation_base
    {
        hpx::coro::coroutine_handle<> coro_handle;

        explicit operation_base(hpx::coro::coroutine_handle<> hcoro) noexcept
          : coro_handle(hcoro)
        {
        }

        operation_base(operation_base&& other) noexcept
          : coro_handle(std::exchange(other.coro_handle, {}))
        {
        }

        ~operation_base()
        {
            if (coro_handle)
                coro_handle.destroy();
        }

        friend void tag_invoke(start_t, operation_base& self) noexcept
        {
            self.coro_handle.resume();
        }
    };

    template <typename ReceiverId>
    struct promise;

    template <typename ReceiverId>
    struct operation : operation_base
    {
        using promise_type = promise<ReceiverId>;
        using operation_base::operation_base;
    };

    template <typename ReceiverId>
    struct promise
      : promise_base
      , hpx::functional::tag<promise<ReceiverId>>
    {
        using Receiver = hpx::meta::hidden<ReceiverId>;

        explicit promise(auto&, Receiver& rcvr_) noexcept
          : rcvr(rcvr_)
        {
        }

        hpx::coro::coroutine_handle<> unhandled_stopped() noexcept
        {
            set_stopped(std::move(rcvr));
            // Returning noop_coroutine here causes the __connect_awaitable
            // coroutine to never resume past the point where it co_await's
            // the awaitable.
            return hpx::coro::noop_coroutine();
        }

        operation<ReceiverId> get_return_object() noexcept
        {
            return operation<ReceiverId>{
                hpx::coro::coroutine_handle<promise>::from_promise(*this)};
        }

        template <typename Awaitable>
        Awaitable&& await_transform(Awaitable&& await) noexcept
        {
            return HPX_FORWARD(Awaitable, await);
        }

        template <typename Awaitable,
            typename = std::enable_if_t<
                hpx::is_invocable_v<as_awaitable_t, Awaitable, promise&>>>
        auto await_transform(Awaitable&& await) noexcept(
            hpx::functional::is_nothrow_tag_invocable_v<as_awaitable_t,
                Awaitable, promise&>)
            -> hpx::functional::tag_invoke_result_t<as_awaitable_t, Awaitable,
                promise&>
        {
            return tag_invoke(
                as_awaitable, HPX_FORWARD(Awaitable, await), *this);
        }

        // Pass through the get_env receiver query
        friend auto tag_invoke(get_env_t, const promise& self)
            -> env_of_t<Receiver>
        {
            return get_env(self.rcvr);
        }

        Receiver& rcvr;
    };

    template <typename Receiver,
        typename = std::enable_if_t<is_receiver_v<Receiver>>>
    using promise_t = promise<hpx::meta::hidden<Receiver>>;

    template <typename Receiver,
        typename = std::enable_if_t<is_receiver_v<Receiver>>>
    using operation_t = operation<hpx::meta::hidden<Receiver>>;

    inline constexpr struct connect_awaitable_t
    {
    private:
        template <typename Awaitable, typename Receiver>
        static operation_t<Receiver> impl(Awaitable await, Receiver rcvr)
        {
            using result_t = await_result_t<Awaitable, promise_t<Receiver>>;
            std::exception_ptr eptr;
            try
            {
                // This is a bit mind bending control-flow wise.
                // We are first evaluating the co_await expression.
                // Then the result of that is passed into a lambda
                // that curries a reference to the result into another
                // lambda which is then returned to 'co_yield'.
                // The 'co_yield' expression then invokes this lambda
                // after the coroutine is suspended so that it is safe
                // for the receiver to destroy the coroutine.
                auto fun = [&](auto&&... as) noexcept {
                    return [&]() noexcept -> void {
                        set_value(HPX_FORWARD(Receiver, rcvr),
                            (std::add_rvalue_reference_t<result_t>) as...);
                    };
                };
                if constexpr (std::is_void_v<result_t>)
                    co_yield(co_await HPX_FORWARD(Awaitable, await), fun());
                else
                    co_yield fun(co_await HPX_FORWARD(Awaitable, await));
            }
            catch (...)
            {
                eptr = std::current_exception();
            }
            co_yield [&]() noexcept -> void {
                set_error(HPX_FORWARD(Receiver, rcvr),
                    HPX_FORWARD(std::exception_ptr, eptr));
            };
        }

        template <typename Receiver, typename Awaitable,
            typename = std::enable_if_t<is_receiver_v<Receiver>>>
        using completions_t = completion_signatures<
            hpx::meta::invoke1<    // set_value_t() or set_value_t(T)
                hpx::meta::remove<void, hpx::meta::compose_func<set_value_t>>,
                await_result_t<Awaitable, promise_t<Receiver>>>,
            set_error_t(std::exception_ptr), set_stopped_t()>;

    public:
        template <typename Receiver, typename Awaitable,
            typename = std::enable_if_t<
                is_awaitable_v<Awaitable, promise_t<Receiver>>>,
            typename = std::enable_if_t<
                is_receiver_of_v<Receiver, completions_t<Receiver, Awaitable>>>>
        operation_t<Receiver> operator()(
            Awaitable&& await, Receiver&& rcvr) const
        {
            return impl(
                HPX_FORWARD(Awaitable, await), HPX_FORWARD(Receiver, rcvr));
        }
    } connect_awaitable{};

}    // namespace hpx::execution::experimental

#endif    // HPX_HAVE_CXX20_COROUTINES
