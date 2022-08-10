//  Copyright (c) 2022 Shreyas Atre
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

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

    namespace impl {
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
                self.result->template emplace<1>((Us &&) us...);
                self.continuation.resume();
            }
            catch (...)
            {
                set_error((receiver_base &&) self, std::current_exception());
            }

            template <typename Error>
            friend void tag_invoke(
                set_error_t, receiver_base&& self, Error&& err) noexcept
            {
                if constexpr (decays_to<Error, std::exception_ptr>)
                    self.result->template emplace<2>((Error &&) err);
                else if constexpr (decays_to<Error, std::error_code>)
                    self.result->template emplace<2>(
                        make_exception_ptr(system_error(err)));
                else
                    self.result->template emplace<2>(
                        make_exception_ptr((Error &&) err));
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
                        return (Value &&) std::get<1>(result);
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

        template <typename Sender, typename Env = no_env>
        using single_sender_value_t = value_types_of_t<Sender, Env>;

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
              : op_state_(connect(
                    (Sender &&) sender, receiver{{&this->result, hcoro}}))
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

        template <class Sender, class Env = no_env>
        inline constexpr bool is_single_typed_sender_v =
            is_sender_v<Sender, Env>&& hpx::meta::value<
                hpx::meta::is_valid<single_sender_value_t, Sender, Env>>;

        template <typename Sender, typename Promise>
        inline constexpr bool is_awaitable_sender_v =
            is_single_typed_sender_v<Sender, env_of_t<Promise>>&&
                is_sender_to_v<Sender, receiver<Sender, Promise>>&&
                    has_unhandled_stopped<Promise>;

    }    // namespace impl

    struct as_awaitable_t
    {
        template <typename T, typename Promise>
        static constexpr bool is_noexcept() noexcept
        {
            if constexpr (impl::is_custom_tag_invoke_awaiter_v<T, Promise>)
            {
                return hpx::functional::is_nothrow_tag_invocable_v<
                    as_awaitable_t, T, Promise&>;
            }
            else if constexpr (is_awaitable_v<T>)
            {
                return true;
            }
            else if constexpr (impl::is_awaitable_sender_v<T, Promise>)
            {
                using Sender = impl::sender_awaitable_t<Promise, T>;
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
            if constexpr (impl::is_custom_tag_invoke_awaiter_v<T, Promise>)
            {
                return tag_invoke(*this, (T &&) t, promise);
            }
            else if constexpr (is_awaitable_v<T>)
            {
                return (T &&) t;
            }
            else if constexpr (impl::is_awaitable_sender_v<T, Promise>)
            {
                auto hcoro =
                    hpx::coro::coroutine_handle<Promise>::from_promise(promise);
                return impl::sender_awaitable_t<Promise, T>{(T &&) t, hcoro};
            }
            else
            {
                return (T &&) t;
            }
        }
    };

}    // namespace hpx::execution::experimental

#endif    // HPX_HAVE_CXX20_COROUTINES
