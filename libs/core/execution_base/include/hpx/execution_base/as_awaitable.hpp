//  Copyright (c) 2022 Shreyas Atre
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/datastructures/variant.hpp>
#include <hpx/execution/execution.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/get_env.hpp>
#include <hpx/execution_base/traits/coroutine_traits.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/local/functional.hpp>
#include <hpx/type_support/meta.hpp>

#include <exception>
#include <system_error>
#include <utility>

namespace hpx::execution::experimental {
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
    //      3. Otherwise, sender-awaitable{e, p} if awaitable-sender<E, P> is true.
    //      4. Otherwise, e.
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
        inline constexpr struct receiver_base
          : hpx::functional::tag<receiver_base>
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
                set_error((receiver_base &&) self, current_exception());
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
        inline constexpr struct sender_awaitable_base
        {
            using Promise = t<PromiseId>;
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
                    assert(!"_Should never get here");
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
                single_sender_value_t<t<SenderId>, env_of_t<t<PromiseId>>>>
        {
        private:
            using Promise = t<_PromiseId>;
            using Sender = t<_SenderId>;
            using Env = env_of_t<Promise>;
            using Value = single_sender_value_t<Sender, Env>;
            using Base = sender_awaitable_base<_PromiseId, Value>;
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
        using sender_awaitable_t = sender_awaitable<x<Promise>, x<Sender>>;

        template <typename T, typename Promise>
        inline constexpr bool is_custom_tag_invoke_awaiter =
            hpx::functional::is_tag_invocable_v<as_awaitable_t, T, Promise&>&&
                is_awaitable<tag_invoke_result_t<as_awaitable_t, _T, Promise&>,
                    Promise>;

        template <typename Sender, typename Promise>
        using receiver = typename sender_awaitable_base<x<Promise>,
            single_sender_value_t<Sender, env_of_t<Promise>>>::receiver;

        template <typename Sender, typename Promise>
        inline constexpr bool awaitable_sender =
            single_typed_sender<Sender, env_of_t<Promise>>&&
                sender_to<Sender, receiver<Sender, Promise>>&& requires(
                    Promise& promise)
        {
            {
                promise.unhandled_stopped()
            }
            ->convertible_to<hpx::coro::coroutine_handle<>>;
        };
    }    // namespace impl

    struct as_awaitable_t
    {
        template <typename T, typename Promise>
        static constexpr bool is_noexcept() noexcept
        {
            if constexpr (impl::custom_tag_invoke_awaiter<_T, Promise>)
            {
                return hpx::functional::is_nothrow_tag_invocable<as_awaitable_t,
                    _T, Promise&>;
            }
            else if constexpr (awaitable<_T>)
            {
                return true;
            }
            else if constexpr (impl::awaitable_sender<_T, Promise>)
            {
                using Sender = impl::sender_awaitable_t<Promise, _T>;
                return is_nothrow_constructible_v<Sender, _T,
                    hpx::coro::coroutine_handle<Promise>>;
            }
            else
            {
                return true;
            }
        }
        template <typename T, typename Promise>
        decltype(auto) operator()(_T&& t, Promise& promise) const
            noexcept(is_noexcept<_T, Promise>())
        {
            if constexpr (impl::custom_tag_invoke_awaiter<_T, Promise>)
            {
                return tag_invoke(*this, (_T &&) t, promise);
            }
            else if constexpr (awaitable<_T>)
            {
                return (_T &&) t;
            }
            else if constexpr (impl::awaitable_sender<_T, Promise>)
            {
                auto hcoro =
                    hpx::coro::coroutine_handle<Promise>::from_promise(promise);
                return impl::sender_awaitable_t<Promise, _T>{(_T &&) t, hcoro};
            }
            else
            {
                return (_T &&) t;
            }
        }
    };

}    // namespace hpx::execution::experimental