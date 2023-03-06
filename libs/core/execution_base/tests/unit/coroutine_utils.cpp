//  Copyright (c) 2022 Shreyas Atre
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config/coroutines_support.hpp>
#include <hpx/execution/algorithms/just.hpp>
#include <hpx/execution/algorithms/sync_wait.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/coroutine_utils.hpp>
#include <hpx/modules/testing.hpp>

#include "coroutine_task.hpp"

#include <exception>
#include <iostream>
#include <utility>

template <typename Error, typename... Values>
auto signature_all(Error, Values...)
    -> hpx::execution::experimental::completion_signatures<
        hpx::execution::experimental::set_value_t(Values...),
        hpx::execution::experimental::set_error_t(Error),
        hpx::execution::experimental::set_stopped_t()>
{
    return {};
}

template <typename Signatures>
struct non_awaitable_sender
{
    using completion_signatures = Signatures;
};

using dependent = hpx::execution::experimental::dependent_completion_signatures<
    hpx::execution::experimental::no_env>;

template <typename Awaiter>
struct promise
{
    hpx::coro::coroutine_handle<promise> get_return_object()
    {
        return {hpx::coro::coroutine_handle<promise>::from_promise(*this)};
    }
    hpx::coro::suspend_always initial_suspend() noexcept
    {
        return {};
    }
    hpx::coro::suspend_always final_suspend() noexcept
    {
        return {};
    }
    void return_void() {}
    void unhandled_exception() {}

    template <typename... T>
    auto await_transform(T&&...) noexcept
    {
        return Awaiter{};
    }
};

struct awaiter
{
    bool await_ready()
    {
        return true;
    }
    bool await_suspend(hpx::coro::coroutine_handle<>)
    {
        return false;
    }
    bool await_resume()
    {
        return false;
    }
};

template <typename Awaiter>
struct awaitable_sender_1
{
    Awaiter operator co_await()
    {
        return Awaiter{};
    }
};

struct awaitable_sender_2
{
    using promise_type = promise<hpx::coro::suspend_always>;

private:
    friend dependent operator co_await(awaitable_sender_2);
};

struct awaitable_sender_3
{
    using promise_type = promise<awaiter>;

private:
    friend dependent operator co_await(awaitable_sender_3);
};

struct awaitable_sender_4
{
    using promise_type = promise<hpx::coro::suspend_always>;

private:
    template <typename Promise>
    friend awaiter tag_invoke(hpx::execution::experimental::as_awaitable_t,
        awaitable_sender_4, Promise&)
    {
        return {};
    }

    friend dependent tag_invoke(hpx::execution::experimental::as_awaitable_t,
        awaitable_sender_4,
        hpx::execution::experimental::detail::env_promise<
            hpx::execution::experimental::no_env>&)
    {
        return {};
    }
};

struct awaitable_sender_5
{
private:
    template <typename Promise>
    friend awaiter tag_invoke(hpx::execution::experimental::as_awaitable_t,
        awaitable_sender_5, Promise&)
    {
        return {};
    }
};

struct recv_set_value
{
    friend void tag_invoke(hpx::execution::experimental::set_value_t,
        recv_set_value,
        decltype(std::declval<dependent>().await_ready())) noexcept
    {
    }
    friend void tag_invoke(
        hpx::execution::experimental::set_stopped_t, recv_set_value) noexcept
    {
    }
    friend void tag_invoke(hpx::execution::experimental::set_error_t,
        recv_set_value, std::exception_ptr) noexcept
    {
    }
    friend dependent tag_invoke(
        hpx::execution::experimental::get_env_t, const recv_set_value&) noexcept
    {
        return {};
    }
};

template <class T>
T& unmove(T&& t)
{
    return t;
}

template <typename S1, typename S2,
    typename = std::enable_if_t<hpx::execution::experimental::is_sender_v<S1> &&
        hpx::execution::experimental::is_sender_v<S2>>>
task<int> async_answer(S1 s1, S2 s2)
{
    // Senders are implicitly awaitable (in this coroutine type):
    // clang-format off
    co_await(S2 &&) s2;
    co_return co_await(S1 &&) s1;
    // clang-format on
}

template <class Sender>
inline constexpr bool is_sender_with_env_v =
    hpx::execution::experimental::is_sender_v<Sender>&&
        hpx::is_invocable_v<hpx::execution::experimental::get_env_t, Sender>;

int main()
{
    using namespace hpx::execution::experimental;

    {
        // clang-format off
            static_assert(
                std::is_same_v<single_sender_value_t<non_awaitable_sender<decltype(
                                   signature_all(std::exception_ptr(), int()))>>,
                    int>);
            static_assert(
                std::is_same_v<single_sender_value_t<non_awaitable_sender<decltype(
                                   signature_all(std::exception_ptr()))>>,
                    void>);
        // clang-format on
    }

    // single sender value
    {
        static_assert(
            std::is_same_v<single_sender_value_t<awaitable_sender_1<awaiter>>,
                bool>);
        static_assert(
            std::is_same_v<single_sender_value_t<
                               awaitable_sender_1<hpx::coro::suspend_always>>,
                void>);
    }

    // connect awaitable
    {
        static_assert(std::is_same_v<decltype(connect_awaitable(
                                         awaitable_sender_1<awaiter>{},
                                         recv_set_value{})),
            operation_t<recv_set_value>>);

        static_assert(
            std::is_same_v<decltype(connect(awaitable_sender_1<awaiter>{},
                               recv_set_value{})),
                operation_t<recv_set_value>>);
    }

    // Promise env
    {
        static_assert(is_awaiter_v<awaiter>);

        static_assert(
            !detail::has_free_operator_co_await_v<awaitable_sender_1<awaiter>>);
        static_assert(detail::has_free_operator_co_await_v<awaitable_sender_2>);
        static_assert(detail::has_free_operator_co_await_v<awaitable_sender_3>);
        static_assert(
            !detail::has_free_operator_co_await_v<awaitable_sender_4>);
        static_assert(
            !detail::has_free_operator_co_await_v<awaitable_sender_5>);

        static_assert(detail::has_member_operator_co_await_v<
            awaitable_sender_1<awaiter>>);
        static_assert(
            !detail::has_member_operator_co_await_v<awaitable_sender_2>);
        static_assert(
            !detail::has_member_operator_co_await_v<awaitable_sender_3>);
        static_assert(
            !detail::has_member_operator_co_await_v<awaitable_sender_4>);
        static_assert(
            !detail::has_member_operator_co_await_v<awaitable_sender_5>);

        static_assert(is_awaitable_v<awaitable_sender_1<awaiter>>);
        static_assert(is_awaitable_v<awaitable_sender_2>);
        static_assert(is_awaitable_v<awaitable_sender_3>);
        static_assert(!is_awaitable_v<awaitable_sender_4>);
        static_assert(!is_awaitable_v<awaitable_sender_5>);

        static_assert(is_awaitable_v<awaitable_sender_2,
            ::promise<hpx::coro::suspend_always>>);
        static_assert(is_awaitable_v<awaitable_sender_3, ::promise<awaiter>>);
        static_assert(is_awaitable_v<awaitable_sender_4, ::promise<awaiter>>);
        static_assert(is_awaitable_v<awaitable_sender_5, ::promise<awaiter>>);
        static_assert(
            std::is_same_v<hpx::functional::tag_invoke_result_t<as_awaitable_t,
                               awaitable_sender_4, ::promise<awaiter>&>,
                awaiter>);
        static_assert(std::is_same_v<
            hpx::functional::tag_invoke_result_t<as_awaitable_t,
                awaitable_sender_4, detail::env_promise<no_env>&>,
            detail::dependent_completion_signatures<no_env>>);
        static_assert(std::is_same_v<
            decltype(get_awaiter(std::declval<awaitable_sender_4>(),
                static_cast<detail::env_promise<no_env>*>(nullptr))),
            detail::dependent_completion_signatures<no_env>>);
        static_assert(is_awaiter_v<decltype(get_awaiter(
                std::declval<awaitable_sender_4>(),
                static_cast<detail::env_promise<no_env>*>(nullptr)))>);
        static_assert(detail::has_await_suspend_v<decltype(get_awaiter(
                std::declval<awaitable_sender_4>(),
                static_cast<detail::env_promise<no_env>*>(nullptr)))>);
        static_assert(detail::is_with_await_suspend_v<
            decltype(get_awaiter(std::declval<awaitable_sender_4>(),
                static_cast<detail::env_promise<no_env>*>(nullptr))),
            detail::env_promise<no_env>>);
        static_assert(
            is_awaitable_v<awaitable_sender_4, detail::env_promise<no_env>>);
        static_assert(
            is_awaitable_v<awaitable_sender_4, detail::env_promise<empty_env>>);
        static_assert(
            is_awaitable_v<awaitable_sender_5, detail::env_promise<no_env>>);
        static_assert(
            is_awaitable_v<awaitable_sender_5, detail::env_promise<empty_env>>);
    }

    // Operation base
    {
        static_assert(is_operation_state_v<operation_t<recv_set_value>>);
    }

    // Connect result type
    {
        static_assert(std::is_same_v<
            connect_result_t<awaitable_sender_1<awaiter>, recv_set_value>,
            operation_t<recv_set_value>>);
    }

    // As awaitable
    {
        static_assert(is_awaitable_v<decltype(as_awaitable(
                awaitable_sender_1<awaiter>{}, unmove(::promise<awaiter>{})))>);
        static_assert(
            std::is_same_v<decltype(as_awaitable(awaitable_sender_1<awaiter>{},
                               unmove(::promise<awaiter>{}))),
                awaitable_sender_1<awaiter>&&>);
    }

    // sender
    {
        static_assert(is_sender_v<awaitable_sender_1<awaiter>>);
        static_assert(is_sender_v<awaitable_sender_2>);
        static_assert(detail::is_enable_sender_v<awaitable_sender_2>);
        static_assert(detail::is_sender_plain_v<awaitable_sender_2, no_env>);
        static_assert(is_sender_v<awaitable_sender_3>);
        static_assert(is_sender_v<awaitable_sender_4>);
    }

    // env promise
    {
        static_assert(is_sender_with_env_v<awaitable_sender_1<awaiter>>);
        static_assert(is_sender_with_env_v<awaitable_sender_2>);
        static_assert(is_sender_with_env_v<awaitable_sender_3>);
        static_assert(is_sender_with_env_v<awaitable_sender_4>);
    }

    try
    {
        // Awaitables are implicitly senders:
        auto i = hpx::this_thread::experimental::sync_wait(
            async_answer(hpx::execution::experimental::just(42),
                hpx::execution::experimental::just()))
                     .value();
        std::cout << "The answer is " << hpx::get<0>(i) << '\n';
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << '\n';
    }
    return hpx::util::report_errors();
}
