//  Copyright (c) 2022 Shreyas Atre
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config/coroutines_support.hpp>
#include <hpx/execution_base/traits/coroutine_traits.hpp>
#include <hpx/modules/testing.hpp>

#include <utility>

struct awaiter_1
{
    bool await_ready()
    {
        return false;
    }
    void await_suspend(hpx::coro::coroutine_handle<>) {}
    void await_resume() {}
};

struct awaiter_2
{
    void await_ready() {}
    void await_suspend(hpx::coro::coroutine_handle<>) {}
    void await_resume() {}
};

struct awaiter_3
{
    void await_ready() {}
    void await_suspend(hpx::coro::coroutine_handle<>) {}
    void await_resume() {}
};

struct awaiter_4
{
    void await_ready() {}
    bool await_suspend(hpx::coro::coroutine_handle<>)
    {
        return false;
    }
    void await_resume() {}
};

struct awaiter_5
{
    void await_ready() {}
    bool await_suspend(hpx::coro::coroutine_handle<>)
    {
        return false;
    }
    bool await_resume()
    {
        return false;
    }
};

struct promise;

template <typename Promise>
struct awaiter_6
{
    bool await_ready()
    {
        return false;
    }
    void await_suspend(hpx::coro::coroutine_handle<Promise>) {}
    void await_resume() {}
};

struct non_awaiter_1
{
    bool await_suspend()
    {
        return false;
    }
    void await_resume() {}
};

struct non_awaiter_2
{
    bool await_suspend()
    {
        return false;
    }
};

struct non_awaiter_3
{
    void await_ready() {}
};

struct non_awaiter_4
{
};

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
        return awaiter_6<promise>{};
    }
};

struct awaitable_1
{
    awaiter_1 operator co_await();
    using promise_type = promise;
};

struct awaitable_2
{
    using promise_type = promise;
};

struct yes
{
    template <class T>
    T await_transform(T);
};

struct yes2
{
    int await_transform(int);
    char await_transform(char);
};

struct yes3
{
    template <typename T>
    T await_transform(T&&) = delete;
};

struct yes4
{
    std::function<int(int)> await_transform;
};

struct yes_final final
{
    template <typename T>
    T await_transform(T);
};

struct yes6
{
    template <typename T>
    static T await_transform(T) = delete;
};

struct yes7
{
    struct callable
    {
        template <typename T>
        T operator()(T) = delete;
    };

    static constexpr callable await_transform = {};
};

int main()
{
    using namespace hpx::execution::experimental;

    static_assert(
        std::is_same_v<decltype(await_suspend<void>(awaiter_1{})), void>);
    static_assert(
        std::is_same_v<decltype(await_suspend<void>(awaiter_2{})), void>);
    static_assert(
        std::is_same_v<decltype(await_suspend<void>(awaiter_3{})), void>);
    static_assert(
        std::is_same_v<decltype(await_suspend<void>(awaiter_4{})), void>);
    static_assert(
        std::is_same_v<decltype(await_suspend<void>(awaiter_5{})), void>);
    static_assert(
        std::is_same_v<decltype(await_suspend<promise>(awaiter_4{})), bool>);
    static_assert(
        std::is_same_v<decltype(await_suspend<promise>(awaiter_5{})), bool>);
    static_assert(
        std::is_same_v<decltype(await_suspend<promise>(awaiter_6<promise>{})),
            void>);
    static_assert(
        std::is_same_v<decltype(await_suspend<void>(awaiter_1{})), void>);

    static_assert(detail::has_await_resume<awaiter_1>);
    static_assert(detail::has_await_resume<awaiter_2>);
    static_assert(detail::has_await_resume<awaiter_3>);
    static_assert(detail::has_await_resume<awaiter_4>);
    static_assert(detail::has_await_resume<awaiter_5>);
    static_assert(detail::has_await_resume<awaiter_6<promise>>);
    static_assert(detail::has_await_resume<non_awaiter_1>);
    static_assert(!detail::has_await_resume<non_awaiter_2>);
    static_assert(!detail::has_await_resume<non_awaiter_3>);
    static_assert(!detail::has_await_resume<non_awaiter_4>);

    static_assert(detail::has_await_ready<awaiter_1>);
    static_assert(detail::has_await_ready<awaiter_2>);
    static_assert(detail::has_await_ready<awaiter_3>);
    static_assert(detail::has_await_ready<awaiter_4>);
    static_assert(detail::has_await_ready<awaiter_5>);
    static_assert(detail::has_await_ready<awaiter_6<promise>>);
    static_assert(!detail::has_await_ready<non_awaiter_1>);
    static_assert(!detail::has_await_ready<non_awaiter_2>);
    static_assert(detail::has_await_ready<non_awaiter_3>);
    static_assert(!detail::has_await_ready<non_awaiter_4>);

    static_assert(detail::has_await_suspend_coro_handle<awaiter_1, void>);
    static_assert(detail::has_await_suspend_coro_handle<awaiter_2, void>);
    static_assert(detail::has_await_suspend_coro_handle<awaiter_3, void>);
    static_assert(detail::has_await_suspend_coro_handle<awaiter_4, void>);
    static_assert(detail::has_await_suspend_coro_handle<awaiter_5, void>);
    static_assert(
        detail::has_await_suspend_coro_handle<awaiter_6<promise>, promise>);

    static_assert(!detail::has_await_suspend_coro_handle<non_awaiter_1, void>);
    static_assert(!detail::has_await_suspend_coro_handle<non_awaiter_2, void>);
    static_assert(!detail::has_await_suspend_coro_handle<non_awaiter_3, void>);
    static_assert(!detail::has_await_suspend_coro_handle<non_awaiter_4, void>);

    static_assert(is_awaiter_v<awaiter_1>);
    static_assert(is_awaiter_v<awaiter_2>);
    static_assert(is_awaiter_v<awaiter_3>);
    static_assert(is_awaiter_v<awaiter_4>);
    static_assert(is_awaiter_v<awaiter_5>);
    static_assert(is_awaiter_v<awaiter_6<promise>, promise>);
    static_assert(!is_awaiter_v<non_awaiter_1>);
    static_assert(!is_awaiter_v<non_awaiter_2>);
    static_assert(!is_awaiter_v<non_awaiter_3>);
    static_assert(!is_awaiter_v<non_awaiter_4>);

    static_assert(!detail::has_free_operator_co_await_v<awaiter_1>);
    static_assert(!detail::has_free_operator_co_await_v<awaiter_2>);
    static_assert(!detail::has_free_operator_co_await_v<awaiter_3>);
    static_assert(!detail::has_free_operator_co_await_v<awaiter_4>);
    static_assert(!detail::has_free_operator_co_await_v<awaiter_5>);
    static_assert(!detail::has_free_operator_co_await_v<awaiter_6<promise>>);
    static_assert(!detail::has_free_operator_co_await_v<non_awaiter_1>);
    static_assert(!detail::has_free_operator_co_await_v<non_awaiter_2>);
    static_assert(!detail::has_free_operator_co_await_v<non_awaiter_3>);
    static_assert(!detail::has_free_operator_co_await_v<non_awaiter_4>);

    static_assert(!detail::has_member_operator_co_await_v<awaiter_1>);
    static_assert(!detail::has_member_operator_co_await_v<awaiter_2>);
    static_assert(!detail::has_member_operator_co_await_v<awaiter_3>);
    static_assert(!detail::has_member_operator_co_await_v<awaiter_4>);
    static_assert(!detail::has_member_operator_co_await_v<awaiter_5>);
    static_assert(!detail::has_member_operator_co_await_v<awaiter_6<promise>>);
    static_assert(!detail::has_member_operator_co_await_v<non_awaiter_1>);
    static_assert(!detail::has_member_operator_co_await_v<non_awaiter_2>);
    static_assert(!detail::has_member_operator_co_await_v<non_awaiter_3>);
    static_assert(!detail::has_member_operator_co_await_v<non_awaiter_4>);

    static_assert(is_awaitable_v<awaiter_1>);
    static_assert(is_awaitable_v<awaiter_2>);
    static_assert(is_awaitable_v<awaiter_3>);
    static_assert(is_awaitable_v<awaiter_4>);
    static_assert(is_awaitable_v<awaiter_5>);
    static_assert(is_awaiter_v<decltype(get_awaiter(
                                   awaiter_6<promise>{}, (promise*) nullptr)),
        promise>);
    static_assert(is_awaitable_v<awaiter_6<promise>, promise>);
    static_assert(!is_awaitable_v<non_awaiter_1>);
    static_assert(!is_awaitable_v<non_awaiter_2>);
    static_assert(!is_awaitable_v<non_awaiter_3>);
    static_assert(!is_awaitable_v<non_awaiter_4>);

    static_assert(is_awaitable_v<awaitable_1>);

    static_assert(detail::has_await_transform_v<promise>);
    static_assert(detail::has_await_transform_v<yes>);
    static_assert(detail::has_await_transform_v<yes2>);
    static_assert(detail::has_await_transform_v<yes2>);
    static_assert(detail::has_await_transform_v<yes3>);
    static_assert(detail::has_await_transform_v<yes4>);
    static_assert(detail::has_await_transform_v<yes6>);
    static_assert(detail::has_await_transform_v<yes7>);

    static_assert(
        is_awaiter_v<decltype(std::declval<promise>().await_transform()),
            promise>);
    static_assert(
        std::is_same_v<decltype(get_awaiter(
                           std::declval<awaitable_2>(), (promise*) nullptr)),
            awaiter_6<promise>>);
    static_assert(
        std::is_same_v<decltype(get_awaiter(
                           std::declval<awaitable_2>(), (void*) nullptr)),
            awaitable_2&&>);
    static_assert(is_awaitable_v<awaitable_2, promise>);

    return hpx::util::report_errors();
}
