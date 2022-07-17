//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This tests whether C++20 coroutines are supported

#if __has_include(<coroutine>)
#include <coroutine>
namespace coro = std;
#else
#include <experimental/coroutine>
namespace coro = std::experimental;
#endif

struct resumable
{
    resumable() = default;
    resumable(int) {}

    struct promise_type
    {
        resumable get_return_object()
        {
            return {};
        }

        coro::suspend_never initial_suspend() const noexcept
        {
            return {};
        }

        coro::suspend_never final_suspend() const noexcept
        {
            return {};
        }

        void return_value(int value) const noexcept {}
        void unhandled_exception() const noexcept {}
    };

    bool await_ready() const noexcept
    {
        return true;
    }
    int await_resume() const noexcept
    {
        return 42;
    }
    void await_suspend(coro::coroutine_handle<promise_type>) const noexcept {}
};

resumable test()
{
    int result = co_await resumable{42};
    (void) result;
    co_return result;
}

int main()
{
    test();
    return 0;
}
