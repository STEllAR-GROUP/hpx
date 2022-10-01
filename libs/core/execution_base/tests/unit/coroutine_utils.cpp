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
};

template <typename S1, typename S2,
    typename = std::enable_if_t<hpx::execution::experimental::is_sender_v<S1> &&
        hpx::execution::experimental::is_sender_v<S2>>>
task<int> async_answer(S1 s1, S2 s2)
{
    // Senders are implicitly awaitable (in this coroutine type):
    co_await (S2 &&) s2;
    co_return co_await (S1 &&) s1;
}

int main()
{
    using namespace hpx::execution::experimental;

    {
        static_assert(std::is_same_v<
            single_sender_value_t<non_awaitable_sender<decltype(signature_all(
                std::exception_ptr(), int()))>>,
            int>);
        static_assert(
            std::is_same_v<single_sender_value_t<non_awaitable_sender<
                               decltype(signature_all(std::exception_ptr()))>>,
                void>);
    }

    try
    {
        // Awaitables are implicitly senders:
        auto [i] = hpx::this_thread::experimental::sync_wait(
            async_answer(hpx::execution::experimental::just(42),
                hpx::execution::experimental::just()))
                       .value();
        std::cout << "The answer is " << i << '\n';
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << '\n';
    }
    return hpx::util::report_errors();
}
