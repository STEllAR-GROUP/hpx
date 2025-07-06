//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2022-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

// Clang V11 ICE's on this test, Clang V8 reports a bogus constexpr problem
#if !defined(HPX_CLANG_VERSION) ||                                             \
    ((HPX_CLANG_VERSION / 10000) != 11 && (HPX_CLANG_VERSION / 10000) != 8)

#include <hpx/condition_variable.hpp>
#include <hpx/execution.hpp>
#include <hpx/functional.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/mutex.hpp>
#include <hpx/thread.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

struct custom_type_non_default_constructible_non_copyable
{
    int x;
    custom_type_non_default_constructible_non_copyable() = delete;
    explicit custom_type_non_default_constructible_non_copyable(int x)
      : x(x) {};
    custom_type_non_default_constructible_non_copyable(
        custom_type_non_default_constructible_non_copyable&&) = default;
    custom_type_non_default_constructible_non_copyable& operator=(
        custom_type_non_default_constructible_non_copyable&&) = default;
    custom_type_non_default_constructible_non_copyable(
        custom_type_non_default_constructible_non_copyable const&) = delete;
    custom_type_non_default_constructible_non_copyable& operator=(
        custom_type_non_default_constructible_non_copyable const&) = delete;
};

namespace ex = hpx::execution::experimental;
namespace tt = hpx::this_thread::experimental;

///////////////////////////////////////////////////////////////////////////////
void test_execute()
{
    hpx::thread::id parent_id = hpx::this_thread::get_id();

    ex::thread_pool_scheduler sched{};
    ex::execute(sched,
        [parent_id]() { HPX_TEST_NEQ(hpx::this_thread::get_id(), parent_id); });
}

struct check_context_receiver
{
    hpx::thread::id parent_id;
    hpx::mutex& mtx;
    hpx::condition_variable& cond;
    bool& executed;
#if defined(HPX_HAVE_STDEXEC)
    using receiver_concept = ex::receiver_t;
#endif
    template <typename E>
    friend void tag_invoke(
        ex::set_error_t, check_context_receiver&&, E&&) noexcept
    {
        HPX_TEST(false);
    }

    friend void tag_invoke(ex::set_stopped_t, check_context_receiver&&) noexcept
    {
        HPX_TEST(false);
    }

    template <typename... Ts>
    friend void tag_invoke(
        ex::set_value_t, check_context_receiver&& r, Ts&&...) noexcept
    {
        HPX_TEST_NEQ(r.parent_id, hpx::this_thread::get_id());
        HPX_TEST_NEQ(hpx::thread::id(hpx::threads::invalid_thread_id),
            hpx::this_thread::get_id());
        std::lock_guard l{r.mtx};
        r.executed = true;
        r.cond.notify_one();
    }
};

void test_sender_receiver_basic()
{
    hpx::thread::id parent_id = hpx::this_thread::get_id();
    hpx::mutex mtx;
    hpx::condition_variable cond;
    bool executed{false};

    ex::thread_pool_scheduler sched{};

    auto begin = ex::schedule(sched);
    auto os = ex::connect(std::move(begin),
        check_context_receiver{parent_id, mtx, cond, executed});
    ex::start(os);

    {
        std::unique_lock l{mtx};
        cond.wait(l, [&]() { return executed; });
    }

    HPX_TEST(executed);
}

hpx::thread::id sender_receiver_then_thread_id;

void test_sender_receiver_then()
{
    ex::thread_pool_scheduler sched{};
    hpx::thread::id parent_id = hpx::this_thread::get_id();
    hpx::mutex mtx;
    hpx::condition_variable cond;
    bool executed{false};

    auto begin = ex::schedule(sched);
    auto work1 = ex::then(std::move(begin), [=]() {
        sender_receiver_then_thread_id = hpx::this_thread::get_id();
        HPX_TEST_NEQ(sender_receiver_then_thread_id, parent_id);
    });
    auto work2 = ex::then(std::move(work1), []() {
        HPX_TEST_EQ(sender_receiver_then_thread_id, hpx::this_thread::get_id());
    });
    auto os = ex::connect(std::move(work2),
        check_context_receiver{parent_id, mtx, cond, executed});
    ex::start(os);

    {
        std::unique_lock l{mtx};
        cond.wait(l, [&]() { return executed; });
    }

    HPX_TEST(executed);
}

void test_sender_receiver_then_wait()
{
    ex::thread_pool_scheduler sched{};
    hpx::thread::id parent_id = hpx::this_thread::get_id();
    std::atomic<std::size_t> then_count{0};
    bool executed{false};

    auto begin = ex::schedule(sched);
    auto work1 = ex::then(std::move(begin), [&then_count, parent_id]() {
        sender_receiver_then_thread_id = hpx::this_thread::get_id();
        HPX_TEST_NEQ(sender_receiver_then_thread_id, parent_id);
        ++then_count;
    });
    auto work2 = ex::then(std::move(work1), [&then_count, &executed]() {
        HPX_TEST_EQ(sender_receiver_then_thread_id, hpx::this_thread::get_id());
        ++then_count;
        executed = true;
    });
    tt::sync_wait(std::move(work2));

    HPX_TEST_EQ(then_count, std::size_t(2));
    HPX_TEST(executed);
}

// NOLINTBEGIN(bugprone-unchecked-optional-access)
void test_sender_receiver_then_sync_wait()
{
    ex::thread_pool_scheduler sched{};
    hpx::thread::id parent_id = hpx::this_thread::get_id();
    std::atomic<std::size_t> then_count{0};

    auto begin = ex::schedule(sched);
    auto work = ex::then(std::move(begin), [&then_count, parent_id]() {
        sender_receiver_then_thread_id = hpx::this_thread::get_id();
        HPX_TEST_NEQ(sender_receiver_then_thread_id, parent_id);
        ++then_count;
        return 42;
    });
    auto result = hpx::get<0>(*(tt::sync_wait(std::move(work))));
    HPX_TEST_EQ(then_count, std::size_t(1));
    static_assert(
        std::is_same<int, typename std::decay<decltype(result)>::type>::value,
        "result should be an int");
    HPX_TEST_EQ(result, 42);
}

void test_sender_receiver_then_arguments()
{
    ex::thread_pool_scheduler sched{};
    hpx::thread::id parent_id = hpx::this_thread::get_id();
    std::atomic<std::size_t> then_count{0};

    auto begin = ex::schedule(sched);
    auto work1 = ex::then(std::move(begin), [&then_count, parent_id]() {
        sender_receiver_then_thread_id = hpx::this_thread::get_id();
        HPX_TEST_NEQ(sender_receiver_then_thread_id, parent_id);
        ++then_count;
        return 3;
    });
    auto work2 =
        ex::then(std::move(work1), [&then_count](int x) -> std::string {
            HPX_TEST_EQ(
                sender_receiver_then_thread_id, hpx::this_thread::get_id());
            ++then_count;
            return std::string("hello") + std::to_string(x);
        });
    auto work3 = ex::then(std::move(work2), [&then_count](std::string s) {
        HPX_TEST_EQ(sender_receiver_then_thread_id, hpx::this_thread::get_id());
        ++then_count;
        return 2 * s.size();
    });
    auto result = hpx::get<0>(*tt::sync_wait(std::move(work3)));
    HPX_TEST_EQ(then_count, std::size_t(3));
    static_assert(std::is_same<std::size_t,
                      typename std::decay<decltype(result)>::type>::value,
        "result should be a std::size_t");
    HPX_TEST_EQ(result, std::size_t(12));
}

template <typename F>
struct callback_receiver
{
    std::decay_t<F> f;
    hpx::mutex& mtx;
    hpx::condition_variable& cond;
    bool& executed;
#if defined(HPX_HAVE_STDEXEC)
    using receiver_concept = ex::receiver_t;
#endif

    template <typename E>
    friend void tag_invoke(ex::set_error_t, callback_receiver&&, E&&) noexcept
    {
        HPX_TEST(false);
    }

    friend void tag_invoke(ex::set_stopped_t, callback_receiver&&) noexcept
    {
        HPX_TEST(false);
    }

    template <typename... Ts>
    friend void tag_invoke(
        ex::set_value_t, callback_receiver&& r, Ts&&...) noexcept
    {
        HPX_INVOKE(r.f, );
        std::lock_guard l{r.mtx};
        r.executed = true;
        r.cond.notify_one();
    }
};

void test_properties()
{
    ex::thread_pool_scheduler sched{};
    hpx::mutex mtx;
    hpx::condition_variable cond;
    bool executed{false};

    constexpr std::array<hpx::threads::thread_priority, 4> priorities{
        {hpx::threads::thread_priority::low,
            hpx::threads::thread_priority::normal,
            hpx::threads::thread_priority::bound,
            hpx::threads::thread_priority::high}};

    for (auto const prio : priorities)
    {
        auto exec_prop = ex::with_priority(sched, prio);
        HPX_TEST_EQ(ex::get_priority(exec_prop), prio);

        auto check = [prio]() {
            HPX_TEST_EQ(prio, hpx::this_thread::get_priority());
        };
        executed = false;
        auto os = ex::connect(ex::schedule(exec_prop),
            callback_receiver<decltype(check)>{check, mtx, cond, executed});
        ex::start(os);
        {
            std::unique_lock l{mtx};
            cond.wait(l, [&]() { return executed; });
        }

        HPX_TEST(executed);
    }

    constexpr std::array<hpx::threads::thread_stacksize, 4> stacksizes{
        {hpx::threads::thread_stacksize::small_,
            hpx::threads::thread_stacksize::medium,
            hpx::threads::thread_stacksize::large,
            hpx::threads::thread_stacksize::huge}};

    for (auto const stacksize : stacksizes)
    {
        auto exec_prop = ex::with_stacksize(sched, stacksize);
        HPX_TEST_EQ(ex::get_stacksize(exec_prop), stacksize);

        auto check = [stacksize]() {
            HPX_TEST_EQ(stacksize,
                hpx::threads::get_thread_id_data(hpx::threads::get_self_id())
                    ->get_stack_size_enum());
        };
        executed = false;
        auto os = ex::connect(ex::schedule(exec_prop),
            callback_receiver<decltype(check)>{check, mtx, cond, executed});
        ex::start(os);
        {
            std::unique_lock l{mtx};
            cond.wait(l, [&]() { return executed; });
        }

        HPX_TEST(executed);
    }

    constexpr std::array<hpx::threads::thread_schedule_hint, 4> hints{
        {hpx::threads::thread_schedule_hint{},
            hpx::threads::thread_schedule_hint{1},
            hpx::threads::thread_schedule_hint{
                hpx::threads::thread_schedule_hint_mode::thread, 2},
            hpx::threads::thread_schedule_hint{
                hpx::threads::thread_schedule_hint_mode::numa, 3}}};

    for (auto const hint : hints)
    {
        auto exec_prop = ex::with_hint(sched, hint);
        HPX_TEST(ex::get_hint(exec_prop) == hint);

        // A hint is not guaranteed to be respected, so we only check that the
        // thread_pool_scheduler holds the property.
    }

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    {
        char const* annotation = "<test>";
        auto exec_prop = ex::with_annotation(sched, annotation);
        HPX_TEST_EQ(std::string(ex::get_annotation(exec_prop)),
            std::string(annotation));

        auto check = [annotation]() {
            HPX_TEST_EQ(std::string(annotation),
                hpx::threads::get_thread_description(
                    hpx::threads::get_self_id())
                    .get_description());
        };
        executed = false;
        auto os = ex::connect(ex::schedule(exec_prop),
            callback_receiver<decltype(check)>{check, mtx, cond, executed});
        ex::start(os);

        {
            std::unique_lock l{mtx};
            cond.wait(l, [&]() { return executed; });
        }

        HPX_TEST(executed);
    }
#endif
}

void test_transfer_basic()
{
    ex::thread_pool_scheduler sched{};
    hpx::thread::id parent_id = hpx::this_thread::get_id();
    hpx::thread::id current_id;

    auto begin = ex::schedule(sched);
    auto work1 = ex::then(begin, [=, &current_id]() {
        current_id = hpx::this_thread::get_id();
        HPX_TEST_NEQ(current_id, parent_id);
    });
    auto work2 = ex::then(work1, [=, &current_id]() {
        HPX_TEST_EQ(current_id, hpx::this_thread::get_id());
    });
    auto transfer1 = ex::transfer(work2, sched);
    auto work3 = ex::then(transfer1, [=, &current_id]() {
        hpx::thread::id new_id = hpx::this_thread::get_id();
        HPX_TEST_NEQ(current_id, new_id);
        current_id = new_id;
        HPX_TEST_NEQ(current_id, parent_id);
    });
    auto work4 = ex::then(work3, [=, &current_id]() {
        HPX_TEST_EQ(current_id, hpx::this_thread::get_id());
    });
    auto transfer2 = ex::transfer(work4, sched);
    auto work5 = ex::then(transfer2, [=, &current_id]() {
        hpx::thread::id new_id = hpx::this_thread::get_id();
        HPX_TEST_NEQ(current_id, new_id);
        current_id = new_id;
        HPX_TEST_NEQ(current_id, parent_id);
    });

    tt::sync_wait(work5);
}

void test_transfer_arguments()
{
    ex::thread_pool_scheduler sched{};
    hpx::thread::id parent_id = hpx::this_thread::get_id();
    hpx::thread::id current_id;

    auto begin = ex::schedule(sched);
    auto work1 = ex::then(begin, [=, &current_id]() {
        current_id = hpx::this_thread::get_id();
        HPX_TEST_NEQ(current_id, parent_id);
        return 3;
    });
    auto work2 = ex::then(work1, [=, &current_id](int x) {
        HPX_TEST_EQ(current_id, hpx::this_thread::get_id());
        return x / 2.0;
    });
    auto transfer1 = ex::transfer(work2, sched);
    auto work3 = ex::then(transfer1, [=, &current_id](double x) {
        hpx::thread::id new_id = hpx::this_thread::get_id();
        HPX_TEST_NEQ(current_id, new_id);
        current_id = new_id;
        HPX_TEST_NEQ(current_id, parent_id);
        return x / 2;
    });
    auto work4 = ex::then(work3, [=, &current_id](int x) {
        HPX_TEST_EQ(current_id, hpx::this_thread::get_id());
        return "result: " + std::to_string(x);
    });
    auto transfer2 = ex::transfer(work4, sched);
    auto work5 = ex::then(transfer2, [=, &current_id](std::string s) {
        hpx::thread::id new_id = hpx::this_thread::get_id();
        HPX_TEST_NEQ(current_id, new_id);
        current_id = new_id;
        HPX_TEST_NEQ(current_id, parent_id);
        return s + "!";
    });

    auto result = hpx::get<0>(*tt::sync_wait(work5));
    static_assert(std::is_same<std::string,
                      typename std::decay<decltype(result)>::type>::value,
        "result should be a std::string");
    HPX_TEST_EQ(result, std::string("result: 0!"));
}

void test_just_void()
{
    {
        hpx::thread::id parent_id = hpx::this_thread::get_id();

        auto begin = ex::just();
        auto work1 = ex::then(begin, [parent_id]() {
            HPX_TEST_EQ(parent_id, hpx::this_thread::get_id());
        });
        tt::sync_wait(work1);
    }

    {
        hpx::thread::id parent_id = hpx::this_thread::get_id();

        auto begin = ex::just();
        auto transfer1 = ex::transfer(begin, ex::thread_pool_scheduler{});
        auto work1 = ex::then(transfer1, [parent_id]() {
            HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
        });
        tt::sync_wait(work1);
    }
}

void test_just_one_arg()
{
    {
        hpx::thread::id parent_id = hpx::this_thread::get_id();

        auto begin = ex::just(3);
        auto work1 = ex::then(begin, [parent_id](int x) {
            HPX_TEST_EQ(parent_id, hpx::this_thread::get_id());
            HPX_TEST_EQ(x, 3);
        });
        tt::sync_wait(work1);
    }

    {
        hpx::thread::id parent_id = hpx::this_thread::get_id();

        auto begin = ex::just(3);
        auto transfer1 = ex::transfer(begin, ex::thread_pool_scheduler{});
        auto work1 = ex::then(transfer1, [parent_id](int x) {
            HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
            HPX_TEST_EQ(x, 3);
        });
        tt::sync_wait(work1);
    }
}

void test_just_two_args()
{
    {
        hpx::thread::id parent_id = hpx::this_thread::get_id();

        auto begin = ex::just(3, std::string("hello"));
        auto work1 = ex::then(begin, [parent_id](int x, std::string y) {
            HPX_TEST_EQ(parent_id, hpx::this_thread::get_id());
            HPX_TEST_EQ(x, 3);
            HPX_TEST_EQ(y, std::string("hello"));
        });
        tt::sync_wait(work1);
    }

    {
        hpx::thread::id parent_id = hpx::this_thread::get_id();

        auto begin = ex::just(3, std::string("hello"));
        auto transfer1 = ex::transfer(begin, ex::thread_pool_scheduler{});
        auto work1 = ex::then(transfer1, [parent_id](int x, std::string y) {
            HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
            HPX_TEST_EQ(x, 3);
            HPX_TEST_EQ(y, std::string("hello"));
        });
        tt::sync_wait(work1);
    }
}

void test_transfer_just_void()
{
    hpx::thread::id parent_id = hpx::this_thread::get_id();

    auto begin = ex::transfer_just(ex::thread_pool_scheduler{});
    auto work1 = ex::then(begin,
        [parent_id]() { HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id()); });
    tt::sync_wait(work1);
}

void test_transfer_just_one_arg()
{
    hpx::thread::id parent_id = hpx::this_thread::get_id();

    auto begin = ex::transfer_just(ex::thread_pool_scheduler{}, 3);
    auto work1 = ex::then(begin, [parent_id](int x) {
        HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
        HPX_TEST_EQ(x, 3);
    });
    tt::sync_wait(work1);
}

void test_transfer_just_two_args()
{
    hpx::thread::id parent_id = hpx::this_thread::get_id();

    auto begin =
        ex::transfer_just(ex::thread_pool_scheduler{}, 3, std::string("hello"));
    auto work1 = ex::then(begin, [parent_id](int x, std::string y) {
        HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
        HPX_TEST_EQ(x, 3);
        HPX_TEST_EQ(y, std::string("hello"));
    });
    tt::sync_wait(work1);
}

void test_when_all()
{
    ex::thread_pool_scheduler sched{};

    {
        hpx::thread::id parent_id = hpx::this_thread::get_id();

        auto work1 = ex::schedule(sched) | ex::then([parent_id]() {
            HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
            return 42;
        });

        auto work2 = ex::schedule(sched) | ex::then([parent_id]() {
            HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
            return std::string("hello");
        });

        auto work3 = ex::schedule(sched) | ex::then([parent_id]() {
            HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
            return 3.14;
        });

        auto when1 =
            ex::when_all(std::move(work1), std::move(work2), std::move(work3));

        bool executed{false};
#if defined(HPX_HAVE_STDEXEC)
        tt::sync_wait(std::move(when1) |
            ex::then([parent_id, &executed](int x, std::string y, double z) {
                HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
                HPX_TEST_EQ(x, 42);
                HPX_TEST_EQ(y, std::string("hello"));
                HPX_TEST_EQ(z, 3.14);
                executed = true;
            }));
#else
        std::move(when1) |
            ex::then([parent_id, &executed](int x, std::string y, double z) {
                HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
                HPX_TEST_EQ(x, 42);
                HPX_TEST_EQ(y, std::string("hello"));
                HPX_TEST_EQ(z, 3.14);
                executed = true;
            }) |
            tt::sync_wait();
#endif
        HPX_TEST(executed);
    }

    {
        hpx::thread::id parent_id = hpx::this_thread::get_id();

        // The exception is likely to be thrown before set_value from the second
        // sender is called because the second sender sleeps.
        auto work1 = ex::schedule(sched) | ex::then([parent_id]() -> int {
            HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
            throw std::runtime_error("error");
        });

        auto work2 = ex::schedule(sched) | ex::then([parent_id]() {
            HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
            hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
            return std::string("hello");
        });

        bool exception_thrown = false;

        try
        {
#if defined(HPX_HAVE_STDEXEC)
            tt::sync_wait(ex::when_all(std::move(work1), std::move(work2)) |
                ex::then([parent_id](int x, std::string y) {
                    HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
                    HPX_TEST_EQ(x, 42);
                    HPX_TEST_EQ(y, std::string("hello"));
                }));
#else
            ex::when_all(std::move(work1), std::move(work2)) |
                ex::then([parent_id](int x, std::string y) {
                    HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
                    HPX_TEST_EQ(x, 42);
                    HPX_TEST_EQ(y, std::string("hello"));
                }) |
                tt::sync_wait();
#endif
            HPX_TEST(false);
        }
        catch (std::runtime_error const& e)
        {
            HPX_TEST_EQ(std::string(e.what()), std::string("error"));
            exception_thrown = true;
        }

        HPX_TEST(exception_thrown);
    }

    {
        hpx::thread::id parent_id = hpx::this_thread::get_id();

        // The exception is likely to be thrown after set_value from the second
        // sender is called because the first sender sleeps before throwing.
        auto work1 = ex::schedule(sched) | ex::then([parent_id]() -> int {
            HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
            hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
            throw std::runtime_error("error");
        });

        auto work2 = ex::schedule(sched) | ex::then([parent_id]() {
            HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
            return std::string("hello");
        });

        bool exception_thrown = false;

        try
        {
#if defined(HPX_HAVE_STDEXEC)
            tt::sync_wait(ex::when_all(std::move(work1), std::move(work2)) |
                ex::then([parent_id](int x, std::string y) {
                    HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
                    HPX_TEST_EQ(x, 42);
                    HPX_TEST_EQ(y, std::string("hello"));
                }));
#else
            ex::when_all(std::move(work1), std::move(work2)) |
                ex::then([parent_id](int x, std::string y) {
                    HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
                    HPX_TEST_EQ(x, 42);
                    HPX_TEST_EQ(y, std::string("hello"));
                }) |
                tt::sync_wait();
#endif
            HPX_TEST(false);
        }
        catch (std::runtime_error const& e)
        {
            HPX_TEST_EQ(std::string(e.what()), std::string("error"));
            exception_thrown = true;
        }

        HPX_TEST(exception_thrown);
    }
}

void test_future_sender()
{
    // futures as senders
    {
        std::atomic<bool> called{false};
        auto f = hpx::async([&]() { called = true; });
        auto fs = ex::as_sender(std::move(f));

        static_assert(ex::is_sender_v<std::decay_t<decltype(fs)>>,
            "a future should be adaptable to a sender");
        static_assert(std::is_same_v<
                          std::decay_t<decltype(*tt::sync_wait(std::move(fs)))>,
#if defined(HPX_HAVE_STDEXEC)
                          std::tuple<>>,
#else
                          hpx::tuple<>>,
#endif
            "sync_wait should return hpx::tuple<>");

        tt::sync_wait(std::move(fs));
        HPX_TEST(called);

        bool exception_thrown = false;
        try
        {
            // The move is intentional. sync_wait should throw.
            // NOLINTNEXTLINE(bugprone-use-after-move)
            tt::sync_wait(ex::as_sender(std::move(f)));
            HPX_TEST(false);
        }
        catch (...)
        {
            exception_thrown = true;
        }
        HPX_TEST(exception_thrown);
    }

    {
        std::atomic<bool> called{false};
        auto f = hpx::async([&]() {
            called = true;
            return 42;
        });
        auto fs = ex::as_sender(std::move(f));

        static_assert(ex::is_sender_v<std::decay_t<decltype(fs)>>,
            "a future should be adaptable as a sender");

        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(std::move(fs))), 42);
        HPX_TEST(called);

        bool exception_thrown = false;
        try
        {
            // The move is intentional. sync_wait should throw.
            // NOLINTNEXTLINE(bugprone-use-after-move)
            tt::sync_wait(ex::as_sender(std::move(f)));
            HPX_TEST(false);
        }
        catch (...)
        {
            exception_thrown = true;
        }
        HPX_TEST(exception_thrown);
    }

    {
        std::atomic<bool> called{false};
        auto f = hpx::async([&]() {
            called = true;
            return 42;
        });

        HPX_TEST_EQ(
            hpx::get<0>(*tt::sync_wait(ex::then(
                ex::as_sender(std::move(f)), [](int x) { return x / 2; }))),
            21);
        HPX_TEST(called);
    }

    {
        std::atomic<std::size_t> calls{0};
        auto sf = hpx::async([&]() { ++calls; }).share();
        auto sfs = ex::as_sender(sf);

        static_assert(ex::is_sender_v<std::decay_t<decltype(sfs)>>,
            "a shared_future should be adaptable as a sender");
        static_assert(
            std::is_same_v<std::decay_t<decltype(*tt::sync_wait(sfs))>,
#if defined(HPX_HAVE_STDEXEC)
                std::tuple<>>,
#else
                hpx::tuple<>>,
#endif
            "sync_wait should return hpx::tuple<>");

        tt::sync_wait(sfs);
        tt::sync_wait(sfs);
        tt::sync_wait(std::move(sfs));
        HPX_TEST_EQ(calls, std::size_t(1));

        bool exception_thrown = false;
        try
        {
            tt::sync_wait(sfs);
            HPX_TEST(false);
        }
        catch (...)
        {
            exception_thrown = true;
        }
        HPX_TEST(exception_thrown);
    }

    {
        std::atomic<std::size_t> calls{0};
        auto sf = hpx::async([&]() {
            ++calls;
            return 42;
        }).share();
        auto sfs = ex::as_sender(sf);

        static_assert(ex::is_sender_v<std::decay_t<decltype(sfs)>>,
            "a shared_future should be a sender");

        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(sfs)), 42);
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(sfs)), 42);
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(std::move(sfs))), 42);
        HPX_TEST_EQ(calls, std::size_t(1));

        bool exception_thrown = false;
        try
        {
            tt::sync_wait(sfs);
            HPX_TEST(false);
        }
        catch (...)
        {
            exception_thrown = true;
        }
        HPX_TEST(exception_thrown);
    }

    // senders as futures
    {
        auto s = ex::just(3);
        auto f = ex::make_future(std::move(s));
        HPX_TEST_EQ(f.get(), 3);
    }

    {
        auto s = ex::transfer_just(ex::thread_pool_scheduler{}, 3);
        auto f = ex::make_future(std::move(s));
        HPX_TEST_EQ(f.get(), 3);
    }

    {
        auto f = ex::just(3) | ex::make_future();
        HPX_TEST_EQ(f.get(), 3);
    }

    {
        auto f = ex::transfer_just(ex::thread_pool_scheduler{}, 3) |
            ex::make_future();
        HPX_TEST_EQ(f.get(), 3);
    }

    {
        std::atomic<bool> called{false};
        auto s = ex::schedule(ex::thread_pool_scheduler{}) |
            ex::then([&] { called = true; });
        auto f = ex::make_future(std::move(s));
        f.get();
        HPX_TEST(called);
    }

    {
        auto s1 =
            ex::transfer_just(ex::thread_pool_scheduler{}, std::size_t(42));
        auto s2 = ex::transfer_just(ex::thread_pool_scheduler{}, 3.14);
        auto s3 = ex::transfer_just(
            ex::thread_pool_scheduler{}, std::string("hello"));
        auto f = ex::make_future(ex::then(
            ex::when_all(std::move(s1), std::move(s2), std::move(s3)),
            [](std::size_t x, double, std::string z) { return z.size() + x; }));
        HPX_TEST_EQ(f.get(), std::size_t(47));
    }

    // mixing senders and futures
    {
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(ex::as_sender(ex::make_future(
                        ex::transfer_just(ex::thread_pool_scheduler{}, 42))))),
            42);
    }

    {
        HPX_TEST_EQ(
            ex::make_future(
                ex::transfer(ex::as_sender(hpx::async([]() { return 42; })),
                    ex::thread_pool_scheduler{}))
                .get(),
            42);
    }

    {
        auto s1 =
            ex::transfer_just(ex::thread_pool_scheduler{}, std::size_t(42));
        auto s2 = ex::transfer_just(ex::thread_pool_scheduler{}, 3.14);
        auto s3 = ex::transfer_just(
            ex::thread_pool_scheduler{}, std::string("hello"));
        auto f = ex::make_future(ex::then(
            ex::when_all(std::move(s1), std::move(s2), std::move(s3)),
            [](std::size_t x, double, std::string z) { return z.size() + x; }));
        auto sf = f.then([](auto&& f) { return f.get() - 40; }).share();
        auto t1 = sf.then([](auto&& sf) { return sf.get() + 1; });
        auto t2 = sf.then([](auto&& sf) { return sf.get() + 2; });
        auto t1s = ex::then(
            ex::as_sender(std::move(t1)), [](std::size_t x) { return x + 1; });
        auto t1f = ex::make_future(std::move(t1s));
        auto last = hpx::dataflow(
            hpx::unwrapping([](std::size_t x, std::size_t y) { return x + y; }),
            t1f, t2);

        HPX_TEST_EQ(last.get(), std::size_t(18));
    }
}

void test_ensure_started()
{
    ex::thread_pool_scheduler sched{};

    {
#if defined(HPX_HAVE_STDEXEC)
        tt::sync_wait(ex::schedule(sched) | ex::ensure_started());
#else
        ex::schedule(sched) | ex::ensure_started() | tt::sync_wait();
#endif
    }

    {
        auto s = ex::transfer_just(sched, 42) | ex::ensure_started();
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(std::move(s))), 42);
    }

    {
        auto s = ex::transfer_just(sched, 42) | ex::ensure_started() |
            ex::transfer(sched);
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(std::move(s))), 42);
    }

    {
        auto s = ex::transfer_just(sched, 42) | ex::ensure_started()
#if defined(HPX_HAVE_STDEXEC)
            | ex::split()
#endif
            ;
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(s)), 42);
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(s)), 42);
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(s)), 42);
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(std::move(s))), 42);
    }
}

void test_ensure_started_when_all()
{
    ex::thread_pool_scheduler sched{};

    {
        std::atomic<std::size_t> first_task_calls{0};
        std::atomic<std::size_t> successor_task_calls{0};
        hpx::mutex mtx;
        hpx::condition_variable cond;
        bool started{false};
        auto s = ex::schedule(sched) | ex::then([&]() {
            ++first_task_calls;
            std::lock_guard l{mtx};
            started = true;
            cond.notify_one();
        }) | ex::ensure_started()
#if defined(HPX_HAVE_STDEXEC)
            | ex::split();
#endif
        ;
        {
            std::unique_lock l{mtx};
            cond.wait(l, [&]() { return started; });
        }
        auto succ1 = s | ex::then([&]() {
            ++successor_task_calls;
            return 1;
        });
        auto succ2 = s | ex::then([&]() {
            ++successor_task_calls;
            return 2;
        });
#if defined(HPX_HAVE_STDEXEC)
        HPX_TEST_EQ(
            hpx::get<0>(*tt::sync_wait(ex::when_all(succ1, succ2) |
                ex::then([](int const& x, int const& y) { return x + y; }))),
            3);
#else
        HPX_TEST_EQ(
            hpx::get<0>(*(ex::when_all(succ1, succ2) |
                ex::then([](int const& x, int const& y) { return x + y; }) |
                tt::sync_wait())),
            3);
#endif
        HPX_TEST_EQ(first_task_calls, std::size_t(1));
        HPX_TEST_EQ(successor_task_calls, std::size_t(2));
    }

    {
        std::atomic<std::size_t> first_task_calls{0};
        std::atomic<std::size_t> successor_task_calls{0};
        hpx::mutex mtx;
        hpx::condition_variable cond;
        bool started{false};
        auto s = ex::schedule(sched) | ex::then([&]() {
            ++first_task_calls;
            std::lock_guard l{mtx};
            started = true;
            cond.notify_one();
            return 3;
        }) | ex::ensure_started()
#if defined(HPX_HAVE_STDEXEC)
            | ex::split()
#endif
            ;
        {
            std::unique_lock l{mtx};
            cond.wait(l, [&]() { return started; });
        }
        HPX_TEST_EQ(first_task_calls, std::size_t(1));
        auto succ1 = s | ex::then([&](int const& x) {
            ++successor_task_calls;
            return x + 1;
        });
        auto succ2 = s | ex::then([&](int const& x) {
            ++successor_task_calls;
            return x + 2;
        });
#if defined(HPX_HAVE_STDEXEC)
        HPX_TEST_EQ(
            hpx::get<0>(*tt::sync_wait(ex::when_all(succ1, succ2) |
                ex::then([](int const& x, int const& y) { return x + y; }))),
            9);
#else
        HPX_TEST_EQ(
            hpx::get<0>(*(ex::when_all(succ1, succ2) |
                ex::then([](int const& x, int const& y) { return x + y; }) |
                tt::sync_wait())),
            9);
#endif
        HPX_TEST_EQ(first_task_calls, std::size_t(1));
        HPX_TEST_EQ(successor_task_calls, std::size_t(2));
    }

    {
        std::atomic<std::size_t> first_task_calls{0};
        std::atomic<std::size_t> successor_task_calls{0};
        hpx::mutex mtx;
        hpx::condition_variable cond;
        bool started{false};
        auto s = ex::schedule(sched) | ex::then([&]() {
            ++first_task_calls;
            std::lock_guard l{mtx};
            started = true;
            cond.notify_one();
            return 3;
        }) | ex::ensure_started()
#if defined(HPX_HAVE_STDEXEC)
            | ex::split()
#endif
            ;
        {
            std::unique_lock l{mtx};
            cond.wait(l, [&]() { return started; });
        }
        auto succ1 = s | ex::transfer(sched) | ex::then([&](int const& x) {
            ++successor_task_calls;
            return x + 1;
        });
        auto succ2 = s | ex::transfer(sched) | ex::then([&](int const& x) {
            ++successor_task_calls;
            return x + 2;
        });
#if defined(HPX_HAVE_STDEXEC)
        HPX_TEST_EQ(
            hpx::get<0>(*tt::sync_wait(ex::when_all(succ1, succ2) |
                ex::then([](int const& x, int const& y) { return x + y; }))),
            9);
#else
        HPX_TEST_EQ(
            hpx::get<0>(*(ex::when_all(succ1, succ2) |
                ex::then([](int const& x, int const& y) { return x + y; }) |
                tt::sync_wait())),
            9);
#endif
        HPX_TEST_EQ(first_task_calls, std::size_t(1));
        HPX_TEST_EQ(successor_task_calls, std::size_t(2));
    }
}

void test_split()
{
    ex::thread_pool_scheduler sched{};

    {
#if defined(HPX_HAVE_STDEXEC)
        tt::sync_wait(ex::schedule(sched) | ex::split());
#else
        ex::schedule(sched) | ex::split() | tt::sync_wait();
#endif
    }

    {
        auto s = ex::transfer_just(sched, 42) | ex::split();
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(std::move(s))), 42);
    }

    {
        auto s =
            ex::transfer_just(sched, 42) | ex::split() | ex::transfer(sched);
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(std::move(s))), 42);
    }

    {
        auto s = ex::transfer_just(sched, 42) | ex::split();
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(s)), 42);
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(s)), 42);
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(s)), 42);
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(std::move(s))), 42);
    }
}

void test_split_when_all()
{
    ex::thread_pool_scheduler sched{};

    {
        std::atomic<std::size_t> first_task_calls{0};
        std::atomic<std::size_t> successor_task_calls{0};
        auto s = ex::schedule(sched) | ex::then([&]() { ++first_task_calls; }) |
            ex::split();
        auto succ1 = s | ex::then([&]() {
            ++successor_task_calls;
            return 1;
        });
        auto succ2 = s | ex::then([&]() {
            ++successor_task_calls;
            return 2;
        });
#if defined(HPX_HAVE_STDEXEC)
        HPX_TEST_EQ(
            hpx::get<0>(*tt::sync_wait(ex::when_all(succ1, succ2) |
                ex::then([](int const& x, int const& y) { return x + y; }))),
            3);
#else
        HPX_TEST_EQ(
            hpx::get<0>(*(ex::when_all(succ1, succ2) |
                ex::then([](int const& x, int const& y) { return x + y; }) |
                tt::sync_wait())),
            3);
#endif
        HPX_TEST_EQ(first_task_calls, std::size_t(1));
        HPX_TEST_EQ(successor_task_calls, std::size_t(2));
    }

    {
        std::atomic<std::size_t> first_task_calls{0};
        std::atomic<std::size_t> successor_task_calls{0};
        auto s = ex::schedule(sched) | ex::then([&]() {
            ++first_task_calls;
            return 3;
        }) | ex::split();
        auto succ1 = s | ex::then([&](int const& x) {
            ++successor_task_calls;
            return x + 1;
        });
        auto succ2 = s | ex::then([&](int const& x) {
            ++successor_task_calls;
            return x + 2;
        });
#if defined(HPX_HAVE_STDEXEC)
        HPX_TEST_EQ(
            hpx::get<0>(*tt::sync_wait(ex::when_all(succ1, succ2) |
                ex::then([](int const& x, int const& y) { return x + y; }))),
            9);
#else

        HPX_TEST_EQ(
            hpx::get<0>(*(ex::when_all(succ1, succ2) |
                ex::then([](int const& x, int const& y) { return x + y; }) |
                tt::sync_wait())),
            9);
#endif
        HPX_TEST_EQ(first_task_calls, std::size_t(1));
        HPX_TEST_EQ(successor_task_calls, std::size_t(2));
    }

    {
        std::atomic<std::size_t> first_task_calls{0};
        std::atomic<std::size_t> successor_task_calls{0};
        auto s = ex::schedule(sched) | ex::then([&]() {
            ++first_task_calls;
            return 3;
        }) | ex::split();
        auto succ1 = s | ex::transfer(sched) | ex::then([&](int const& x) {
            ++successor_task_calls;
            return x + 1;
        });
        auto succ2 = s | ex::transfer(sched) | ex::then([&](int const& x) {
            ++successor_task_calls;
            return x + 2;
        });
#if defined(HPX_HAVE_STDEXEC)
        HPX_TEST_EQ(
            hpx::get<0>(*(tt::sync_wait(ex::when_all(succ1, succ2) |
                ex::then([](int const& x, int const& y) { return x + y; })))),
            9);
#else
        HPX_TEST_EQ(
            hpx::get<0>(*(ex::when_all(succ1, succ2) |
                ex::then([](int const& x, int const& y) { return x + y; }) |
                tt::sync_wait())),
            9);
#endif
        HPX_TEST_EQ(first_task_calls, std::size_t(1));
        HPX_TEST_EQ(successor_task_calls, std::size_t(2));
    }
}

void test_let_value()
{
    ex::thread_pool_scheduler sched{};

    // void predecessor
    {
#if defined(HPX_HAVE_STDEXEC)
        auto result = hpx::get<0>(*(tt::sync_wait(ex::schedule(sched) |
            ex::let_value([]() { return ex::just(42); }))));
#else
        auto result = hpx::get<0>(*(ex::schedule(sched) |
            ex::let_value([]() { return ex::just(42); }) | tt::sync_wait()));
#endif
        HPX_TEST_EQ(result, 42);
    }

    {
#if defined(HPX_HAVE_STDEXEC)
        auto result = hpx::get<0>(*(tt::sync_wait(ex::schedule(sched) |
            ex::let_value([=]() { return ex::transfer_just(sched, 42); }))));
#else
        auto result = hpx::get<0>(*(ex::schedule(sched) | ex::let_value([=]() {
            return ex::transfer_just(sched, 42);
        }) | tt::sync_wait()));
#endif
        HPX_TEST_EQ(result, 42);
    }

    {
#if defined(HPX_HAVE_STDEXEC)
        auto result = hpx::get<0>(*tt::sync_wait((ex::just() |
            ex::let_value([=]() { return ex::transfer_just(sched, 42); }))));
#else
        auto result = hpx::get<0>(*(ex::just() | ex::let_value([=]() {
            return ex::transfer_just(sched, 42);
        }) | tt::sync_wait()));
#endif
        HPX_TEST_EQ(result, 42);
    }

    // int predecessor, value ignored
    {
#if defined(HPX_HAVE_STDEXEC)
        auto result = hpx::get<0>(*(tt::sync_wait(ex::transfer_just(sched, 43) |
            ex::let_value([](int&) { return ex::just(42); }))));
#else
        auto result = hpx::get<0>(*(ex::transfer_just(sched, 43) |
            ex::let_value([](int&) { return ex::just(42); }) |
            tt::sync_wait()));
#endif
        HPX_TEST_EQ(result, 42);
    }

    {
#if defined(HPX_HAVE_STDEXEC)
        auto result = hpx::get<0>(*(tt::sync_wait(ex::transfer_just(sched, 43) |
            ex::let_value(
                [=](int&) { return ex::transfer_just(sched, 42); }))));
#else
        auto result = hpx::get<0>(*(ex::transfer_just(sched, 43) |
            ex::let_value([=](int&) { return ex::transfer_just(sched, 42); }) |
            tt::sync_wait()));
#endif
        HPX_TEST_EQ(result, 42);
    }

    {
#if defined(HPX_HAVE_STDEXEC)
        auto result =
            hpx::get<0>(*(tt::sync_wait(ex::just(43) | ex::let_value([=](int&) {
                return ex::transfer_just(sched, 42);
            }))));
#else
        auto result = hpx::get<0>(*(ex::just(43) | ex::let_value([=](int&) {
            return ex::transfer_just(sched, 42);
        }) | tt::sync_wait()));
#endif
        HPX_TEST_EQ(result, 42);
    }

    // int predecessor, value used
    {
#if defined(HPX_HAVE_STDEXEC)
        auto result = hpx::get<0>(*(tt::sync_wait(
            ex::transfer_just(sched, 43) | ex::let_value([](int& x) {
                return ex::just(42) | ex::then([&](int y) { return x + y; });
            }))));

#else
        auto result = hpx::get<0>(
            *(ex::transfer_just(sched, 43) | ex::let_value([](int& x) {
                return ex::just(42) | ex::then([&](int y) { return x + y; });
            }) | tt::sync_wait()));
#endif
        HPX_TEST_EQ(result, 85);
    }

    {
#if defined(HPX_HAVE_STDEXEC)
        auto result = hpx::get<0>(*(tt::sync_wait(
            ex::transfer_just(sched, 43) | ex::let_value([=](int& x) {
                return ex::transfer_just(sched, 42) |
                    ex::then([&](int y) { return x + y; });
            }))));
#else
        auto result = hpx::get<0>(
            *(ex::transfer_just(sched, 43) | ex::let_value([=](int& x) {
                return ex::transfer_just(sched, 42) |
                    ex::then([&](int y) { return x + y; });
            }) | tt::sync_wait()));
#endif
        HPX_TEST_EQ(result, 85);
    }

    {
#if defined(HPX_HAVE_STDEXEC)
        auto result = hpx::get<0>(
            *(tt::sync_wait(ex::just(43) | ex::let_value([=](int& x) {
                return ex::transfer_just(sched, 42) |
                    ex::then([&](int y) { return x + y; });
            }))));
#else
        auto result = hpx::get<0>(*(ex::just(43) | ex::let_value([=](int& x) {
            return ex::transfer_just(sched, 42) |
                ex::then([&](int y) { return x + y; });
        }) | tt::sync_wait()));
#endif
        HPX_TEST_EQ(result, 85);
    }

    // predecessor throws, let sender is ignored
    {
        bool exception_thrown = false;

        try
        {
#if defined(HPX_HAVE_STDEXEC)
            tt::sync_wait(ex::transfer_just(sched, 43) | ex::then([](int x) {
                throw std::runtime_error("error");
                return x;
            }) | ex::let_value([](int&) {
                HPX_TEST(false);
                return ex::just(0);
            }));
#else
            ex::transfer_just(sched, 43) | ex::then([](int x) {
                throw std::runtime_error("error");
                return x;
            }) | ex::let_value([](int&) {
                HPX_TEST(false);
                return ex::just(0);
            }) | tt::sync_wait();
#endif
            HPX_TEST(false);
        }
        catch (std::runtime_error const& e)
        {
            HPX_TEST_EQ(std::string(e.what()), std::string("error"));
            exception_thrown = true;
        }

        HPX_TEST(exception_thrown);
    }
}

void check_exception_ptr_message(
    std::exception_ptr ep, std::string const& message)
{
    try
    {
        std::rethrow_exception(ep);
    }
    catch (std::runtime_error const& e)
    {
        HPX_TEST_EQ(std::string(e.what()), message);
        return;
    }

    HPX_TEST(false);
}

void test_let_error()
{
    ex::thread_pool_scheduler sched{};

    // void predecessor
    {
        std::atomic<bool> called{false};
#if defined(HPX_HAVE_STDEXEC)
        tt::sync_wait(ex::schedule(sched) | ex::then([]() {
            throw std::runtime_error("error");
        }) | ex::let_error([&called](std::exception_ptr& ep) {
            called = true;
            check_exception_ptr_message(ep, "error");
            return ex::just();
        }));
#else
        ex::schedule(sched) | ex::then([]() {
            throw std::runtime_error("error");
        }) | ex::let_error([&called](std::exception_ptr& ep) {
            called = true;
            check_exception_ptr_message(ep, "error");
            return ex::just();
        }) | tt::sync_wait();
#endif
        HPX_TEST(called);
    }

    {
        std::atomic<bool> called{false};
#if defined(HPX_HAVE_STDEXEC)
        tt::sync_wait(ex::schedule(sched) | ex::then([]() {
            throw std::runtime_error("error");
        }) | ex::let_error([=, &called](std::exception_ptr& ep) {
            called = true;
            check_exception_ptr_message(ep, "error");
            return ex::transfer_just(sched);
        }));
#else
        ex::schedule(sched) | ex::then([]() {
            throw std::runtime_error("error");
        }) | ex::let_error([=, &called](std::exception_ptr& ep) {
            called = true;
            check_exception_ptr_message(ep, "error");
            return ex::transfer_just(sched);
        }) | tt::sync_wait();
#endif
        HPX_TEST(called);
    }

    {
        std::atomic<bool> called{false};
#if defined(HPX_HAVE_STDEXEC)
        tt::sync_wait(ex::just() | ex::then([]() {
            throw std::runtime_error("error");
        }) | ex::let_error([=, &called](std::exception_ptr& ep) {
            called = true;
            check_exception_ptr_message(ep, "error");
            return ex::transfer_just(sched);
        }));
#else
        ex::just() | ex::then([]() { throw std::runtime_error("error"); }) |
            ex::let_error([=, &called](std::exception_ptr& ep) {
                called = true;
                check_exception_ptr_message(ep, "error");
                return ex::transfer_just(sched);
            }) |
            tt::sync_wait();
#endif
        HPX_TEST(called);
    }

    // int predecessor
    {
#if defined(HPX_HAVE_STDEXEC)
        auto result =
            hpx::get<0>(*(tt::sync_wait(ex::schedule(sched) | ex::then([]() {
                throw std::runtime_error("error");
                return 43;
            }) | ex::let_error([](std::exception_ptr& ep) {
                check_exception_ptr_message(ep, "error");
                return ex::just(42);
            }))));
#else
        auto result = hpx::get<0>(*(ex::schedule(sched) | ex::then([]() {
            throw std::runtime_error("error");
            return 43;
        }) | ex::let_error([](std::exception_ptr& ep) {
            check_exception_ptr_message(ep, "error");
            return ex::just(42);
        }) | tt::sync_wait()));
#endif
        HPX_TEST_EQ(result, 42);
    }

    {
#if defined(HPX_HAVE_STDEXEC)
        auto result =
            hpx::get<0>(*(tt::sync_wait(ex::schedule(sched) | ex::then([]() {
                throw std::runtime_error("error");
                return 43;
            }) | ex::let_error([=](std::exception_ptr& ep) {
                check_exception_ptr_message(ep, "error");
                return ex::transfer_just(sched, 42);
            }))));
#else
        auto result = hpx::get<0>(*(ex::schedule(sched) | ex::then([]() {
            throw std::runtime_error("error");
            return 43;
        }) | ex::let_error([=](std::exception_ptr& ep) {
            check_exception_ptr_message(ep, "error");
            return ex::transfer_just(sched, 42);
        }) | tt::sync_wait()));
#endif
        HPX_TEST_EQ(result, 42);
    }

    {
#if defined(HPX_HAVE_STDEXEC)
        auto result = hpx::get<0>(*(tt::sync_wait(ex::just() | ex::then([]() {
            throw std::runtime_error("error");
            return 43;
        }) | ex::let_error([=](std::exception_ptr& ep) {
            check_exception_ptr_message(ep, "error");
            return ex::transfer_just(sched, 42);
        }))));
#else
        auto result = hpx::get<0>(*(ex::just() | ex::then([]() {
            throw std::runtime_error("error");
            return 43;
        }) | ex::let_error([=](std::exception_ptr& ep) {
            check_exception_ptr_message(ep, "error");
            return ex::transfer_just(sched, 42);
        }) | tt::sync_wait()));
#endif
        HPX_TEST_EQ(result, 42);
    }

    // predecessor doesn't throw, let sender is ignored
    {
#if defined(HPX_HAVE_STDEXEC)
        auto result = hpx::get<0>(*(tt::sync_wait(ex::transfer_just(sched, 42) |
            ex::let_error([](std::exception_ptr) {
                HPX_TEST(false);
                return ex::just(43);
            }))));
#else
        auto result = hpx::get<0>(*(ex::transfer_just(sched, 42) |
            ex::let_error([](std::exception_ptr) {
                HPX_TEST(false);
                return ex::just(43);
            }) |
            tt::sync_wait()));
#endif
        HPX_TEST_EQ(result, 42);
    }

    {
#if defined(HPX_HAVE_STDEXEC)
        auto result = hpx::get<0>(*(tt::sync_wait(ex::transfer_just(sched, 42) |
            ex::let_error([=](std::exception_ptr) {
                HPX_TEST(false);
                return ex::transfer_just(sched, 43);
            }))));
#else
        auto result = hpx::get<0>(*(ex::transfer_just(sched, 42) |
            ex::let_error([=](std::exception_ptr) {
                HPX_TEST(false);
                return ex::transfer_just(sched, 43);
            }) |
            tt::sync_wait()));
#endif
        HPX_TEST_EQ(result, 42);
    }

    {
#if defined(HPX_HAVE_STDEXEC)
        auto result = hpx::get<0>(*(
            tt::sync_wait(ex::just(42) | ex::let_error([=](std::exception_ptr) {
                HPX_TEST(false);
                return ex::transfer_just(sched, 43);
            }))));
#else
        auto result =
            hpx::get<0>(*(ex::just(42) | ex::let_error([=](std::exception_ptr) {
                HPX_TEST(false);
                return ex::transfer_just(sched, 43);
            }) | tt::sync_wait()));
#endif
        HPX_TEST_EQ(result, 42);
    }
}

void test_detach()
{
    ex::thread_pool_scheduler sched{};

    {
        bool called = false;
        hpx::mutex mtx;
        hpx::condition_variable cond;
#if defined(HPX_HAVE_STDEXEC)
        ex::start_detached(ex::schedule(sched) | ex::then([&]() {
            std::unique_lock l{mtx};
            called = true;
            cond.notify_one();
        }));
#else
        ex::schedule(sched) | ex::then([&]() {
            std::unique_lock l{mtx};
            called = true;
            cond.notify_one();
        }) | ex::start_detached();
#endif

        {
            std::unique_lock l{mtx};
            HPX_TEST(cond.wait_for(
                l, std::chrono::seconds(1), [&]() { return called; }));
        }
        HPX_TEST(called);
    }

    // Values passed to set_value are ignored
    {
        bool called = false;
        hpx::mutex mtx;
        hpx::condition_variable cond;
#if defined(HPX_HAVE_STDEXEC)
        ex::start_detached(ex::schedule(sched) | ex::then([&]() {
            std::lock_guard l{mtx};
            called = true;
            cond.notify_one();
            return 42;
        }));
#else
        ex::schedule(sched) | ex::then([&]() {
            std::lock_guard l{mtx};
            called = true;
            cond.notify_one();
            return 42;
        }) | ex::start_detached();
#endif

        {
            std::unique_lock l{mtx};
            HPX_TEST(cond.wait_for(
                l, std::chrono::seconds(1), [&]() { return called; }));
        }
        HPX_TEST(called);
    }
}

void test_keep_future_sender()
{
    // the future should be passed to then, not it's contained value
    {
#if defined(HPX_HAVE_STDEXEC)
        tt::sync_wait(ex::keep_future(hpx::make_ready_future<void>()) |
            ex::then([](hpx::future<void>&& f) { HPX_TEST(f.is_ready()); }));
#else
        ex::keep_future(hpx::make_ready_future<void>()) |
            ex::then([](hpx::future<void>&& f) { HPX_TEST(f.is_ready()); }) |
            tt::sync_wait();
#endif
    }

    {
#if defined(HPX_HAVE_STDEXEC)
        tt::sync_wait(ex::keep_future(hpx::make_ready_future<void>().share()) |
            ex::then(
                [](hpx::shared_future<void>&& f) { HPX_TEST(f.is_ready()); }));
#else
        ex::keep_future(hpx::make_ready_future<void>().share()) |
            ex::then(
                [](hpx::shared_future<void>&& f) { HPX_TEST(f.is_ready()); }) |
            tt::sync_wait();
#endif
    }

    {
#if defined(HPX_HAVE_STDEXEC)
        tt::sync_wait(ex::keep_future(hpx::make_ready_future<int>(42)) |
            ex::then([](hpx::future<int>&& f) {
                HPX_TEST(f.is_ready());
                HPX_TEST_EQ(f.get(), 42);
            }));
#else
        ex::keep_future(hpx::make_ready_future<int>(42)) |
            ex::then([](hpx::future<int>&& f) {
                HPX_TEST(f.is_ready());
                HPX_TEST_EQ(f.get(), 42);
            }) |
            tt::sync_wait();
#endif
    }

    {
#if defined(HPX_HAVE_STDEXEC)
        tt::sync_wait(ex::keep_future(hpx::make_ready_future<int>(42).share()) |
            ex::then([](hpx::shared_future<int>&& f) {
                HPX_TEST(f.is_ready());
                HPX_TEST_EQ(f.get(), 42);
            }));
#else
        ex::keep_future(hpx::make_ready_future<int>(42).share()) |
            ex::then([](hpx::shared_future<int>&& f) {
                HPX_TEST(f.is_ready());
                HPX_TEST_EQ(f.get(), 42);
            }) |
            tt::sync_wait();
#endif
    }

    {
        std::atomic<bool> called{false};
        auto f = hpx::async([&]() { called = true; });

        auto r = hpx::get<0>(*tt::sync_wait(ex::keep_future(std::move(f))));
        static_assert(
            std::is_same<std::decay_t<decltype(r)>, hpx::future<void>>::value,
            "sync_wait should return future<void>");

        HPX_TEST(called);
        HPX_TEST(r.is_ready());

        bool exception_thrown = false;
        try
        {
            // The move is intentional. sync_wait should throw.
            // NOLINTNEXTLINE(bugprone-use-after-move)
            tt::sync_wait(ex::keep_future(std::move(f)));
            HPX_TEST(false);
        }
        catch (...)
        {
            exception_thrown = true;
        }
        HPX_TEST(exception_thrown);
    }

    {
        std::atomic<bool> called{false};
        auto f = hpx::async([&]() {
            called = true;
            return 42;
        });

        auto r = hpx::get<0>(*tt::sync_wait(ex::keep_future(std::move(f))));
        static_assert(
            std::is_same<std::decay_t<decltype(r)>, hpx::future<int>>::value,
            "sync_wait should return future<int>");

        HPX_TEST(called);
        HPX_TEST(r.is_ready());
        HPX_TEST_EQ(r.get(), 42);

        bool exception_thrown = false;
        try
        {
            // The move is intentional. sync_wait should throw.
            // NOLINTNEXTLINE(bugprone-use-after-move)
            tt::sync_wait(ex::keep_future(std::move(f)));
            HPX_TEST(false);
        }
        catch (...)
        {
            exception_thrown = true;
        }
        HPX_TEST(exception_thrown);
    }

    {
        std::atomic<bool> called{false};
        auto f = hpx::async([&]() {
            called = true;
            return 42;
        });

        HPX_TEST_EQ(
            hpx::get<0>(*tt::sync_wait(ex::then(ex::keep_future(std::move(f)),
                [](hpx::future<int>&& f) { return f.get() / 2; }))),
            21);
        HPX_TEST(called);
    }

    {
        std::atomic<std::size_t> calls{0};
        auto sf = hpx::async([&]() { ++calls; }).share();

        tt::sync_wait(ex::keep_future(sf));
        tt::sync_wait(ex::keep_future(sf));
        tt::sync_wait(ex::keep_future(std::move(sf)));
        HPX_TEST_EQ(calls, std::size_t(1));

        bool exception_thrown = false;
        try
        {
            tt::sync_wait(ex::keep_future(sf));
            HPX_TEST(false);
        }
        catch (...)
        {
            exception_thrown = true;
        }
        HPX_TEST(exception_thrown);
    }

    {
        std::atomic<std::size_t> calls{0};
        auto sf = hpx::async([&]() {
            ++calls;
            return 42;
        }).share();

        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(ex::keep_future(sf))).get(), 42);
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(ex::keep_future(sf))).get(), 42);
        HPX_TEST_EQ(
            hpx::get<0>(*tt::sync_wait(ex::keep_future(std::move(sf)))).get(),
            42);
        HPX_TEST_EQ(calls, std::size_t(1));

        bool exception_thrown = false;
        try
        {
            tt::sync_wait(ex::keep_future(sf));
            HPX_TEST(false);
        }
        catch (...)
        {
            exception_thrown = true;
        }
        HPX_TEST(exception_thrown);
    }

    // Keep future alive across on
    {
        auto f = hpx::async([&]() { return 42; });
#if defined(HPX_HAVE_STDEXEC)
        auto r = hpx::get<0>(*(tt::sync_wait(ex::keep_future(std::move(f)) |
            ex::transfer(ex::thread_pool_scheduler{}))));
#else
        auto r = hpx::get<0>(*(ex::keep_future(std::move(f)) |
            ex::transfer(ex::thread_pool_scheduler{}) | tt::sync_wait()));
#endif
        HPX_TEST(r.is_ready());
        HPX_TEST_EQ(r.get(), 42);
    }

    {
        auto sf = hpx::async([&]() { return 42; }).share();
#if defined(HPX_HAVE_STDEXEC)
        auto r = hpx::get<0>(*(tt::sync_wait(ex::keep_future(std::move(sf)) |
            ex::transfer(ex::thread_pool_scheduler{}))));
#else
        auto r = hpx::get<0>(*(ex::keep_future(std::move(sf)) |
            ex::transfer(ex::thread_pool_scheduler{}) | tt::sync_wait()));
#endif
        HPX_TEST(r.is_ready());
        HPX_TEST_EQ(r.get(), 42);
    }

    {
        auto sf = hpx::async([&]() {
            return custom_type_non_default_constructible_non_copyable{42};
        }).share();

        // NOTE: Without keep_future this should fail to compile, since
        // sync_wait would receive a const& to the value which requires a copy
        // or storing a const&. The copy is not possible because the type is
        // noncopyable, and storing a reference is not acceptable since the
        // reference may outlive the value.
#if defined(HPX_HAVE_STDEXEC)
        auto r = hpx::get<0>(*(tt::sync_wait(ex::keep_future(std::move(sf)) |
            ex::transfer(ex::thread_pool_scheduler{}))));
#else
        auto r = hpx::get<0>(*(ex::keep_future(std::move(sf)) |
            ex::transfer(ex::thread_pool_scheduler{}) | tt::sync_wait()));
#endif
        HPX_TEST(r.is_ready());
        HPX_TEST_EQ(r.get().x, 42);
    }

    // Use unwrapping with keep_future
    {
        auto f = hpx::async([]() { return 42; });
        auto sf = hpx::async([]() { return 3.14; }).share();

        auto fun = hpx::unwrapping(
            [](int&& x, double const& y) { return x * 2 + (int(y) / 2); });
#if defined(HPX_HAVE_STDEXEC)
        HPX_TEST_EQ(hpx::get<0>(*(tt::sync_wait(
                        ex::when_all(ex::keep_future(std::move(f)),
                            ex::keep_future(sf)) |
                        ex::then(fun)))),
            85);
#else
        HPX_TEST_EQ(hpx::get<0>(*(ex::when_all(ex::keep_future(std::move(f)),
                                      ex::keep_future(sf)) |
                        ex::then(fun) | tt::sync_wait())),
            85);
#endif
    }

    {
        auto f = hpx::async([]() { return 42; });
        auto sf = hpx::async([]() { return 3.14; }).share();

        auto fun = hpx::unwrapping(
            [](int&& x, double const& y) { return x * 2 + (int(y) / 2); });
#if defined(HPX_HAVE_STDEXEC)
        HPX_TEST_EQ(
            hpx::get<0>(*(tt::sync_wait(
                ex::when_all(
                    ex::keep_future(std::move(f)), ex::keep_future(sf)) |
                ex::transfer(ex::thread_pool_scheduler{}) | ex::then(fun)))),
            85);
#else
        HPX_TEST_EQ(hpx::get<0>(*(ex::when_all(ex::keep_future(std::move(f)),
                                      ex::keep_future(sf)) |
                        ex::transfer(ex::thread_pool_scheduler{}) |
                        ex::then(fun) | tt::sync_wait())),
            85);
#endif
    }
}

void test_bulk()
{
    std::vector<int> const ns = {0, 1, 10, 43};

    for (int n : ns)
    {
        std::vector<int> v(n, 0);
        hpx::thread::id parent_id = hpx::this_thread::get_id();

#if defined(HPX_HAVE_STDEXEC)
        tt::sync_wait(
            ex::schedule(ex::thread_pool_scheduler{}) | ex::bulk(n, [&](int i) {
                ++v[i];
                HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
            }));
#else
        ex::schedule(ex::thread_pool_scheduler{}) | ex::bulk(n, [&](int i) {
            ++v[i];
            HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
        }) | tt::sync_wait();
#endif

        for (int i = 0; i < n; ++i)
        {
            HPX_TEST_EQ(v[i], 1);
        }
    }

    for (auto n : ns)
    {
        std::vector<int> v(n, -1);
        hpx::thread::id parent_id = hpx::this_thread::get_id();

#if defined(HPX_HAVE_STDEXEC)
        auto v_out = hpx::get<0>(*(tt::sync_wait(
            ex::transfer_just(ex::thread_pool_scheduler{}, std::move(v)) |
            ex::bulk(n, [&parent_id](int i, std::vector<int>& v) {
                v[i] = i;
                HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
            }))));
#else
        auto v_out = hpx::get<0>(
            *(ex::transfer_just(ex::thread_pool_scheduler{}, std::move(v)) |
                ex::bulk(n,
                    [&parent_id](int i, std::vector<int>& v) {
                        v[i] = i;
                        HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
                    }) |
                tt::sync_wait()));
#endif

        for (int i = 0; i < n; ++i)
        {
            HPX_TEST_EQ(v_out[i], i);
        }
    }

    {
        std::unordered_set<std::string> string_map;
        std::vector<std::string> v = {"hello", "brave", "new", "world"};
        std::vector<std::string> v_ref = v;

        hpx::mutex mtx;
#if defined(HPX_HAVE_STDEXEC)
        tt::sync_wait(ex::schedule(ex::thread_pool_scheduler{}) |
            ex::bulk(std::move(v), [&](std::string const& s) {
                std::lock_guard lk(mtx);
                string_map.insert(s);
            }));
#else
        ex::schedule(ex::thread_pool_scheduler{}) |
            ex::bulk(std::move(v),
                [&](std::string const& s) {
                    std::lock_guard lk(mtx);
                    string_map.insert(s);
                }) |
            tt::sync_wait();
#endif

        for (auto const& s : v_ref)
        {
            HPX_TEST(string_map.find(s) != string_map.end());
        }
    }

    for (auto n : ns)
    {
        int i_fail = 3;

        std::vector<int> v(n, -1);
        bool const expect_exception = n > i_fail;

        try
        {
#if defined(HPX_HAVE_STDEXEC)
            tt::sync_wait(ex::transfer_just(ex::thread_pool_scheduler{}) |
                ex::bulk(n, [&v, i_fail](int i) {
                    if (i == i_fail)
                    {
                        throw std::runtime_error("error");
                    }
                    v[i] = i;
                }));
#else
            ex::transfer_just(ex::thread_pool_scheduler{}) |
                ex::bulk(n,
                    [&v, i_fail](int i) {
                        if (i == i_fail)
                        {
                            throw std::runtime_error("error");
                        }
                        v[i] = i;
                    }) |
                tt::sync_wait();

#endif

            if (expect_exception)
            {
                HPX_TEST(false);
            }
        }
        catch (std::runtime_error const& e)
        {
            if (!expect_exception)
            {
                HPX_TEST(false);
            }

            HPX_TEST(std::string(e.what()).find("error") == 0);
        }

        if (expect_exception)
        {
            HPX_TEST_EQ(v[i_fail], -1);
        }
        else
        {
            for (int i = 0; i < n; ++i)
            {
                HPX_TEST_EQ(v[i], i);
            }
        }
    }
}
// NOLINTEND(bugprone-unchecked-optional-access)

void test_completion_scheduler()
{
    namespace ex = hpx::execution::experimental;
    {
        auto sender = ex::schedule(ex::thread_pool_scheduler{});
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(
#if defined(HPX_HAVE_STDEXEC)
                ex::get_env(sender)
#else
                sender
#endif
            );
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>,
                ex::thread_pool_scheduler>,
            "the completion scheduler should be a thread_pool_scheduler");
        // clang-format off
    } // Clang format bug: thinks this '}' is the end of a namespace.
    // clang-format on

    {
        auto sender =
            ex::then(ex::schedule(ex::thread_pool_scheduler{}), []() {});
        using hpx::functional::tag_invoke;
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(
#if defined(HPX_HAVE_STDEXEC)
                ex::get_env(sender)
#else
                sender
#endif
            );
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>,
                ex::thread_pool_scheduler>,
            "the completion scheduler should be a thread_pool_scheduler");
    }

    {
        auto sender = ex::transfer_just(ex::thread_pool_scheduler{}, 42);
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(
#if defined(HPX_HAVE_STDEXEC)
                ex::get_env(sender)
#else
                sender
#endif
            );
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>,
                ex::thread_pool_scheduler>,
            "the completion scheduler should be a thread_pool_scheduler");
    }

    {
        auto sender =
            ex::bulk(ex::schedule(ex::thread_pool_scheduler{}), 10, [](int) {});
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(
#if defined(HPX_HAVE_STDEXEC)
                ex::get_env(sender)
#else
                sender
#endif
            );
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>,
                ex::thread_pool_scheduler>,
            "the completion scheduler should be a thread_pool_scheduler");
    }

    {
        auto sender = ex::then(
            ex::bulk(ex::transfer_just(ex::thread_pool_scheduler{}, 42), 10,
                [](int, int) {}),
            [](int) {});
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(
#if defined(HPX_HAVE_STDEXEC)
                ex::get_env(sender)
#else
                sender
#endif
            );
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>,
                ex::thread_pool_scheduler>,
            "the completion scheduler should be a thread_pool_scheduler");
    }

    {
        auto sender = ex::bulk(
            ex::then(
                ex::transfer_just(ex::thread_pool_scheduler{}, 42), [](int) {}),
            10, [](int, int) {});
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(
#if defined(HPX_HAVE_STDEXEC)
                ex::get_env(sender)
#else
                sender
#endif
            );
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>,
                ex::thread_pool_scheduler>,
            "the completion scheduler should be a thread_pool_scheduler");
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_execute();
    test_sender_receiver_basic();
    test_sender_receiver_then();
    test_sender_receiver_then_wait();
    test_sender_receiver_then_sync_wait();
    test_sender_receiver_then_arguments();
    test_properties();
    test_transfer_basic();
    test_transfer_arguments();
    test_just_void();
    test_just_one_arg();
    test_just_two_args();
    test_transfer_just_void();
    test_transfer_just_one_arg();
    test_transfer_just_two_args();
    test_when_all();
    test_future_sender();
    test_keep_future_sender();
    test_ensure_started();
    test_ensure_started_when_all();
    test_split();
    test_split_when_all();
    test_let_value();
    test_let_error();
    test_detach();
    test_bulk();
    test_completion_scheduler();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
#else
int main()
{
    return 0;
}
#endif
