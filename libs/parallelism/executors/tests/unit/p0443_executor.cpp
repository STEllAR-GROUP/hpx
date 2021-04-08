//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/condition_variable.hpp>
#include <hpx/execution.hpp>
#include <hpx/functional.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

namespace ex = hpx::execution::experimental;

///////////////////////////////////////////////////////////////////////////////
void test_execute()
{
    hpx::thread::id parent_id = hpx::this_thread::get_id();

    ex::executor exec{};
    ex::execute(exec,
        [parent_id]() { HPX_TEST_NEQ(hpx::this_thread::get_id(), parent_id); });
}

struct check_context_receiver
{
    hpx::thread::id parent_id;
    hpx::lcos::local::condition_variable& cond;
    std::atomic<bool>& executed;

    template <typename E>
    void set_error(E&&) noexcept
    {
        HPX_TEST(false);
    }

    void set_done() noexcept
    {
        HPX_TEST(false);
    };

    template <typename... Ts>
    void set_value(Ts&&...) noexcept
    {
        HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
        HPX_TEST_NEQ(hpx::thread::id(hpx::threads::invalid_thread_id),
            hpx::this_thread::get_id());
        executed = true;
        cond.notify_one();
    }
};

void test_sender_receiver_basic()
{
    hpx::thread::id parent_id = hpx::this_thread::get_id();
    hpx::lcos::local::mutex mtx;
    hpx::lcos::local::condition_variable cond;
    std::atomic<bool> executed{false};

    ex::executor exec{};

    auto begin = ex::schedule(exec);
    auto os = ex::connect(
        std::move(begin), check_context_receiver{parent_id, cond, executed});
    ex::start(os);

    {
        std::unique_lock<hpx::lcos::local::mutex> l{mtx};
        cond.wait(l, [&]() { return executed.load(); });
    }

    HPX_TEST(executed);
}

void test_sender_receiver_basic2()
{
    hpx::thread::id parent_id = hpx::this_thread::get_id();
    hpx::lcos::local::mutex mtx;
    hpx::lcos::local::condition_variable cond;
    std::atomic<bool> executed{false};

    auto os = ex::connect(
        ex::executor{}, check_context_receiver{parent_id, cond, executed});
    ex::start(os);

    {
        std::unique_lock<hpx::lcos::local::mutex> l{mtx};
        cond.wait(l, [&]() { return executed.load(); });
    }

    HPX_TEST(executed);
}

hpx::thread::id sender_receiver_transform_thread_id;

void test_sender_receiver_transform()
{
    ex::executor exec{};
    hpx::thread::id parent_id = hpx::this_thread::get_id();
    hpx::lcos::local::mutex mtx;
    hpx::lcos::local::condition_variable cond;
    std::atomic<bool> executed{false};

    auto begin = ex::schedule(exec);
    auto work1 = ex::transform(std::move(begin), [=]() {
        sender_receiver_transform_thread_id = hpx::this_thread::get_id();
        HPX_TEST_NEQ(sender_receiver_transform_thread_id, parent_id);
    });
    auto work2 = ex::transform(std::move(work1), []() {
        HPX_TEST_EQ(
            sender_receiver_transform_thread_id, hpx::this_thread::get_id());
    });
    auto os = ex::connect(
        std::move(work2), check_context_receiver{parent_id, cond, executed});
    ex::start(os);

    {
        std::unique_lock<hpx::lcos::local::mutex> l{mtx};
        cond.wait(l, [&]() { return executed.load(); });
    }

    HPX_TEST(executed);
}

void test_sender_receiver_transform_wait()
{
    ex::executor exec{};
    hpx::thread::id parent_id = hpx::this_thread::get_id();
    std::atomic<std::size_t> transform_count{0};
    std::atomic<bool> executed{false};

    auto begin = ex::schedule(exec);
    auto work1 =
        ex::transform(std::move(begin), [&transform_count, parent_id]() {
            sender_receiver_transform_thread_id = hpx::this_thread::get_id();
            HPX_TEST_NEQ(sender_receiver_transform_thread_id, parent_id);
            ++transform_count;
        });
    auto work2 =
        ex::transform(std::move(work1), [&transform_count, &executed]() {
            HPX_TEST_EQ(sender_receiver_transform_thread_id,
                hpx::this_thread::get_id());
            ++transform_count;
            executed = true;
        });
    ex::sync_wait(std::move(work2));
    HPX_TEST_EQ(transform_count, std::size_t(2));
    HPX_TEST(executed);
}

void test_sender_receiver_transform_sync_wait()
{
    ex::executor exec{};
    hpx::thread::id parent_id = hpx::this_thread::get_id();
    std::atomic<std::size_t> transform_count{0};

    auto begin = ex::schedule(exec);
    auto work =
        ex::transform(std::move(begin), [&transform_count, parent_id]() {
            sender_receiver_transform_thread_id = hpx::this_thread::get_id();
            HPX_TEST_NEQ(sender_receiver_transform_thread_id, parent_id);
            ++transform_count;
            return 42;
        });
    auto result = ex::sync_wait(std::move(work));
    HPX_TEST_EQ(transform_count, std::size_t(1));
    static_assert(
        std::is_same<int, typename std::decay<decltype(result)>::type>::value,
        "result should be an int");
    HPX_TEST_EQ(result, 42);
}

void test_sender_receiver_transform_arguments()
{
    ex::executor exec{};
    hpx::thread::id parent_id = hpx::this_thread::get_id();
    std::atomic<std::size_t> transform_count{0};

    auto begin = ex::schedule(exec);
    auto work1 =
        ex::transform(std::move(begin), [&transform_count, parent_id]() {
            sender_receiver_transform_thread_id = hpx::this_thread::get_id();
            HPX_TEST_NEQ(sender_receiver_transform_thread_id, parent_id);
            ++transform_count;
            return 3;
        });
    auto work2 = ex::transform(
        std::move(work1), [&transform_count](int x) -> std::string {
            HPX_TEST_EQ(sender_receiver_transform_thread_id,
                hpx::this_thread::get_id());
            ++transform_count;
            return std::string("hello") + std::to_string(x);
        });
    auto work3 =
        ex::transform(std::move(work2), [&transform_count](std::string s) {
            HPX_TEST_EQ(sender_receiver_transform_thread_id,
                hpx::this_thread::get_id());
            ++transform_count;
            return 2 * s.size();
        });
    auto result = ex::sync_wait(std::move(work3));
    HPX_TEST_EQ(transform_count, std::size_t(3));
    static_assert(std::is_same<std::size_t,
                      typename std::decay<decltype(result)>::type>::value,
        "result should be a std::size_t");
    HPX_TEST_EQ(result, std::size_t(12));
}

template <typename F>
struct callback_receiver
{
    std::decay_t<F> f;
    hpx::lcos::local::condition_variable& cond;
    std::atomic<bool>& executed;

    template <typename E>
    void set_error(E&&) noexcept
    {
        HPX_TEST(false);
    }

    void set_done() noexcept
    {
        HPX_TEST(false);
    };

    template <typename... Ts>
    void set_value(Ts&&...) noexcept
    {
        HPX_INVOKE(f);
        executed = true;
        cond.notify_one();
    }
};

void test_properties()
{
    ex::executor exec{};
    hpx::lcos::local::mutex mtx;
    hpx::lcos::local::condition_variable cond;
    std::atomic<bool> executed{false};

    constexpr std::array<hpx::threads::thread_priority, 3> priorities{
        {hpx::threads::thread_priority::low,
            hpx::threads::thread_priority::normal,
            hpx::threads::thread_priority::high}};

    for (auto const prio : priorities)
    {
        auto exec_prop = ex::make_with_priority(exec, prio);
        HPX_TEST_EQ(ex::get_priority(exec_prop), prio);

        auto check = [prio]() {
            HPX_TEST_EQ(prio, hpx::this_thread::get_priority());
        };
        executed = false;
        auto os = ex::connect(ex::schedule(exec_prop),
            callback_receiver<decltype(check)>{check, cond, executed});
        ex::start(os);
        {
            std::unique_lock<hpx::lcos::local::mutex> l{mtx};
            cond.wait(l, [&]() { return executed.load(); });
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
        auto exec_prop = ex::make_with_stacksize(exec, stacksize);
        HPX_TEST_EQ(ex::get_stacksize(exec_prop), stacksize);

        auto check = [stacksize]() {
            HPX_TEST_EQ(stacksize,
                hpx::threads::get_thread_id_data(hpx::threads::get_self_id())
                    ->get_stack_size_enum());
        };
        executed = false;
        auto os = ex::connect(ex::schedule(exec_prop),
            callback_receiver<decltype(check)>{check, cond, executed});
        ex::start(os);
        {
            std::unique_lock<hpx::lcos::local::mutex> l{mtx};
            cond.wait(l, [&]() { return executed.load(); });
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
        auto exec_prop = ex::make_with_hint(exec, hint);
        HPX_TEST(ex::get_hint(exec_prop) == hint);

        // A hint is not guaranteed to be respected, so we only check that the
        // executor holds the property.
    }
}

void test_on_basic()
{
    ex::executor exec{};
    hpx::thread::id parent_id = hpx::this_thread::get_id();
    hpx::thread::id current_id;

    auto begin = ex::schedule(exec);
    auto work1 = ex::transform(begin, [=, &current_id]() {
        current_id = hpx::this_thread::get_id();
        HPX_TEST_NEQ(current_id, parent_id);
    });
    auto work2 = ex::transform(work1, [=, &current_id]() {
        HPX_TEST_EQ(current_id, hpx::this_thread::get_id());
    });
    auto on1 = ex::on(work2, exec);
    auto work3 = ex::transform(on1, [=, &current_id]() {
        hpx::thread::id new_id = hpx::this_thread::get_id();
        HPX_TEST_NEQ(current_id, new_id);
        current_id = new_id;
        HPX_TEST_NEQ(current_id, parent_id);
    });
    auto work4 = ex::transform(work3, [=, &current_id]() {
        HPX_TEST_EQ(current_id, hpx::this_thread::get_id());
    });
    auto on2 = ex::on(work4, exec);
    auto work5 = ex::transform(on2, [=, &current_id]() {
        hpx::thread::id new_id = hpx::this_thread::get_id();
        HPX_TEST_NEQ(current_id, new_id);
        current_id = new_id;
        HPX_TEST_NEQ(current_id, parent_id);
    });

    ex::sync_wait(work5);
}

void test_on_arguments()
{
    ex::executor exec{};
    hpx::thread::id parent_id = hpx::this_thread::get_id();
    hpx::thread::id current_id;

    auto begin = ex::schedule(exec);
    auto work1 = ex::transform(begin, [=, &current_id]() {
        current_id = hpx::this_thread::get_id();
        HPX_TEST_NEQ(current_id, parent_id);
        return 3;
    });
    auto work2 = ex::transform(work1, [=, &current_id](int x) {
        HPX_TEST_EQ(current_id, hpx::this_thread::get_id());
        return x / 2.0;
    });
    auto on1 = ex::on(work2, exec);
    auto work3 = ex::transform(on1, [=, &current_id](double x) {
        hpx::thread::id new_id = hpx::this_thread::get_id();
        HPX_TEST_NEQ(current_id, new_id);
        current_id = new_id;
        HPX_TEST_NEQ(current_id, parent_id);
        return x / 2;
    });
    auto work4 = ex::transform(work3, [=, &current_id](int x) {
        HPX_TEST_EQ(current_id, hpx::this_thread::get_id());
        return "result: " + std::to_string(x);
    });
    auto on2 = ex::on(work4, exec);
    auto work5 = ex::transform(on2, [=, &current_id](std::string s) {
        hpx::thread::id new_id = hpx::this_thread::get_id();
        HPX_TEST_NEQ(current_id, new_id);
        current_id = new_id;
        HPX_TEST_NEQ(current_id, parent_id);
        return s + "!";
    });

    auto result = ex::sync_wait(work5);
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
        auto work1 = ex::transform(begin, [parent_id]() {
            HPX_TEST_EQ(parent_id, hpx::this_thread::get_id());
        });
        ex::sync_wait(work1);
    }

    {
        hpx::thread::id parent_id = hpx::this_thread::get_id();

        auto begin = ex::just();
        auto on1 = ex::on(begin, ex::executor{});
        auto work1 = ex::transform(on1, [parent_id]() {
            HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
        });
        ex::sync_wait(work1);
    }
}

void test_just_one_arg()
{
    {
        hpx::thread::id parent_id = hpx::this_thread::get_id();

        auto begin = ex::just(3);
        auto work1 = ex::transform(begin, [parent_id](int x) {
            HPX_TEST_EQ(parent_id, hpx::this_thread::get_id());
            HPX_TEST_EQ(x, 3);
        });
        ex::sync_wait(work1);
    }

    {
        hpx::thread::id parent_id = hpx::this_thread::get_id();

        auto begin = ex::just(3);
        auto on1 = ex::on(begin, ex::executor{});
        auto work1 = ex::transform(on1, [parent_id](int x) {
            HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
            HPX_TEST_EQ(x, 3);
        });
        ex::sync_wait(work1);
    }
}

void test_just_two_args()
{
    {
        hpx::thread::id parent_id = hpx::this_thread::get_id();

        auto begin = ex::just(3, std::string("hello"));
        auto work1 = ex::transform(begin, [parent_id](int x, std::string y) {
            HPX_TEST_EQ(parent_id, hpx::this_thread::get_id());
            HPX_TEST_EQ(x, 3);
            HPX_TEST_EQ(y, std::string("hello"));
        });
        ex::sync_wait(work1);
    }

    {
        hpx::thread::id parent_id = hpx::this_thread::get_id();

        auto begin = ex::just(3, std::string("hello"));
        auto on1 = ex::on(begin, ex::executor{});
        auto work1 = ex::transform(on1, [parent_id](int x, std::string y) {
            HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
            HPX_TEST_EQ(x, 3);
            HPX_TEST_EQ(y, std::string("hello"));
        });
        ex::sync_wait(work1);
    }
}

void test_just_on_void()
{
    hpx::thread::id parent_id = hpx::this_thread::get_id();

    auto begin = ex::just_on(ex::executor{});
    auto work1 = ex::transform(begin,
        [parent_id]() { HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id()); });
    ex::sync_wait(work1);
}

void test_just_on_one_arg()
{
    hpx::thread::id parent_id = hpx::this_thread::get_id();

    auto begin = ex::just_on(ex::executor{}, 3);
    auto work1 = ex::transform(begin, [parent_id](int x) {
        HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
        HPX_TEST_EQ(x, 3);
    });
    ex::sync_wait(work1);
}

void test_just_on_two_args()
{
    hpx::thread::id parent_id = hpx::this_thread::get_id();

    auto begin = ex::just_on(ex::executor{}, 3, std::string("hello"));
    auto work1 = ex::transform(begin, [parent_id](int x, std::string y) {
        HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
        HPX_TEST_EQ(x, 3);
        HPX_TEST_EQ(y, std::string("hello"));
    });
    ex::sync_wait(work1);
}

void test_when_all()
{
#if defined(HPX_HAVE_CXX17_STD_VARIANT)
    ex::executor exec{};

    {
        hpx::thread::id parent_id = hpx::this_thread::get_id();

        auto begin1 = ex::schedule(exec);
        auto work1 = ex::transform(begin1, [parent_id]() {
            HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
            return 42;
        });

        auto begin2 = ex::schedule(exec);
        auto work2 = ex::transform(begin2, [parent_id]() {
            HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
            return std::string("hello");
        });

        auto begin3 = ex::schedule(exec);
        auto work3 = ex::transform(begin3, [parent_id]() {
            HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
            return 3.14;
        });

        auto when1 = ex::when_all(work1, work2, work3);

        std::atomic<bool> executed{false};
        auto transform1 = ex::transform(
            when1, [parent_id, &executed](int x, std::string y, double z) {
                HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
                HPX_TEST_EQ(x, 42);
                HPX_TEST_EQ(y, std::string("hello"));
                HPX_TEST_EQ(z, 3.14);
                executed = true;
            });
        ex::sync_wait(transform1);
        HPX_TEST(executed);
    }

    {
        hpx::thread::id parent_id = hpx::this_thread::get_id();

        // The exception is likely to be thrown before set_value from the second
        // sender is called because the second sender sleeps.
        auto begin1 = ex::schedule(exec);
        auto work1 = ex::transform(begin1, [parent_id]() {
            HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
            throw std::runtime_error("error");
            return 42;
        });

        auto begin2 = ex::schedule(exec);
        auto work2 = ex::transform(begin2, [parent_id]() {
            HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return std::string("hello");
        });

        auto when1 = ex::when_all(work1, work2);
        auto transform1 =
            ex::transform(when1, [parent_id](int x, std::string y) {
                HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
                HPX_TEST_EQ(x, 42);
                HPX_TEST_EQ(y, std::string("hello"));
            });

        bool exception_thrown = false;

        try
        {
            ex::sync_wait(transform1);
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
        auto begin1 = ex::schedule(exec);
        auto work1 = ex::transform(begin1, [parent_id]() {
            HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            throw std::runtime_error("error");
            return 42;
        });

        auto begin2 = ex::schedule(exec);
        auto work2 = ex::transform(begin2, [parent_id]() {
            HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
            return std::string("hello");
        });

        auto when1 = ex::when_all(work1, work2);
        auto transform1 =
            ex::transform(when1, [parent_id](int x, std::string y) {
                HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
                HPX_TEST_EQ(x, 42);
                HPX_TEST_EQ(y, std::string("hello"));
            });

        bool exception_thrown = false;

        try
        {
            ex::sync_wait(transform1);
        }
        catch (std::runtime_error const& e)
        {
            HPX_TEST_EQ(std::string(e.what()), std::string("error"));
            exception_thrown = true;
        }

        HPX_TEST(exception_thrown);
    }
#endif
}

void test_future_sender()
{
    // futures as senders
    {
        std::atomic<bool> called{false};
        auto f = hpx::async([&]() { called = true; });

        static_assert(ex::is_sender_v<std::decay_t<decltype(f)>>,
            "a future should be a sender");
        static_assert(
            std::is_void<decltype(ex::sync_wait(std::move(f)))>::value,
            "sync_wait should return void");

        ex::sync_wait(std::move(f));
        HPX_TEST(called);

        bool exception_thrown = false;
        try
        {
            ex::sync_wait(std::move(f));
        }
        catch (...)
        {
            exception_thrown = true;
        };
        HPX_TEST(exception_thrown);
    }

    {
        std::atomic<bool> called{false};
        auto f = hpx::async([&]() {
            called = true;
            return 42;
        });

        static_assert(ex::is_sender_v<std::decay_t<decltype(f)>>,
            "a future should be a sender");

        HPX_TEST_EQ(ex::sync_wait(std::move(f)), 42);
        HPX_TEST(called);

        bool exception_thrown = false;
        try
        {
            ex::sync_wait(std::move(f));
        }
        catch (...)
        {
            exception_thrown = true;
        };
        HPX_TEST(exception_thrown);
    }

    {
        std::atomic<bool> called{false};
        auto f = hpx::async([&]() {
            called = true;
            return 42;
        });

        HPX_TEST_EQ(ex::sync_wait(ex::transform(
                        std::move(f), [](int x) { return x / 2; })),
            21);
        HPX_TEST(called);
    }

    {
        std::atomic<std::size_t> calls{0};
        auto sf = hpx::async([&]() { ++calls; }).share();

        static_assert(ex::is_sender_v<std::decay_t<decltype(sf)>>,
            "a shared_future should be a sender");
        static_assert(
            std::is_void<decltype(ex::sync_wait(std::move(sf)))>::value,
            "sync_wait should return void");

        ex::sync_wait(sf);
        ex::sync_wait(sf);
        ex::sync_wait(std::move(sf));
        HPX_TEST_EQ(calls, std::size_t(1));

        bool exception_thrown = false;
        try
        {
            ex::sync_wait(sf);
        }
        catch (...)
        {
            exception_thrown = true;
        };
        HPX_TEST(exception_thrown);
    }

    {
        std::atomic<std::size_t> calls{0};
        auto sf = hpx::async([&]() {
            ++calls;
            return 42;
        }).share();

        static_assert(ex::is_sender_v<std::decay_t<decltype(sf)>>,
            "a shared_future should be a sender");

        HPX_TEST_EQ(ex::sync_wait(sf), 42);
        HPX_TEST_EQ(ex::sync_wait(sf), 42);
        HPX_TEST_EQ(ex::sync_wait(std::move(sf)), 42);
        HPX_TEST_EQ(calls, std::size_t(1));

        bool exception_thrown = false;
        try
        {
            ex::sync_wait(sf);
        }
        catch (...)
        {
            exception_thrown = true;
        };
        HPX_TEST(exception_thrown);
    }

    // senders as futures
    {
        auto s = ex::just(3);
        auto f = ex::make_future(std::move(s));
        HPX_TEST_EQ(f.get(), 3);
    }

    {
        auto s = ex::just_on(ex::executor{}, 3);
        auto f = ex::make_future(std::move(s));
        HPX_TEST_EQ(f.get(), 3);
    }

    {
        std::atomic<bool> called{false};
        auto s = ex::schedule(ex::executor{}) |
            ex::transform([&] { called = true; });
        auto f = ex::make_future(std::move(s));
        f.get();
        HPX_TEST(called);
    }

#if defined(HPX_HAVE_CXX17_STD_VARIANT)
    {
        auto s1 = ex::just_on(ex::executor{}, std::size_t(42));
        auto s2 = ex::just_on(ex::executor{}, 3.14);
        auto s3 = ex::just_on(ex::executor{}, std::string("hello"));
        auto f = ex::make_future(ex::transform(
            ex::when_all(std::move(s1), std::move(s2), std::move(s3)),
            [](std::size_t x, double, std::string z) { return z.size() + x; }));
        HPX_TEST_EQ(f.get(), std::size_t(47));
    }
#endif

    // mixing senders and futures
    {
        HPX_TEST_EQ(
            ex::sync_wait(ex::make_future(ex::just_on(ex::executor{}, 42))),
            42);
    }

    {
        HPX_TEST_EQ(ex::make_future(
                        ex::on(hpx::async([]() { return 42; }), ex::executor{}))
                        .get(),
            42);
    }

#if defined(HPX_HAVE_CXX17_STD_VARIANT)
    {
        auto s1 = ex::just_on(ex::executor{}, std::size_t(42));
        auto s2 = ex::just_on(ex::executor{}, 3.14);
        auto s3 = ex::just_on(ex::executor{}, std::string("hello"));
        auto f = ex::make_future(ex::transform(
            ex::when_all(std::move(s1), std::move(s2), std::move(s3)),
            [](std::size_t x, double, std::string z) { return z.size() + x; }));
        auto sf = f.then([](auto&& f) { return f.get() - 40; }).share();
        auto t1 = sf.then([](auto&& sf) { return sf.get() + 1; });
        auto t2 = sf.then([](auto&& sf) { return sf.get() + 2; });
        auto t1s =
            ex::transform(std::move(t1), [](std::size_t x) { return x + 1; });
        auto t1f = ex::make_future(std::move(t1s));
        auto last = hpx::dataflow(
            hpx::util::unwrapping(
                [](std::size_t x, std::size_t y) { return x + y; }),
            t1f, t2);

        HPX_TEST_EQ(last.get(), std::size_t(18));
    }
#endif
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_execute();
    test_sender_receiver_basic();
    test_sender_receiver_basic2();
    test_sender_receiver_transform();
    test_sender_receiver_transform_wait();
    test_sender_receiver_transform_sync_wait();
    test_sender_receiver_transform_arguments();
    test_properties();
    test_on_basic();
    test_on_arguments();
    test_just_void();
    test_just_one_arg();
    test_just_two_args();
    test_just_on_void();
    test_just_on_one_arg();
    test_just_on_two_args();
    test_when_all();
    test_future_sender();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(
        hpx::init(argc, argv), 0, "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
