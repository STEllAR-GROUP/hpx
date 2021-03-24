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
#include <cstddef>
#include <string>
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
    auto work = ex::connect(
        std::move(begin), check_context_receiver{parent_id, cond, executed});
    ex::start(std::move(work));

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

    ex::start(ex::connect(
        ex::executor{}, check_context_receiver{parent_id, cond, executed}));

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
    auto end = ex::connect(
        std::move(work2), check_context_receiver{parent_id, cond, executed});
    ex::start(std::move(end));

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
        ex::start(ex::connect(ex::schedule(exec_prop),
            callback_receiver<decltype(check)>{check, cond, executed}));
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
        ex::start(ex::connect(ex::schedule(exec_prop),
            callback_receiver<decltype(check)>{check, cond, executed}));
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

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(
        hpx::init(argc, argv), 0, "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
