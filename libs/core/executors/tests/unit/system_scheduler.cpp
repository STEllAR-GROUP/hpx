//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/local/execution.hpp>
#include <hpx/local/init.hpp>
#include <hpx/local/latch.hpp>
#include <hpx/local/thread.hpp>
#include <hpx/modules/testing.hpp>

#include "algorithm_test_utils.hpp"

namespace ex = hpx::execution::experimental;
namespace tt = hpx::this_thread::experimental;

void test_concepts()
{
    ex::system_context ctx;

    auto sched = ctx.get_scheduler();
    static_assert(ex::is_scheduler_v<decltype(sched)>,
        "ex::is_scheduler_v<decltype(sched)>");

    auto s = ex::schedule(sched);
    static_assert(
        std::is_same_v<ex::schedule_result_t<decltype(sched)>, decltype(s)>,
        "ex::schedule_result_t<decltype(sched)> must be result of "
        "ex::schedule(sched)");
    static_assert(ex::is_sender_v<decltype(s)>, "ex::is_sender_v<decltype(s)>");
    static_assert(
        ex::is_sender_of_v<decltype(s)>, "ex::is_sender_of_v<decltype(s)>");

    static_assert(std::is_same_v<ex::value_types_of_t<decltype(s)>,
                      hpx::variant<hpx::tuple<>>>,
        "ex::value_types_of_t<decltype(s)> must be variant<hpx::tuple<>>");
    static_assert(std::is_same_v<ex::error_types_of_t<decltype(s)>,
                      hpx::variant<std::exception_ptr>>,
        "ex::error_types_of_t<decltype(s)> must be "
        "variant<std::exception_ptr>");
    static_assert(ex::sends_stopped_of_v<decltype(s)>,
        "ex::sends_stopped_of_v<decltype(s)> must be true");

    auto hi = ex::then(std::move(s), [] { return 13; });
    auto add_42 = ex::then(std::move(hi), [](int arg) { return arg + 42; });

    auto result = tt::sync_wait(std::move(add_42));

    HPX_TEST(hpx::get<0>(*result) == 55);
}

void test_compare()
{
    ex::system_context ctx1;
    auto sched11 = ctx1.get_scheduler();
    auto sched12 = ctx1.get_scheduler();

    HPX_TEST(sched11 == sched11);
    HPX_TEST(sched12 == sched12);
    HPX_TEST(sched11 == sched12);

    ex::system_context ctx2;
    auto sched2 = ctx2.get_scheduler();

    HPX_TEST(sched2 == sched2);
    HPX_TEST(sched11 != sched2);
    HPX_TEST(sched12 != sched2);
}

void test_execute()
{
    hpx::thread::id parent_id = hpx::this_thread::get_id();

    hpx::latch l(2);

    ex::system_context ctx;
    auto sched = ctx.get_scheduler();

    ex::execute(sched, [&]() {
        HPX_TEST_NEQ(hpx::this_thread::get_id(), parent_id);
        l.count_down(1);
    });

    l.arrive_and_wait();
}

struct check_context_receiver
{
    hpx::thread::id parent_id;
    hpx::latch& l;
    bool& executed;

    template <typename E>
    friend void tag_invoke(
        ex::set_error_t, check_context_receiver&& r, E&&) noexcept
    {
        r.l.count_down(1);
        HPX_TEST(false);
    }

    friend void tag_invoke(
        ex::set_stopped_t, check_context_receiver&& r) noexcept
    {
        r.l.count_down(1);
        HPX_TEST(false);
    }

    template <typename... Ts>
    friend void tag_invoke(ex::set_value_t, check_context_receiver&& r, Ts&&...)
    {
        HPX_TEST_NEQ(r.parent_id, hpx::this_thread::get_id());
        HPX_TEST_NEQ(hpx::thread::id(hpx::threads::invalid_thread_id),
            hpx::this_thread::get_id());

        r.executed = true;
        r.l.count_down(1);
    }
};

void test_sender_receiver_basic()
{
    hpx::latch l(2);

    ex::system_context ctx;
    auto sched = ctx.get_scheduler();

    hpx::thread::id parent_id = hpx::this_thread::get_id();
    bool executed{false};

    auto begin = ex::schedule(sched);
    auto os = ex::connect(
        std::move(begin), check_context_receiver{parent_id, l, executed});
    ex::start(os);

    l.arrive_and_wait();

    HPX_TEST(executed);
}

hpx::thread::id sender_receiver_then_thread_id;

void test_sender_receiver_then()
{
    hpx::latch l(2);

    ex::system_context ctx;
    auto sched = ctx.get_scheduler();

    hpx::thread::id parent_id = hpx::this_thread::get_id();
    bool executed{false};

    auto begin = ex::schedule(sched);
    auto work1 = ex::then(std::move(begin), [=]() {
        sender_receiver_then_thread_id = hpx::this_thread::get_id();
        HPX_TEST_NEQ(sender_receiver_then_thread_id, parent_id);
    });
    auto work2 = ex::then(std::move(work1), []() {
        HPX_TEST_EQ(sender_receiver_then_thread_id, hpx::this_thread::get_id());
    });
    auto os = ex::connect(
        std::move(work2), check_context_receiver{parent_id, l, executed});
    ex::start(os);

    l.arrive_and_wait();

    HPX_TEST(executed);
}

void test_sender_receiver_then_wait()
{
    ex::system_context ctx;
    auto sched = ctx.get_scheduler();

    hpx::thread::id parent_id = hpx::this_thread::get_id();
    std::atomic<std::size_t> then_count{0};
    bool executed{false};

    auto begin = ex::schedule(sched);

    static_assert(
        ex::detail::is_completion_scheduler_tag_invocable_v<ex::set_value_t,
            decltype(begin), tt::sync_wait_t>);
    auto compl_sched_begin =
        ex::get_completion_scheduler<ex::set_value_t>(begin);
    HPX_TEST(sched == compl_sched_begin);

    auto work1 = ex::then(std::move(begin), [&then_count, parent_id]() {
        sender_receiver_then_thread_id = hpx::this_thread::get_id();
        HPX_TEST_NEQ(sender_receiver_then_thread_id, parent_id);
        ++then_count;
    });

    static_assert(
        ex::detail::is_completion_scheduler_tag_invocable_v<ex::set_value_t,
            decltype(work1), tt::sync_wait_t>);
    auto compl_sched_work1 =
        ex::get_completion_scheduler<ex::set_value_t>(work1);
    HPX_TEST(sched == compl_sched_work1);

    auto work2 = ex::then(std::move(work1), [&then_count, &executed]() {
        HPX_TEST_EQ(sender_receiver_then_thread_id, hpx::this_thread::get_id());
        ++then_count;
        executed = true;
    });

    static_assert(
        ex::detail::is_completion_scheduler_tag_invocable_v<ex::set_value_t,
            decltype(work2), tt::sync_wait_t>);
    auto compl_sched_work2 =
        ex::get_completion_scheduler<ex::set_value_t>(work2);
    HPX_TEST(sched == compl_sched_work2);

    tt::sync_wait(std::move(work2));

    HPX_TEST_EQ(then_count, std::size_t(2));
    HPX_TEST(executed);
}

void test_sender_receiver_then_sync_wait()
{
    ex::system_context ctx;
    auto sched = ctx.get_scheduler();

    hpx::thread::id parent_id = hpx::this_thread::get_id();
    std::atomic<std::size_t> then_count{0};

    auto begin = ex::schedule(sched);
    auto work = ex::then(std::move(begin), [&then_count, parent_id]() {
        sender_receiver_then_thread_id = hpx::this_thread::get_id();
        HPX_TEST_NEQ(sender_receiver_then_thread_id, parent_id);
        ++then_count;
        return 42;
    });
    auto work_result = tt::sync_wait(std::move(work));
    auto result = hpx::get<0>(*work_result);
    HPX_TEST_EQ(then_count, std::size_t(1));
    static_assert(std::is_same_v<int, std::decay_t<decltype(result)>>,
        "result should be an int");
    HPX_TEST_EQ(result, 42);
}

void test_sender_receiver_then_arguments()
{
    ex::system_context ctx;
    auto sched = ctx.get_scheduler();

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
    auto work_result = tt::sync_wait(std::move(work3));
    auto result = hpx::get<0>(*work_result);
    HPX_TEST_EQ(then_count, std::size_t(3));
    static_assert(std::is_same_v<std::size_t, std::decay_t<decltype(result)>>,
        "result should be a std::size_t");
    HPX_TEST_EQ(result, std::size_t(12));
}

void test_transfer_basic()
{
    ex::system_context ctx;
    auto sched = ctx.get_scheduler();

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

    tt::sync_wait(std::move(work5));
}

void test_transfer_arguments()
{
    ex::system_context ctx;
    auto sched = ctx.get_scheduler();

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

    auto work_result = tt::sync_wait(work5);
    auto result = hpx::get<0>(*work_result);
    static_assert(std::is_same_v<std::string, std::decay_t<decltype(result)>>,
        "result should be a std::string");
    HPX_TEST_EQ(result, std::string("result: 0!"));
}

void test_just_void()
{
    ex::system_context ctx;

    hpx::thread::id parent_id = hpx::this_thread::get_id();

    auto begin = ex::just();
    auto transfer1 = ex::transfer(begin, ctx.get_scheduler());
    auto work1 = ex::then(transfer1,
        [parent_id]() { HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id()); });

    tt::sync_wait(work1);
}

void test_just_one_arg()
{
    ex::system_context ctx;

    hpx::thread::id parent_id = hpx::this_thread::get_id();

    auto begin = ex::just(3);
    auto transfer1 = ex::transfer(begin, ctx.get_scheduler());
    auto work1 = ex::then(transfer1, [parent_id](int x) {
        HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
        HPX_TEST_EQ(x, 3);
    });

    tt::sync_wait(work1);
}

void test_just_two_args()
{
    ex::system_context ctx;

    hpx::thread::id parent_id = hpx::this_thread::get_id();

    auto begin = ex::just(3, std::string("hello"));
    auto transfer1 = ex::transfer(begin, ctx.get_scheduler());
    auto work1 = ex::then(transfer1, [parent_id](int x, std::string y) {
        HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
        HPX_TEST_EQ(x, 3);
        HPX_TEST_EQ(y, std::string("hello"));
    });

    tt::sync_wait(work1);
}

void test_transfer_just_void()
{
    ex::system_context ctx;
    auto sched = ctx.get_scheduler();

    hpx::thread::id parent_id = hpx::this_thread::get_id();

    auto begin = ex::transfer_just(sched);
    auto work1 = ex::then(begin,
        [parent_id]() { HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id()); });

    tt::sync_wait(work1);
}

void test_transfer_just_one_arg()
{
    ex::system_context ctx;
    auto sched = ctx.get_scheduler();

    hpx::thread::id parent_id = hpx::this_thread::get_id();

    auto begin = ex::transfer_just(sched, 3);
    auto work1 = ex::then(begin, [parent_id](int x) {
        HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
        HPX_TEST_EQ(x, 3);
    });

    tt::sync_wait(work1);
}

void test_transfer_just_two_args()
{
    ex::system_context ctx;
    auto sched = ctx.get_scheduler();

    hpx::thread::id parent_id = hpx::this_thread::get_id();

    auto begin = ex::transfer_just(sched, 3, std::string("hello"));
    auto work1 = ex::then(begin, [parent_id](int x, std::string y) {
        HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
        HPX_TEST_EQ(x, 3);
        HPX_TEST_EQ(y, std::string("hello"));
    });

    tt::sync_wait(work1);
}

void test_completion_scheduler()
{
    ex::system_context ctx;
    auto sched = ctx.get_scheduler();

    {
        auto sender = ex::schedule(sched);
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(sender);
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>,
                decltype(sched)>,
            "the completion scheduler should be a run_pool_scheduler");
    }

    {
        auto sender = ex::then(ex::schedule(sched), []() {});
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(sender);
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>,
                decltype(sched)>,
            "the completion scheduler should be a run_pool_scheduler");
    }

    {
        auto sender = ex::transfer_just(sched, 42);
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(sender);
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>,
                decltype(sched)>,
            "the completion scheduler should be a run_pool_scheduler");
    }

    {
        auto sender = ex::bulk(ex::schedule(sched), 10, [](int) {});
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(sender);
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>,
                decltype(sched)>,
            "the completion scheduler should be a run_pool_scheduler");
    }

    {
        auto sender = ex::then(
            ex::bulk(ex::transfer_just(sched, 42), 10, [](int, int) {}),
            [](int) {});
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(sender);
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>,
                decltype(sched)>,
            "the completion scheduler should be a run_pool_scheduler");
    }

    {
        auto sender =
            ex::bulk(ex::then(ex::transfer_just(sched, 42), [](int) {}), 10,
                [](int, int) {});
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(sender);
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>,
                decltype(sched)>,
            "the completion scheduler should be a run_pool_scheduler");
    }
}

// Note: when_all does not propagate the completion scheduler, for this reason
// any senders coming after it need to be explicitly provided with the required
// scheduler again.
void test_when_all()
{
    ex::system_context ctx;
    auto sched = ctx.get_scheduler();

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
        std::move(when1) |
            ex::then([parent_id, &executed](int x, std::string y, double z) {
                HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
                HPX_TEST_EQ(x, 42);
                HPX_TEST_EQ(y, std::string("hello"));
                HPX_TEST_EQ(z, 3.14);
                executed = true;
            }) |
            tt::sync_wait(sched);

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
            ex::when_all(std::move(work1), std::move(work2)) |
                ex::then([parent_id](int x, std::string y) {
                    HPX_TEST_NEQ(parent_id, hpx::this_thread::get_id());
                    HPX_TEST_EQ(x, 42);
                    HPX_TEST_EQ(y, std::string("hello"));
                }) |
                tt::sync_wait(sched);

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
            ex::when_all(std::move(work1), std::move(work2)) |
                ex::then([parent_id](int x, std::string y) {
                    HPX_TEST_EQ(parent_id, hpx::this_thread::get_id());
                    HPX_TEST_EQ(x, 42);
                    HPX_TEST_EQ(y, std::string("hello"));
                }) |
                tt::sync_wait(sched);

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

// Note: make_future does not propagate the completion scheduler, for this
// reason any senders coming after it need to be explicitly provided with the
// required scheduler again.
void test_future_sender()
{
    // senders as futures
    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto s = ex::transfer_just(sched, 3);
        auto f = ex::make_future(std::move(s));
        HPX_TEST_EQ(f.get(), 3);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto f = ex::transfer_just(sched, 3) | ex::make_future();
        HPX_TEST_EQ(f.get(), 3);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        std::atomic<bool> called{false};
        auto s = ex::schedule(sched) | ex::then([&] { called = true; });
        auto f = ex::make_future(std::move(s));
        f.get();
        HPX_TEST(called);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto s1 = ex::transfer_just(sched, std::size_t(42));
        auto s2 = ex::transfer_just(sched, 3.14);
        auto s3 = ex::transfer_just(sched, std::string("hello"));
        auto f = ex::make_future(ex::then(
            ex::when_all(std::move(s1), std::move(s2), std::move(s3)),
            [](std::size_t x, double, std::string z) { return z.size() + x; }));
        HPX_TEST_EQ(f.get(), std::size_t(47));
    }

    // mixing senders and futures
    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto result = tt::sync_wait(sched,
            ex::as_sender(ex::make_future(ex::transfer_just(sched, 42))));
        HPX_TEST_EQ(hpx::get<0>(*result), 42);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto f = hpx::async([]() {
            hpx::this_thread::sleep_for(std::chrono::seconds(1));
            return 42;
        });

        HPX_TEST_EQ(
            ex::make_future(ex::transfer(ex::as_sender(std::move(f)), sched))
                .get(),
            42);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto s1 = ex::transfer_just(sched, std::size_t(42));
        auto s2 = ex::transfer_just(sched, 3.14);
        auto s3 = ex::transfer_just(sched, std::string("hello"));
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
    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        ex::schedule(sched) | ex::ensure_started() | tt::sync_wait();
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto s = ex::transfer_just(sched, 42) | ex::ensure_started();
        auto result = tt::sync_wait(std::move(s));
        HPX_TEST_EQ(hpx::get<0>(*result), 42);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto s = ex::transfer_just(sched, 42) | ex::ensure_started() |
            ex::transfer(sched);
        auto result = tt::sync_wait(std::move(s));
        HPX_TEST_EQ(hpx::get<0>(*result), 42);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto s = ex::transfer_just(sched, 42) | ex::ensure_started();
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(s)), 42);
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(s)), 42);
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(s)), 42);
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(std::move(s))), 42);
    }
}

void test_ensure_started_when_all()
{
    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

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
        }) | ex::ensure_started();
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
        HPX_TEST_EQ(
            hpx::get<0>(*(ex::when_all(succ1, succ2) |
                ex::then([](int const& x, int const& y) { return x + y; }) |
                tt::sync_wait(sched))),
            3);
        HPX_TEST_EQ(first_task_calls, std::size_t(1));
        HPX_TEST_EQ(successor_task_calls, std::size_t(2));
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

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
        }) | ex::ensure_started();
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
        HPX_TEST_EQ(
            hpx::get<0>(*(ex::when_all(succ1, succ2) |
                ex::then([](int const& x, int const& y) { return x + y; }) |
                tt::sync_wait(sched))),
            9);
        HPX_TEST_EQ(first_task_calls, std::size_t(1));
        HPX_TEST_EQ(successor_task_calls, std::size_t(2));
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

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
        }) | ex::ensure_started();
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
        HPX_TEST_EQ(
            hpx::get<0>(*(ex::when_all(succ1, succ2) |
                ex::then([](int const& x, int const& y) { return x + y; }) |
                tt::sync_wait(sched))),
            9);
        HPX_TEST_EQ(first_task_calls, std::size_t(1));
        HPX_TEST_EQ(successor_task_calls, std::size_t(2));
    }
}

void test_split()
{
    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        ex::schedule(sched) | ex::split() | tt::sync_wait();
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto s = ex::transfer_just(sched, 42) | ex::split();
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(std::move(s))), 42);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto s =
            ex::transfer_just(sched, 42) | ex::split() | ex::transfer(sched);
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(std::move(s))), 42);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto s = ex::transfer_just(sched, 42) | ex::split();
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(s)), 42);
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(s)), 42);
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(s)), 42);
        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(std::move(s))), 42);
    }
}

void test_split_when_all()
{
    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        std::atomic<std::size_t> first_task_calls{0};
        std::atomic<std::size_t> successor_task_calls{0};
        auto s = ex::schedule(sched) | ex::then([&]() {
            HPX_TEST_EQ(first_task_calls, std::size_t(0));
            HPX_TEST_EQ(successor_task_calls, std::size_t(0));
            ++first_task_calls;
        }) | ex::split();
        auto succ1 = s | ex::then([&]() {
            HPX_TEST_EQ(first_task_calls, std::size_t(1));
            HPX_TEST_EQ(successor_task_calls, std::size_t(0));
            ++successor_task_calls;
            return 1;
        });
        auto succ2 = s | ex::then([&]() {
            HPX_TEST_EQ(first_task_calls, std::size_t(1));
            HPX_TEST_EQ(successor_task_calls, std::size_t(1));
            ++successor_task_calls;
            return 2;
        });
        HPX_TEST_EQ(
            hpx::get<0>(*(ex::when_all(succ1, succ2) |
                ex::then([](int const& x, int const& y) { return x + y; }) |
                tt::sync_wait(sched))),
            3);
        HPX_TEST_EQ(first_task_calls, std::size_t(1));
        HPX_TEST_EQ(successor_task_calls, std::size_t(2));
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        std::atomic<std::size_t> first_task_calls{0};
        std::atomic<std::size_t> successor_task_calls{0};
        auto s = ex::schedule(sched) | ex::then([&]() {
            HPX_TEST_EQ(first_task_calls, std::size_t(0));
            HPX_TEST_EQ(successor_task_calls, std::size_t(0));
            ++first_task_calls;
            return 3;
        }) | ex::split();
        auto succ1 = s | ex::then([&](int const& x) {
            HPX_TEST_EQ(first_task_calls, std::size_t(1));
            HPX_TEST_EQ(successor_task_calls, std::size_t(0));
            ++successor_task_calls;
            return x + 1;
        });
        auto succ2 = s | ex::then([&](int const& x) {
            HPX_TEST_EQ(first_task_calls, std::size_t(1));
            HPX_TEST_EQ(successor_task_calls, std::size_t(1));
            ++successor_task_calls;
            return x + 2;
        });
        HPX_TEST_EQ(
            hpx::get<0>(*(ex::when_all(succ1, succ2) |
                ex::then([](int const& x, int const& y) { return x + y; }) |
                tt::sync_wait(sched))),
            9);
        HPX_TEST_EQ(first_task_calls, std::size_t(1));
        HPX_TEST_EQ(successor_task_calls, std::size_t(2));
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        std::atomic<std::size_t> first_task_calls{0};
        std::atomic<std::size_t> successor_task_calls{0};
        auto s = ex::schedule(sched) | ex::then([&]() {
            HPX_TEST_EQ(first_task_calls, std::size_t(0));
            HPX_TEST_EQ(successor_task_calls, std::size_t(0));
            ++first_task_calls;
            return 3;
        }) | ex::split();
        auto succ1 = s | ex::transfer(sched) | ex::then([&](int const& x) {
            HPX_TEST_EQ(first_task_calls, std::size_t(1));
            HPX_TEST_EQ(successor_task_calls, std::size_t(0));
            ++successor_task_calls;
            return x + 1;
        });
        auto succ2 = s | ex::transfer(sched) | ex::then([&](int const& x) {
            HPX_TEST_EQ(first_task_calls, std::size_t(1));
            HPX_TEST_EQ(successor_task_calls, std::size_t(1));
            ++successor_task_calls;
            return x + 2;
        });
        HPX_TEST_EQ(
            hpx::get<0>(*(ex::when_all(succ1, succ2) |
                ex::then([](int const& x, int const& y) { return x + y; }) |
                tt::sync_wait(sched))),
            9);
        HPX_TEST_EQ(first_task_calls, std::size_t(1));
        HPX_TEST_EQ(successor_task_calls, std::size_t(2));
    }
}

void test_let_value()
{
    // void predecessor
    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto result = hpx::get<0>(*(ex::schedule(sched) |
            ex::let_value([]() { return ex::just(42); }) | tt::sync_wait()));
        HPX_TEST_EQ(result, 42);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto result = hpx::get<0>(*(ex::schedule(sched) | ex::let_value([=]() {
            return ex::transfer_just(sched, 42);
        }) | tt::sync_wait()));
        HPX_TEST_EQ(result, 42);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto result = hpx::get<0>(*(ex::just() | ex::let_value([=]() {
            return ex::transfer_just(sched, 42);
        }) | tt::sync_wait(sched)));
        HPX_TEST_EQ(result, 42);
    }

    // int predecessor, value ignored
    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto result = hpx::get<0>(*(ex::transfer_just(sched, 43) |
            ex::let_value([](int&) { return ex::just(42); }) |
            tt::sync_wait()));
        HPX_TEST_EQ(result, 42);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto result = hpx::get<0>(*(ex::transfer_just(sched, 43) |
            ex::let_value([=](int&) { return ex::transfer_just(sched, 42); }) |
            tt::sync_wait()));
        HPX_TEST_EQ(result, 42);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto result = hpx::get<0>(*(ex::just(43) | ex::let_value([=](int&) {
            return ex::transfer_just(sched, 42);
        }) | tt::sync_wait(sched)));
        HPX_TEST_EQ(result, 42);
    }

    // int predecessor, value used
    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto result = hpx::get<0>(
            *(ex::transfer_just(sched, 43) | ex::let_value([](int& x) {
                return ex::just(42) | ex::then([&](int y) { return x + y; });
            }) | tt::sync_wait()));
        HPX_TEST_EQ(result, 85);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto result = hpx::get<0>(
            *(ex::transfer_just(sched, 43) | ex::let_value([=](int& x) {
                return ex::transfer_just(sched, 42) |
                    ex::then([&](int y) { return x + y; });
            }) | tt::sync_wait()));
        HPX_TEST_EQ(result, 85);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto result = hpx::get<0>(*(ex::just(43) | ex::let_value([=](int& x) {
            return ex::transfer_just(sched, 42) |
                ex::then([&](int y) { return x + y; });
        }) | tt::sync_wait(sched)));
        HPX_TEST_EQ(result, 85);
    }

    // predecessor throws, let sender is ignored
    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        bool exception_thrown = false;

        try
        {
            ex::transfer_just(sched, 43) | ex::then([](int) -> int {
                throw std::runtime_error("error");
            }) | ex::let_value([](int&) {
                HPX_TEST(false);
                return ex::just(0);
            }) | tt::sync_wait();

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
    // void predecessor
    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        std::atomic<bool> called{false};
        ex::schedule(sched) | ex::then([]() {
            throw std::runtime_error("error");
        }) | ex::let_error([&called](std::exception_ptr& ep) {
            called = true;
            check_exception_ptr_message(ep, "error");
            return ex::just();
        }) | tt::sync_wait();
        HPX_TEST(called);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        std::atomic<bool> called{false};
        ex::schedule(sched) | ex::then([]() {
            throw std::runtime_error("error");
        }) | ex::let_error([=, &called](std::exception_ptr& ep) {
            called = true;
            check_exception_ptr_message(ep, "error");
            return ex::transfer_just(sched);
        }) | tt::sync_wait();
        HPX_TEST(called);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        std::atomic<bool> called{false};
        ex::just() | ex::then([]() { throw std::runtime_error("error"); }) |
            ex::let_error([=, &called](std::exception_ptr& ep) {
                called = true;
                check_exception_ptr_message(ep, "error");
                return ex::transfer_just(sched);
            }) |
            tt::sync_wait(sched);
        HPX_TEST(called);
    }

    // int predecessor
    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto result = hpx::get<0>(*(ex::schedule(sched) | ex::then([]() {
            throw std::runtime_error("error");
            return 43;
        }) | ex::let_error([](std::exception_ptr& ep) {
            check_exception_ptr_message(ep, "error");
            return ex::just(42);
        }) | tt::sync_wait()));
        HPX_TEST_EQ(result, 42);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto result = hpx::get<0>(*(ex::schedule(sched) | ex::then([]() {
            throw std::runtime_error("error");
            return 43;
        }) | ex::let_error([=](std::exception_ptr& ep) {
            check_exception_ptr_message(ep, "error");
            return ex::transfer_just(sched, 42);
        }) | tt::sync_wait()));
        HPX_TEST_EQ(result, 42);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto result = hpx::get<0>(*(ex::just() | ex::then([]() {
            throw std::runtime_error("error");
            return 43;
        }) | ex::let_error([=](std::exception_ptr& ep) {
            check_exception_ptr_message(ep, "error");
            return ex::transfer_just(sched, 42);
        }) | tt::sync_wait(sched)));
        HPX_TEST_EQ(result, 42);
    }

    // predecessor doesn't throw, let sender is ignored
    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto result = hpx::get<0>(*(ex::transfer_just(sched, 42) |
            ex::let_error([](std::exception_ptr) {
                HPX_TEST(false);
                return ex::just(43);
            }) |
            tt::sync_wait()));
        HPX_TEST_EQ(result, 42);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto result = hpx::get<0>(*(ex::transfer_just(sched, 42) |
            ex::let_error([=](std::exception_ptr) {
                HPX_TEST(false);
                return ex::transfer_just(sched, 43);
            }) |
            tt::sync_wait()));
        HPX_TEST_EQ(result, 42);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto result =
            hpx::get<0>(*(ex::just(42) | ex::let_error([=](std::exception_ptr) {
                HPX_TEST(false);
                return ex::transfer_just(sched, 43);
            }) | tt::sync_wait(sched)));
        HPX_TEST_EQ(result, 42);
    }
}

void test_detach()
{
    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        bool called = false;
        hpx::mutex mtx;
        hpx::condition_variable cond;
        ex::schedule(sched) | ex::then([&]() {
            std::unique_lock l{mtx};
            called = true;
            cond.notify_one();
        }) | ex::start_detached();

        {
            std::unique_lock l{mtx};
            HPX_TEST(cond.wait_for(
                l, std::chrono::seconds(1), [&]() { return called; }));
        }
        HPX_TEST(called);
    }

    // Values passed to set_value are ignored
    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        bool called = false;
        hpx::mutex mtx;
        hpx::condition_variable cond;
        ex::schedule(sched) | ex::then([&]() {
            std::lock_guard l{mtx};
            called = true;
            cond.notify_one();
            return 42;
        }) | ex::start_detached();

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
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        ex::keep_future(hpx::make_ready_future<void>()) |
            ex::then([](hpx::future<void>&& f) { HPX_TEST(f.is_ready()); }) |
            tt::sync_wait(sched);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        ex::keep_future(hpx::make_ready_future<void>().share()) |
            ex::then(
                [](hpx::shared_future<void>&& f) { HPX_TEST(f.is_ready()); }) |
            tt::sync_wait(sched);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        ex::keep_future(hpx::make_ready_future<int>(42)) |
            ex::then([](hpx::future<int>&& f) {
                HPX_TEST(f.is_ready());
                HPX_TEST_EQ(f.get(), 42);
            }) |
            tt::sync_wait(sched);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        ex::keep_future(hpx::make_ready_future<int>(42).share()) |
            ex::then([](hpx::shared_future<int>&& f) {
                HPX_TEST(f.is_ready());
                HPX_TEST_EQ(f.get(), 42);
            }) |
            tt::sync_wait(sched);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        std::atomic<bool> called{false};
        auto f = hpx::async([&]() { called = true; });

        auto r =
            hpx::get<0>(*tt::sync_wait(sched, ex::keep_future(std::move(f))));
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
            tt::sync_wait(sched, ex::keep_future(std::move(f)));
            HPX_TEST(false);
        }
        catch (...)
        {
            exception_thrown = true;
        }
        HPX_TEST(exception_thrown);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        std::atomic<bool> called{false};
        auto f = hpx::async([&]() {
            called = true;
            return 42;
        });

        auto r =
            hpx::get<0>(*tt::sync_wait(sched, ex::keep_future(std::move(f))));
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
            tt::sync_wait(sched, ex::keep_future(std::move(f)));
            HPX_TEST(false);
        }
        catch (...)
        {
            exception_thrown = true;
        }
        HPX_TEST(exception_thrown);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        std::atomic<bool> called{false};
        auto f = hpx::async([&]() {
            called = true;
            return 42;
        });

        HPX_TEST_EQ(hpx::get<0>(*tt::sync_wait(sched,
                        ex::then(ex::keep_future(std::move(f)),
                            [](hpx::future<int>&& f) { return f.get() / 2; }))),
            21);
        HPX_TEST(called);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        std::atomic<std::size_t> calls{0};
        auto sf = hpx::async([&]() { ++calls; }).share();

        tt::sync_wait(sched, ex::keep_future(sf));
        tt::sync_wait(sched, ex::keep_future(sf));
        tt::sync_wait(sched, ex::keep_future(std::move(sf)));
        HPX_TEST_EQ(calls, std::size_t(1));

        bool exception_thrown = false;
        try
        {
            tt::sync_wait(sched, ex::keep_future(sf));
            HPX_TEST(false);
        }
        catch (...)
        {
            exception_thrown = true;
        }
        HPX_TEST(exception_thrown);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        std::atomic<std::size_t> calls{0};
        auto sf = hpx::async([&]() {
            ++calls;
            return 42;
        }).share();

        HPX_TEST_EQ(
            hpx::get<0>(*tt::sync_wait(sched, ex::keep_future(sf))).get(), 42);
        HPX_TEST_EQ(
            hpx::get<0>(*tt::sync_wait(sched, ex::keep_future(sf))).get(), 42);
        HPX_TEST_EQ(
            hpx::get<0>(*tt::sync_wait(sched, ex::keep_future(std::move(sf))))
                .get(),
            42);
        HPX_TEST_EQ(calls, std::size_t(1));

        bool exception_thrown = false;
        try
        {
            tt::sync_wait(sched, ex::keep_future(sf));
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
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto f = hpx::async([&]() { return 42; });

        auto r = hpx::get<0>(*(ex::keep_future(std::move(f)) |
            ex::transfer(sched) | tt::sync_wait()));
        HPX_TEST(r.is_ready());
        HPX_TEST_EQ(r.get(), 42);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto sf = hpx::async([&]() { return 42; }).share();

        auto r = hpx::get<0>(*(ex::keep_future(std::move(sf)) |
            ex::transfer(sched) | tt::sync_wait()));
        HPX_TEST(r.is_ready());
        HPX_TEST_EQ(r.get(), 42);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto sf = hpx::async([&]() {
            return custom_type_non_default_constructible_non_copyable{42};
        }).share();

        // NOTE: Without keep_future this should fail to compile, since
        // sync_wait would receive a const& to the value which requires a copy
        // or storing a const&. The copy is not possible because the type is
        // noncopyable, and storing a reference is not acceptable since the
        // reference may outlive the value.
        auto r = hpx::get<0>(*(ex::keep_future(std::move(sf)) |
            ex::transfer(sched) | tt::sync_wait()));
        HPX_TEST(r.is_ready());
        HPX_TEST_EQ(r.get().x, 42);
    }

    // Use unwrapping with keep_future
    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto f = hpx::async([]() { return 42; });
        auto sf = hpx::async([]() { return 3.14; }).share();

        auto fun = hpx::unwrapping(
            [](int&& x, double const& y) { return x * 2 + (int(y) / 2); });
        HPX_TEST_EQ(hpx::get<0>(*(ex::when_all(ex::keep_future(std::move(f)),
                                      ex::keep_future(std::move(sf))) |
                        ex::then(fun) | tt::sync_wait(sched))),
            85);
    }

    {
        ex::system_context ctx;
        auto sched = ctx.get_scheduler();

        auto f = hpx::async([]() { return 42; });
        auto sf = hpx::async([]() { return 3.14; }).share();

        auto fun = hpx::unwrapping(
            [](int&& x, double const& y) { return x * 2 + (int(y) / 2); });
        HPX_TEST_EQ(hpx::get<0>(*(ex::when_all(ex::keep_future(std::move(f)),
                                      ex::keep_future(sf)) |
                        ex::transfer(sched) | ex::then(fun) | tt::sync_wait())),
            85);
    }
}

int hpx_main()
{
    test_concepts();
    test_compare();
    test_execute();
    test_sender_receiver_basic();
    test_sender_receiver_then();
    test_sender_receiver_then_wait();
    test_sender_receiver_then_sync_wait();
    test_sender_receiver_then_arguments();
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
    test_ensure_started();
    test_ensure_started_when_all();
    test_split();
    test_split_when_all();
    test_let_value();
    test_let_error();
    test_detach();
    test_keep_future_sender();

    test_completion_scheduler();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
