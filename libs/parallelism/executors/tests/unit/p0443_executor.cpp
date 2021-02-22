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

#include <atomic>
#include <cstddef>
#include <string>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
void test_execute()
{
    hpx::thread::id parent_id = hpx::this_thread::get_id();

    hpx::execution::experimental::executor exec{};
    hpx::execution::experimental::execute(exec,
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

    hpx::execution::experimental::executor exec{};

    auto begin = hpx::execution::experimental::schedule(exec);
    auto work = hpx::execution::experimental::connect(
        begin, check_context_receiver{parent_id, cond, executed});
    hpx::execution::experimental::start(work);

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

    hpx::execution::experimental::start(hpx::execution::experimental::connect(
        hpx::execution::experimental::executor{},
        check_context_receiver{parent_id, cond, executed}));

    {
        std::unique_lock<hpx::lcos::local::mutex> l{mtx};
        cond.wait(l, [&]() { return executed.load(); });
    }

    HPX_TEST(executed);
}

hpx::thread::id sender_receiver_transform_thread_id;

void test_sender_receiver_transform()
{
    hpx::execution::experimental::executor exec{};
    hpx::thread::id parent_id = hpx::this_thread::get_id();
    hpx::lcos::local::mutex mtx;
    hpx::lcos::local::condition_variable cond;
    std::atomic<bool> executed{false};

    auto begin = hpx::execution::experimental::schedule(exec);
    auto work1 = hpx::execution::experimental::transform(begin, [=]() {
        sender_receiver_transform_thread_id = hpx::this_thread::get_id();
        HPX_TEST_NEQ(sender_receiver_transform_thread_id, parent_id);
    });
    auto work2 = hpx::execution::experimental::transform(work1, []() {
        HPX_TEST_EQ(
            sender_receiver_transform_thread_id, hpx::this_thread::get_id());
    });
    auto end = hpx::execution::experimental::connect(
        work2, check_context_receiver{parent_id, cond, executed});
    hpx::execution::experimental::start(end);

    {
        std::unique_lock<hpx::lcos::local::mutex> l{mtx};
        cond.wait(l, [&]() { return executed.load(); });
    }

    HPX_TEST(executed);
}

void test_sender_receiver_transform_wait()
{
    hpx::execution::experimental::executor exec{};
    hpx::thread::id parent_id = hpx::this_thread::get_id();
    std::atomic<std::size_t> transform_count{0};
    std::atomic<bool> executed{false};

    auto begin = hpx::execution::experimental::schedule(exec);
    auto work1 = hpx::execution::experimental::transform(
        begin, [&transform_count, parent_id]() {
            sender_receiver_transform_thread_id = hpx::this_thread::get_id();
            HPX_TEST_NEQ(sender_receiver_transform_thread_id, parent_id);
            ++transform_count;
        });
    auto work2 = hpx::execution::experimental::transform(
        work1, [&transform_count, &executed]() {
            HPX_TEST_EQ(sender_receiver_transform_thread_id,
                hpx::this_thread::get_id());
            ++transform_count;
            executed = true;
        });
    hpx::execution::experimental::sync_wait(work2);
    HPX_TEST_EQ(transform_count, std::size_t(2));
    HPX_TEST(executed);
}

void test_sender_receiver_transform_sync_wait()
{
    hpx::execution::experimental::executor exec{};
    hpx::thread::id parent_id = hpx::this_thread::get_id();
    std::atomic<std::size_t> transform_count{0};

    auto begin = hpx::execution::experimental::schedule(exec);
    auto work = hpx::execution::experimental::transform(
        begin, [&transform_count, parent_id]() {
            sender_receiver_transform_thread_id = hpx::this_thread::get_id();
            HPX_TEST_NEQ(sender_receiver_transform_thread_id, parent_id);
            ++transform_count;
            return 42;
        });
    auto result = hpx::execution::experimental::sync_wait(work);
    HPX_TEST_EQ(transform_count, std::size_t(1));
    static_assert(
        std::is_same<int, typename std::decay<decltype(result)>::type>::value,
        "result should be an int");
    HPX_TEST_EQ(result, 42);
}

void test_sender_receiver_transform_arguments()
{
    hpx::execution::experimental::executor exec{};
    hpx::thread::id parent_id = hpx::this_thread::get_id();
    std::atomic<std::size_t> transform_count{0};

    auto begin = hpx::execution::experimental::schedule(exec);
    auto work1 = hpx::execution::experimental::transform(
        begin, [&transform_count, parent_id]() {
            sender_receiver_transform_thread_id = hpx::this_thread::get_id();
            HPX_TEST_NEQ(sender_receiver_transform_thread_id, parent_id);
            ++transform_count;
            return 3;
        });
    auto work2 = hpx::execution::experimental::transform(
        work1, [&transform_count](int x) -> std::string {
            HPX_TEST_EQ(sender_receiver_transform_thread_id,
                hpx::this_thread::get_id());
            ++transform_count;
            return std::string("hello") + std::to_string(x);
        });
    auto work3 = hpx::execution::experimental::transform(
        work2, [&transform_count](std::string s) {
            HPX_TEST_EQ(sender_receiver_transform_thread_id,
                hpx::this_thread::get_id());
            ++transform_count;
            return 2 * s.size();
        });
    auto result = hpx::execution::experimental::sync_wait(work3);
    HPX_TEST_EQ(transform_count, std::size_t(3));
    static_assert(std::is_same<std::size_t,
                      typename std::decay<decltype(result)>::type>::value,
        "result should be a std::size_t");
    HPX_TEST_EQ(result, std::size_t(12));
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

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(
        hpx::init(argc, argv), 0, "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
