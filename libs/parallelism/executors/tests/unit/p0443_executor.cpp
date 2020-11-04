//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution.hpp>
#include <hpx/functional.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <atomic>
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

void test_sender_receiver_basic()
{
    hpx::execution::experimental::executor exec{};

    auto begin = hpx::execution::experimental::schedule(exec);
    auto work = hpx::execution::experimental::connect(
        begin, hpx::execution::experimental::sink_receiver{});
    hpx::execution::experimental::start(work);
}

void test_sender_receiver_basic2()
{
    hpx::execution::experimental::start(hpx::execution::experimental::connect(
        hpx::execution::experimental::executor{},
        hpx::execution::experimental::sink_receiver{}));
}

hpx::thread::id sender_receiver_then_thread_id;

void test_sender_receiver_then()
{
    hpx::execution::experimental::executor exec{};
    hpx::thread::id parent_id = hpx::this_thread::get_id();

    auto begin = hpx::execution::experimental::schedule(exec);
    auto work1 = hpx::execution::experimental::then(begin, [=]() {
        sender_receiver_then_thread_id = hpx::this_thread::get_id();
        HPX_TEST_NEQ(sender_receiver_then_thread_id, parent_id);
    });
    auto work2 = hpx::execution::experimental::then(work1, []() {
        HPX_TEST_EQ(sender_receiver_then_thread_id, hpx::this_thread::get_id());
    });
    auto end = hpx::execution::experimental::connect(
        work2, hpx::execution::experimental::sink_receiver{});
    hpx::execution::experimental::start(end);
}

void test_sender_receiver_then_wait()
{
    hpx::execution::experimental::executor exec{};
    hpx::thread::id parent_id = hpx::this_thread::get_id();
    std::atomic<std::size_t> then_count{0};

    auto begin = hpx::execution::experimental::schedule(exec);
    auto work1 =
        hpx::execution::experimental::then(begin, [&then_count, parent_id]() {
            sender_receiver_then_thread_id = hpx::this_thread::get_id();
            HPX_TEST_NEQ(sender_receiver_then_thread_id, parent_id);
            ++then_count;
        });
    auto work2 =
        hpx::execution::experimental::then(work1, [&then_count, parent_id]() {
            HPX_TEST_EQ(
                sender_receiver_then_thread_id, hpx::this_thread::get_id());
            ++then_count;
        });
    hpx::execution::experimental::wait(work2);
    HPX_TEST_EQ(then_count, std::size_t(2));

    static_assert(hpx::execution::experimental::traits::is_receiver_v<
                      hpx::execution::experimental::detail::wait_receiver>,
        "asd");
}

void test_sender_receiver_then_get()
{
    hpx::execution::experimental::executor exec{};
    hpx::thread::id parent_id = hpx::this_thread::get_id();
    std::atomic<std::size_t> then_count{0};

    auto begin = hpx::execution::experimental::schedule(exec);
    auto work =
        hpx::execution::experimental::then(begin, [&then_count, parent_id]() {
            sender_receiver_then_thread_id = hpx::this_thread::get_id();
            HPX_TEST_NEQ(sender_receiver_then_thread_id, parent_id);
            ++then_count;
            return 42;
        });
    auto result = hpx::execution::experimental::get(work);
    HPX_TEST_EQ(then_count, std::size_t(1));
    static_assert(
        std::is_same<int, typename std::decay<decltype(result)>::type>::value,
        "result should be an int");
    HPX_TEST_EQ(result, 42);
}

void test_sender_receiver_then_arguments()
{
    hpx::execution::experimental::executor exec{};
    hpx::thread::id parent_id = hpx::this_thread::get_id();
    std::atomic<std::size_t> then_count{0};

    auto begin = hpx::execution::experimental::schedule(exec);
    auto work1 =
        hpx::execution::experimental::then(begin, [&then_count, parent_id]() {
            sender_receiver_then_thread_id = hpx::this_thread::get_id();
            HPX_TEST_NEQ(sender_receiver_then_thread_id, parent_id);
            ++then_count;
            return 3;
        });
    auto work2 = hpx::execution::experimental::then(
        work1, [&then_count, parent_id](int x) -> std::string {
            HPX_TEST_EQ(
                sender_receiver_then_thread_id, hpx::this_thread::get_id());
            ++then_count;
            return std::string("hello") + std::to_string(x);
        });
    auto work3 = hpx::execution::experimental::then(
        work2, [&then_count, parent_id](std::string s) {
            HPX_TEST_EQ(
                sender_receiver_then_thread_id, hpx::this_thread::get_id());
            ++then_count;
            return 2 * s.size();
        });
    auto result = hpx::execution::experimental::get(work3);
    HPX_TEST_EQ(then_count, std::size_t(3));
    static_assert(std::is_same<std::size_t,
                      typename std::decay<decltype(result)>::type>::value,
        "result should be a std::size_t");
    HPX_TEST_EQ(result, std::size_t(12));
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[])
{
    test_execute();
    test_sender_receiver_basic();
    test_sender_receiver_basic2();
    test_sender_receiver_then();
    test_sender_receiver_then_wait();
    test_sender_receiver_then_get();
    test_sender_receiver_then_arguments();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(
        hpx::init(argc, argv), 0, "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
