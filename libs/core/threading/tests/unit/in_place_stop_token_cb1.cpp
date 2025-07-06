//  Copyright (c) 2020-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Parts of this code were inspired by https://github.com/josuttis/jthread. The
//  original code was published by Nicolai Josuttis and Lewis Baker under the
//  Creative Commons Attribution 4.0 International License
//  (http://creativecommons.org/licenses/by/4.0/).

#include <hpx/chrono.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/optional.hpp>
#include <hpx/thread.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <mutex>
#include <utility>

//////////////////////////////////////////////////////////////////////////////
void test_default_token_is_not_stoppable()
{
    hpx::experimental::in_place_stop_token t;
    HPX_TEST(!t.stop_requested());
    HPX_TEST(!t.stop_possible());
}

//////////////////////////////////////////////////////////////////////////////
void test_requesting_stop_on_source_updates_token()
{
    hpx::experimental::in_place_stop_source s;
    hpx::experimental::in_place_stop_token t = s.get_token();
    HPX_TEST(t.stop_possible());
    HPX_TEST(!t.stop_requested());
    s.request_stop();
    HPX_TEST(t.stop_requested());
    HPX_TEST(t.stop_possible());
}

//////////////////////////////////////////////////////////////////////////////
void test_callback_not_executed_immediately_if_stop_not_yet_requested()
{
    hpx::experimental::in_place_stop_source s;

    bool callback_executed = false;
    {
        auto f = [&] { callback_executed = true; };
        hpx::experimental::in_place_stop_callback<decltype(f)> cb(
            s.get_token(), std::move(f));
    }

    HPX_TEST(!callback_executed);

    s.request_stop();

    HPX_TEST(!callback_executed);
}

//////////////////////////////////////////////////////////////////////////////
void test_callback_executed_if_stop_requested_before_destruction()
{
    hpx::experimental::in_place_stop_source s;

    bool callback_executed = false;
    auto f = [&] { callback_executed = true; };
    hpx::experimental::in_place_stop_callback<decltype(f)> cb(
        s.get_token(), std::move(f));

    HPX_TEST(!callback_executed);

    s.request_stop();

    HPX_TEST(callback_executed);
}

//////////////////////////////////////////////////////////////////////////////
void test_callback_executed_immediately_if_stop_already_requested()
{
    hpx::experimental::in_place_stop_source s;
    s.request_stop();

    bool executed = false;
    auto f = [&] { executed = true; };
    hpx::experimental::in_place_stop_callback<decltype(f)> cb(
        s.get_token(), std::move(f));
    HPX_TEST(executed);
}

//////////////////////////////////////////////////////////////////////////////
void test_register_multiple_callbacks()
{
    hpx::experimental::in_place_stop_source s;
    auto t = s.get_token();

    int callback_execution_count = 0;
    auto callback = [&] { ++callback_execution_count; };

    hpx::experimental::in_place_stop_callback<decltype(callback)> r1(
        t, callback);
    hpx::experimental::in_place_stop_callback<decltype(callback)> r2(
        t, callback);
    hpx::experimental::in_place_stop_callback<decltype(callback)> r3(
        t, callback);
    hpx::experimental::in_place_stop_callback<decltype(callback)> r4(
        t, callback);
    hpx::experimental::in_place_stop_callback<decltype(callback)> r5(
        t, callback);
    hpx::experimental::in_place_stop_callback<decltype(callback)> r6(
        t, callback);
    hpx::experimental::in_place_stop_callback<decltype(callback)> r7(
        t, callback);
    hpx::experimental::in_place_stop_callback<decltype(callback)> r8(
        t, callback);
    hpx::experimental::in_place_stop_callback<decltype(callback)> r9(
        t, callback);
    hpx::experimental::in_place_stop_callback<decltype(callback)> r10(
        t, callback);

    s.request_stop();

    HPX_TEST(callback_execution_count == 10);
}

//////////////////////////////////////////////////////////////////////////////
void test_concurrent_callback_registration()
{
    auto f2 = [] {};

    auto thread_loop = [&](hpx::experimental::in_place_stop_token token) {
        std::atomic<bool> cancelled{false};
        auto f1 = [&] { cancelled = true; };
        while (!cancelled)
        {
            hpx::experimental::in_place_stop_callback<std::function<void()>>
                registration(token, f1);

            hpx::experimental::in_place_stop_callback<decltype(f2)> cb0(
                token, f2);
            hpx::experimental::in_place_stop_callback<decltype(f2)> cb1(
                token, f2);
            hpx::experimental::in_place_stop_callback<decltype(f2)> cb2(
                token, f2);
            hpx::experimental::in_place_stop_callback<decltype(f2)> cb3(
                token, f2);
            hpx::experimental::in_place_stop_callback<decltype(f2)> cb4(
                token, f2);
            hpx::experimental::in_place_stop_callback<decltype(f2)> cb5(
                token, f2);
            hpx::experimental::in_place_stop_callback<decltype(f2)> cb6(
                token, f2);
            hpx::experimental::in_place_stop_callback<decltype(f2)> cb7(
                token, f2);
            hpx::experimental::in_place_stop_callback<decltype(f2)> cb8(
                token, f2);
            hpx::experimental::in_place_stop_callback<decltype(f2)> cb9(
                token, f2);
            hpx::experimental::in_place_stop_callback<decltype(f2)> cb10(
                token, f2);
            hpx::experimental::in_place_stop_callback<decltype(f2)> cb11(
                token, f2);
            hpx::experimental::in_place_stop_callback<decltype(f2)> cb12(
                token, f2);
            hpx::experimental::in_place_stop_callback<decltype(f2)> cb13(
                token, f2);
            hpx::experimental::in_place_stop_callback<decltype(f2)> cb14(
                token, f2);
            hpx::experimental::in_place_stop_callback<decltype(f2)> cb15(
                token, f2);
            hpx::experimental::in_place_stop_callback<decltype(f2)> cb16(
                token, f2);

            hpx::this_thread::yield();
        }
    };

    // Just assert this runs and terminates without crashing.
    for (int i = 0; i < 100; ++i)
    {
        hpx::experimental::in_place_stop_source source;

        hpx::thread waiter1{thread_loop, source.get_token()};
        hpx::thread waiter2{thread_loop, source.get_token()};
        hpx::thread waiter3{thread_loop, source.get_token()};

        hpx::this_thread::sleep_for(std::chrono::milliseconds(10));

        hpx::thread canceller{[&source] { source.request_stop(); }};

        canceller.join();
        waiter1.join();
        waiter2.join();
        waiter3.join();
    }
}

//////////////////////////////////////////////////////////////////////////////
void test_callback_deregistered_from_within_callback_does_not_deadlock()
{
    hpx::experimental::in_place_stop_source src;
    hpx::optional<
        hpx::experimental::in_place_stop_callback<std::function<void()>>>
        cb;

    cb.emplace(src.get_token(), [&] { cb.reset(); });

    src.request_stop();

    HPX_TEST(!cb.has_value());
}

//////////////////////////////////////////////////////////////////////////////
void test_callback_deregistration_doesnt_wait_for_others_to_finish_executing()
{
    hpx::experimental::in_place_stop_source src;

    hpx::mutex mut;
    hpx::condition_variable cv;

    bool release_callback = false;
    bool callback_executing = false;

    auto dummy_callback = [] {};

    hpx::optional<
        hpx::experimental::in_place_stop_callback<decltype(dummy_callback)&>>
        cb1(std::in_place, src.get_token(), dummy_callback);

    // Register a first callback that will signal when it starts executing
    // and then block until it receives a signal.
    auto f = [&] {
        std::unique_lock<hpx::mutex> lock{mut};
        callback_executing = true;
        cv.notify_all();
        cv.wait(lock, [&] { return release_callback; });
    };
    hpx::experimental::in_place_stop_callback<decltype(f)> blocking_callback(
        src.get_token(), std::move(f));

    hpx::optional<
        hpx::experimental::in_place_stop_callback<decltype(dummy_callback)&>>
        cb2{std::in_place, src.get_token(), dummy_callback};

    hpx::thread signalling_thread{[&] { src.request_stop(); }};

    // Wait until the callback starts executing on the signaling-thread.
    // The signaling thread will remain blocked in this callback until we
    // release it.
    {
        std::unique_lock<hpx::mutex> lock{mut};
        cv.wait(lock, [&] { return callback_executing; });
    }

    // Then try and unregister the other callbacks.
    // This operation should not be blocked on other callbacks.
    cb1.reset();
    cb2.reset();

    // Finally, signal the callback to unblock and wait for the signaling
    // thread to finish.
    {
        std::unique_lock<hpx::mutex> lock{mut};
        release_callback = true;
        cv.notify_all();
    }

    signalling_thread.join();
}

//////////////////////////////////////////////////////////////////////////////
void test_callback_deregistration_blocks_until_callback_finishes()
{
    hpx::experimental::in_place_stop_source src;

    hpx::mutex mut;
    hpx::condition_variable cv;

    bool callback_registered = false;

    hpx::thread callback_registering_thread{[&] {
        bool callback_executing = false;
        bool callback_about_to_return = false;
        bool callback_unregistered = false;

        {
            auto f = [&] {
                std::unique_lock<hpx::mutex> lock{mut};
                callback_executing = true;
                cv.notify_all();
                lock.unlock();
                hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
                HPX_TEST(!callback_unregistered);
                callback_about_to_return = true;
            };

            hpx::experimental::in_place_stop_callback<std::function<void()>> cb(
                src.get_token(), f);

            {
                std::unique_lock<hpx::mutex> lock{mut};
                callback_registered = true;
                cv.notify_all();
                cv.wait(lock, [&] { return callback_executing; });
            }

            HPX_TEST(!callback_about_to_return);
        }

        callback_unregistered = true;

        HPX_TEST(callback_executing);
        HPX_TEST(callback_about_to_return);
    }};

    // Make sure to release the lock before requesting stop
    // since this will execute the callback which will try to
    // acquire the lock on the mutex.
    {
        std::unique_lock<hpx::mutex> lock{mut};
        cv.wait(lock, [&] { return callback_registered; });
    }

    src.request_stop();

    callback_registering_thread.join();
}

//////////////////////////////////////////////////////////////////////////////
template <typename CB>
struct callback_batch
{
    using callback_type = hpx::experimental::in_place_stop_callback<CB&>;

    callback_batch(hpx::experimental::in_place_stop_token t, CB& callback)
      : r0(t, callback)
      , r1(t, callback)
      , r2(t, callback)
      , r3(t, callback)
      , r4(t, callback)
      , r5(t, callback)
      , r6(t, callback)
      , r7(t, callback)
      , r8(t, callback)
      , r9(t, callback)
    {
    }

    callback_type r0;
    callback_type r1;
    callback_type r2;
    callback_type r3;
    callback_type r4;
    callback_type r5;
    callback_type r6;
    callback_type r7;
    callback_type r8;
    callback_type r9;
};

void test_cancellation_single_thread_performance()
{
    auto callback = [] {};

    hpx::experimental::in_place_stop_source s;

    constexpr int iteration_count = 100'000;

    auto start = hpx::chrono::high_resolution_clock::now();

    for (int i = 0; i < iteration_count; ++i)
    {
        hpx::experimental::in_place_stop_callback<decltype(callback)> r(
            s.get_token(), callback);
    }

    auto end = hpx::chrono::high_resolution_clock::now();

    auto time1 = end - start;

    start = end;

    for (int i = 0; i < iteration_count; ++i)
    {
        callback_batch<decltype(callback)> b{s.get_token(), callback};
    }

    end = hpx::chrono::high_resolution_clock::now();

    auto time2 = end - start;

    start = end;

    for (int i = 0; i < iteration_count; ++i)
    {
        callback_batch<decltype(callback)> b0{s.get_token(), callback};
        callback_batch<decltype(callback)> b1{s.get_token(), callback};
        callback_batch<decltype(callback)> b2{s.get_token(), callback};
        callback_batch<decltype(callback)> b3{s.get_token(), callback};
        callback_batch<decltype(callback)> b4{s.get_token(), callback};
    }

    end = hpx::chrono::high_resolution_clock::now();

    auto time3 = end - start;

    auto report = [](const char* label, auto time, std::uint64_t count) {
        auto ms = std::chrono::duration<double, std::milli>(time).count();
        auto ns = std::chrono::duration<double, std::nano>(time).count();
        std::cout << label << " took " << ms << "ms ("
                  << (ns / static_cast<double>(count)) << " ns/item)"
                  << std::endl;
    };

    report("Individual", time1, iteration_count);
    report("Batch10", time2,
        static_cast<std::uint64_t>(10) *
            static_cast<std::uint64_t>(iteration_count));
    report("Batch50", time3,
        static_cast<std::uint64_t>(50) *
            static_cast<std::uint64_t>(iteration_count));
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_default_token_is_not_stoppable();
    test_requesting_stop_on_source_updates_token();
    test_callback_not_executed_immediately_if_stop_not_yet_requested();
    test_callback_executed_if_stop_requested_before_destruction();
    test_callback_executed_immediately_if_stop_already_requested();
    test_register_multiple_callbacks();
    test_concurrent_callback_registration();
    test_callback_deregistered_from_within_callback_does_not_deadlock();
    test_callback_deregistration_doesnt_wait_for_others_to_finish_executing();
    test_callback_deregistration_blocks_until_callback_finishes();
    test_cancellation_single_thread_performance();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
