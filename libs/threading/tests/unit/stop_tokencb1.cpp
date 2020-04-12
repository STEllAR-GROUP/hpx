//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Parts of this code were inspired by https://github.com/josuttis/jthread. The
//  original code was published by Nicolai Josuttis and Lewis Baker under the
//  Creative Commons Attribution 4.0 International License
//  (http://creativecommons.org/licenses/by/4.0/).

#include <hpx/datastructures.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/synchronization.hpp>
#include <hpx/testing.hpp>
#include <hpx/threading.hpp>
#include <hpx/timing.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <mutex>

//////////////////////////////////////////////////////////////////////////////
void test_default_token_is_not_stoppable()
{
    hpx::stop_token t;
    HPX_TEST(!t.stop_requested());
    HPX_TEST(!t.stop_possible());
}

//////////////////////////////////////////////////////////////////////////////
void test_requesting_stop_on_source_updates_token()
{
    hpx::stop_source s;
    hpx::stop_token t = s.get_token();
    HPX_TEST(t.stop_possible());
    HPX_TEST(!t.stop_requested());
    s.request_stop();
    HPX_TEST(t.stop_requested());
    HPX_TEST(t.stop_possible());
}

//////////////////////////////////////////////////////////////////////////////
void test_token_cant_be_stopped_when_no_more_sources()
{
    hpx::stop_token t;
    {
        hpx::stop_source s;
        t = s.get_token();
        HPX_TEST(t.stop_possible());
    }

    HPX_TEST(!t.stop_possible());
}

//////////////////////////////////////////////////////////////////////////////
void test_token_can_be_stopped_when_no_more_sources_if_stop_already_requested()
{
    hpx::stop_token t;
    {
        hpx::stop_source s;
        t = s.get_token();
        HPX_TEST(t.stop_possible());
        s.request_stop();
    }

    HPX_TEST(t.stop_possible());
    HPX_TEST(t.stop_requested());
}

//////////////////////////////////////////////////////////////////////////////
void test_callback_not_executed_immediately_if_stop_not_yet_requested()
{
    hpx::stop_source s;

    bool callback_executed = false;
    {
        auto cb = hpx::make_stop_callback(
            s.get_token(), [&] { callback_executed = true; });
    }

    HPX_TEST(!callback_executed);

    s.request_stop();

    HPX_TEST(!callback_executed);
}

//////////////////////////////////////////////////////////////////////////////
void test_callback_executed_if_stop_requested_before_destruction()
{
    hpx::stop_source s;

    bool callback_executed = false;
    auto cb = hpx::make_stop_callback(
        s.get_token(), [&] { callback_executed = true; });

    HPX_TEST(!callback_executed);

    s.request_stop();

    HPX_TEST(callback_executed);
}

//////////////////////////////////////////////////////////////////////////////
void test_callback_executed_immediately_if_stop_already_requested()
{
    hpx::stop_source s;
    s.request_stop();

    bool executed = false;
    auto cb = hpx::make_stop_callback(s.get_token(), [&] { executed = true; });
    HPX_TEST(executed);
}

//////////////////////////////////////////////////////////////////////////////
void test_register_multiple_callbacks()
{
    hpx::stop_source s;
    auto t = s.get_token();

    int callback_execution_count = 0;
    auto callback = [&] { ++callback_execution_count; };

    auto r1 = hpx::make_stop_callback(t, callback);
    auto r2 = hpx::make_stop_callback(t, callback);
    auto r3 = hpx::make_stop_callback(t, callback);
    auto r4 = hpx::make_stop_callback(t, callback);
    auto r5 = hpx::make_stop_callback(t, callback);
    auto r6 = hpx::make_stop_callback(t, callback);
    auto r7 = hpx::make_stop_callback(t, callback);
    auto r8 = hpx::make_stop_callback(t, callback);
    auto r9 = hpx::make_stop_callback(t, callback);
    auto r10 = hpx::make_stop_callback(t, callback);

    s.request_stop();

    HPX_TEST(callback_execution_count == 10);
}

//////////////////////////////////////////////////////////////////////////////
void test_concurrent_callback_registration()
{
    auto thread_loop = [](hpx::stop_token token) {
        std::atomic<bool> cancelled{false};
        while (!cancelled)
        {
            auto registration =
                hpx::make_stop_callback(token, [&] { cancelled = true; });

            auto cb0 = hpx::make_stop_callback(token, [] {});
            auto cb1 = hpx::make_stop_callback(token, [] {});
            auto cb2 = hpx::make_stop_callback(token, [] {});
            auto cb3 = hpx::make_stop_callback(token, [] {});
            auto cb4 = hpx::make_stop_callback(token, [] {});
            auto cb5 = hpx::make_stop_callback(token, [] {});
            auto cb6 = hpx::make_stop_callback(token, [] {});
            auto cb7 = hpx::make_stop_callback(token, [] {});
            auto cb8 = hpx::make_stop_callback(token, [] {});
            auto cb9 = hpx::make_stop_callback(token, [] {});
            auto cb10 = hpx::make_stop_callback(token, [] {});
            auto cb11 = hpx::make_stop_callback(token, [] {});
            auto cb12 = hpx::make_stop_callback(token, [] {});
            auto cb13 = hpx::make_stop_callback(token, [] {});
            auto cb14 = hpx::make_stop_callback(token, [] {});
            auto cb15 = hpx::make_stop_callback(token, [] {});
            auto cb16 = hpx::make_stop_callback(token, [] {});

            hpx::this_thread::yield();
        }
    };

    // Just assert this runs and terminates without crashing.
    for (int i = 0; i < 100; ++i)
    {
        hpx::stop_source source;

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
    hpx::stop_source src;
    hpx::util::optional<hpx::stop_callback<std::function<void()>>> cb;

    cb.emplace(src.get_token(), [&] { cb.reset(); });

    src.request_stop();

    HPX_TEST(!cb.has_value());
}

//////////////////////////////////////////////////////////////////////////////
void test_callback_deregistration_doesnt_wait_for_others_to_finish_executing()
{
    hpx::stop_source src;

    hpx::lcos::local::mutex mut;
    hpx::lcos::local::condition_variable cv;

    bool release_callback = false;
    bool callback_executing = false;

    auto dummy_callback = [] {};

    hpx::util::optional<hpx::stop_callback<decltype(dummy_callback)&>> cb1(
        hpx::util::in_place, src.get_token(), dummy_callback);

    // Register a first callback that will signal when it starts executing
    // and then block until it receives a signal.
    auto blocking_callback = hpx::make_stop_callback(src.get_token(), [&] {
        std::unique_lock<hpx::lcos::local::mutex> lock{mut};
        callback_executing = true;
        cv.notify_all();
        cv.wait(lock, [&] { return release_callback; });
    });

    hpx::util::optional<hpx::stop_callback<decltype(dummy_callback)&>> cb2{
        hpx::util::in_place, src.get_token(), dummy_callback};

    hpx::thread signalling_thread{[&] { src.request_stop(); }};

    // Wait until the callback starts executing on the signaling-thread.
    // The signaling thread will remain blocked in this callback until we
    // release it.
    {
        std::unique_lock<hpx::lcos::local::mutex> lock{mut};
        cv.wait(lock, [&] { return callback_executing; });
    }

    // Then try and unregister the other callbacks.
    // This operation should not be blocked on other callbacks.
    cb1.reset();
    cb2.reset();

    // Finally, signal the callback to unblock and wait for the signaling
    // thread to finish.
    {
        std::unique_lock<hpx::lcos::local::mutex> lock{mut};
        release_callback = true;
        cv.notify_all();
    }

    signalling_thread.join();
}

//////////////////////////////////////////////////////////////////////////////
void test_callback_deregistration_blocks_until_callback_finishes()
{
    hpx::stop_source src;

    hpx::lcos::local::mutex mut;
    hpx::lcos::local::condition_variable cv;

    bool callback_registered = false;

    hpx::thread callback_registering_thread{[&] {
        bool callback_executing = false;
        bool callback_about_to_return = false;
        bool callback_unregistered = false;

        {
            auto f = [&] {
                std::unique_lock<hpx::lcos::local::mutex> lock{mut};
                callback_executing = true;
                cv.notify_all();
                lock.unlock();
                hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
                HPX_TEST(!callback_unregistered);
                callback_about_to_return = true;
            };

            auto cb = hpx::make_stop_callback(src.get_token(), f);

            {
                std::unique_lock<hpx::lcos::local::mutex> lock{mut};
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
        std::unique_lock<hpx::lcos::local::mutex> lock{mut};
        cv.wait(lock, [&] { return callback_registered; });
    }

    src.request_stop();

    callback_registering_thread.join();
}

//////////////////////////////////////////////////////////////////////////////
template <typename CB>
struct callback_batch
{
    using callback_type = hpx::stop_callback<CB&>;

    callback_batch(hpx::stop_token t, CB& callback)
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

    hpx::stop_source s;

    constexpr int iteration_count = 100'000;

    auto start = hpx::util::high_resolution_clock::now();

    for (int i = 0; i < iteration_count; ++i)
    {
        auto r = hpx::make_stop_callback(s.get_token(), callback);
    }

    auto end = hpx::util::high_resolution_clock::now();

    auto time1 = end - start;

    start = end;

    for (int i = 0; i < iteration_count; ++i)
    {
        callback_batch b{s.get_token(), callback};
    }

    end = hpx::util::high_resolution_clock::now();

    auto time2 = end - start;

    start = end;

    for (int i = 0; i < iteration_count; ++i)
    {
        callback_batch b0{s.get_token(), callback};
        callback_batch b1{s.get_token(), callback};
        callback_batch b2{s.get_token(), callback};
        callback_batch b3{s.get_token(), callback};
        callback_batch b4{s.get_token(), callback};
    }

    end = hpx::util::high_resolution_clock::now();

    auto time3 = end - start;

    auto report = [](const char* label, auto time, std::uint64_t count) {
        auto ms = std::chrono::duration<double, std::milli>(time).count();
        auto ns = std::chrono::duration<double, std::nano>(time).count();
        std::cout << label << " took " << ms << "ms ("
                  << (ns / static_cast<double>(count)) << " ns/item)"
                  << std::endl;
    };

    report("Individual", time1, iteration_count);
    report("Batch10", time2, 10 * iteration_count);
    report("Batch50", time3, 50 * iteration_count);
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    test_default_token_is_not_stoppable();
    test_requesting_stop_on_source_updates_token();
    test_token_cant_be_stopped_when_no_more_sources();
    test_token_can_be_stopped_when_no_more_sources_if_stop_already_requested();
    test_callback_not_executed_immediately_if_stop_not_yet_requested();
    test_callback_executed_if_stop_requested_before_destruction();
    test_callback_executed_immediately_if_stop_already_requested();
    test_register_multiple_callbacks();
    test_concurrent_callback_registration();
    test_callback_deregistered_from_within_callback_does_not_deadlock();
    test_callback_deregistration_doesnt_wait_for_others_to_finish_executing();
    test_callback_deregistration_blocks_until_callback_finishes();
    test_cancellation_single_thread_performance();

    return hpx::util::report_errors();
}
