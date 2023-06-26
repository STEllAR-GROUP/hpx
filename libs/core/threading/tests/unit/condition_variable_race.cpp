//  Copyright (c) 2020-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Parts of this code were inspired by https://github.com/josuttis/jthread. The
//  original code was published by Nicolai Josuttis and Lewis Baker under the
//  Creative Commons Attribution 4.0 International License
//  (http://creativecommons.org/licenses/by/4.0/).

#include <hpx/init.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>

///////////////////////////////////////////////////////////////////////////////
//
// Test-Case by Howard Hinnant
// - emails 8.-9.11.18
//
// Original problem:
//  There's a bug in condition_variable_any2 that I don't think impacts the
//  implementation of jthread.
//  However this is such a complex subject that no stone should be left unturned.
//
//     ~condition_variable_any();
//
//     Requires: There shall be no thread blocked on *this.
//     [Note: That is, all threads shall have been notified; they may
//            subsequently block on the lock specified in the wait. This relaxes
//            the usual rules, which would have required all wait calls to
//            happen before destruction. Only the notification to unblock the
//            wait needs to happen before destruction. The user should take care
//            to ensure that no threads wait on *this once the destructor has
//            been started, especially when the waiting threads are calling the
//            wait functions in a loop or using the overloads of wait, wait_for,
//            or wait_until that take a predicate.
//      end note]
//
// That big long note means ~condition_variable_any() can execute before a
// signaled thread returns from a wait.
// If this happens with condition_variable_any, that waiting thread will attempt
// to lock the destructed mutex mut.
// To fix this, there must be shared ownership of the data member mut between
// the condition_variable_any  and the member functions wait (wait_for, etc.).
//
// libc++'s implementation gets this right:
//  https://github.com/llvm-mirror/libcxx/blob/master/include/condition_variable
//
// It holds the data member mutex with a shared_ptr<mutex> instead of mutex
// directly, and the wait functions create a local shared_ptr<mutex> copy on
// entry so that if *this destructs out from under the thread executing
// the wait function, the mutex stays alive until the wait function returns.
//
// Nico, after fixed by Anthony:
//  Thanks, but if I now use the cv_any implementation, fixed by Anthony,
//  I still get a core dump:
//    https://wandbox.org/permlink/VvG1UKubY69yAK7g
//  (#ifdef for both CV implementations)
//  So, either the test or the fix seems to be wrong.
//
// HH:
//  I'm guessing that to reliably test this, one is going to have to rebuild
//  your condition_variable_any with an internal mutex that checks for unlock-
//  after-destruction.
//  And the problem with that is now you no no longer have a std::mutex to put
//  into your internal std::condition_variable...
//
//  _Maybe_ you could test it by making your internal std::condition_variable a
//  std::condition_variable_any then you could put a debugging mutex into it.
//  But I'm not sure, because this is getting pretty weird and I have not
//  actually tried this.
///////////////////////////////////////////////////////////////////////////////

// Original test case from HH:
//
// hpx::condition_variable_any* cv = nullptr;
// hpx::mutex m;
// bool f_ready = false;
// bool g_ready = false;
//
// void f()
// {
//     m.lock();
//     f_ready = true;
//     cv->notify_one();
//     cv->~condition_variable_any();
//     std::memset(cv, 0x55, sizeof(*cv));    // UB but OK to ensure the check
//     m.unlock();
// }
//
// void g()
// {
//     m.lock();
//     g_ready = true;
//     cv->notify_one();
//     while (!f_ready)
//     {
//         cv->wait(m);
//     }
//     m.unlock();
// }
//
// void test_cv_any_mutex()
// {
//     // AW 9.11.18:
//     // Writing over the deleted memory is undefined behavior. In particular,
//     // it can destroy the heap data structure, and cause other problems.
//     // If you replace new/delete with malloc and free, then it's OK
//     void* raw = std::malloc(sizeof(hpx::condition_variable_any));
//     cv = new (raw) hpx::condition_variable_any;
//
//     hpx::thread th2(g);
//     m.lock();
//     while (!g_ready)
//         cv->wait(m);
//     m.unlock();
//
//     hpx::thread th1(f);
//     th1.join();
//
//     th2.join();
//     std::free(raw);
// }

void test_cv_mutex()
{
    void* raw = std::malloc(sizeof(hpx::condition_variable));
    hpx::condition_variable* cv = new (raw) hpx::condition_variable;

    hpx::mutex m;
    std::atomic<bool> f_ready{false};
    std::atomic<bool> g_ready{false};

    hpx::thread t2([&] {
        std::unique_lock<hpx::mutex> ul{m};
        g_ready = true;
        cv->notify_one();
        while (!f_ready)
        {
            cv->wait(ul);
        }
    });

    {
        std::unique_lock<hpx::mutex> ul{m};
        while (!g_ready)
        {
            cv->wait(ul);
        }
    }

    hpx::thread t1([&] {
        std::unique_lock<hpx::mutex> ul{m};
        f_ready = true;
        cv->notify_one();
        std::destroy_at(cv);
        // NOLINTNEXTLINE(bugprone-undefined-memory-manipulation)
        std::memset(
            (void*) cv, 0x55, sizeof(*cv));    // UB but OK to ensure the check
    });

    t1.join();
    t2.join();

    std::free(raw);
}

void test_cv_any_mutex()
{
    void* raw = std::malloc(sizeof(hpx::condition_variable_any));
    hpx::condition_variable_any* cv = new (raw) hpx::condition_variable_any;

    hpx::mutex m;
    std::atomic<bool> f_ready{false};
    std::atomic<bool> g_ready{false};

    hpx::thread t2([&] {
        std::unique_lock<hpx::mutex> ul{m};
        g_ready = true;
        cv->notify_one();
        while (!f_ready)
        {
            cv->wait(ul);
        }
    });

    {
        std::unique_lock<hpx::mutex> ul{m};
        while (!g_ready)
        {
            cv->wait(ul);
        }
    }

    hpx::thread t1([&] {
        std::unique_lock<hpx::mutex> ul{m};
        f_ready = true;
        cv->notify_one();
        std::destroy_at(cv);
        // NOLINTNEXTLINE(bugprone-undefined-memory-manipulation)
        std::memset(
            (void*) cv, 0x55, sizeof(*cv));    // UB but OK to ensure the check
    });

    t1.join();
    t2.join();

    std::free(raw);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    std::set_terminate([]() { HPX_TEST(false); });
    try
    {
        test_cv_mutex();
        test_cv_any_mutex();
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
