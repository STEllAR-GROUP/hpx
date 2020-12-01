//  Copyright (c)      2019 Mikael Simberg
//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The thread_pool_os_executor is an executor that creates a new thread pool for
// itself. This checks that the usual functions of an executor work with this
// executor when used *without the HPX runtime*. This test fails if thread
// pools, schedulers etc. assume that the global runtime (configuration, thread
// manager, etc.) always exists.

#include <hpx/execution_base/this_thread.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
hpx::thread::id test(int passed_through)
{
    HPX_TEST_EQ(passed_through, 42);
    return hpx::this_thread::get_id();
}

template <typename Executor>
void test_sync(Executor& exec)
{
    HPX_TEST(hpx::parallel::execution::sync_execute(exec, &test, 42) !=
        hpx::this_thread::get_id());
}

template <typename Executor>
void test_async(Executor& exec)
{
    HPX_TEST(hpx::parallel::execution::async_execute(exec, &test, 42).get() !=
        hpx::this_thread::get_id());
}

///////////////////////////////////////////////////////////////////////////////
hpx::thread::id test_f(hpx::future<void> f, int passed_through)
{
    HPX_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    HPX_TEST_EQ(passed_through, 42);
    return hpx::this_thread::get_id();
}

template <typename Executor>
void test_then(Executor& exec)
{
    hpx::future<void> f = hpx::make_ready_future();

    HPX_TEST(
        hpx::parallel::execution::then_execute(exec, &test_f, f, 42).get() !=
        hpx::this_thread::get_id());
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test(int, hpx::thread::id tid, int passed_through)    //-V813
{
    HPX_TEST_NEQ(tid, hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);
}

template <typename Executor>
void test_bulk_sync(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    hpx::parallel::execution::bulk_sync_execute(
        exec, hpx::util::bind(&bulk_test, _1, tid, _2), v, 42);
    hpx::parallel::execution::bulk_sync_execute(exec, &bulk_test, v, tid, 42);
}

template <typename Executor>
void test_bulk_async(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    hpx::when_all(hpx::parallel::execution::bulk_async_execute(
                      exec, hpx::util::bind(&bulk_test, _1, tid, _2), v, 42))
        .get();
    hpx::when_all(hpx::parallel::execution::bulk_async_execute(
                      exec, &bulk_test, v, tid, 42))
        .get();
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test_f(int, hpx::shared_future<void> f, hpx::thread::id tid,
    int passed_through)    //-V813
{
    HPX_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    HPX_TEST_NEQ(tid, hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);
}

template <typename Executor>
void test_bulk_then(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;
    using hpx::util::placeholders::_3;

    hpx::shared_future<void> f = hpx::make_ready_future();

    hpx::parallel::execution::bulk_then_execute(
        exec, hpx::util::bind(&bulk_test_f, _1, _2, tid, _3), v, f, 42)
        .get();
    hpx::parallel::execution::bulk_then_execute(
        exec, &bulk_test_f, v, f, tid, 42)
        .get();
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_thread_pool_os_executor(Executor exec)
{
    test_sync(exec);
    test_async(exec);
    test_then(exec);
    test_bulk_sync(exec);
    test_bulk_async(exec);
    test_bulk_then(exec);
}

template <typename Executor>
void spawn_test()
{
    using namespace hpx::parallel;

    std::size_t const num_threads = 4;
    std::size_t const max_cores = num_threads;
    hpx::threads::policies::detail::affinity_data ad{};
    ad.init(num_threads, max_cores);
    hpx::threads::policies::callback_notifier notifier{};

    Executor exec(num_threads, ad, notifier);

    // We can't wait for futures on the main thread, so we spawn a thread to
    // run the tests for us.
    hpx::apply(exec, &test_thread_pool_os_executor<Executor>, exec);

    //  NOTE: This is currently required because the executor is reference
    //  counted and copies may be created inside the spawned task, meaning the
    //  destructor does not necessarily block at the end of the scope.
    hpx::util::yield_while(
        [exec]() { return exec.num_pending_closures() != 0; });
}

int main()
{
    using namespace hpx::parallel;

#if defined(HPX_HAVE_LOCAL_SCHEDULER)
    spawn_test<execution::local_queue_os_executor>();
#endif

    spawn_test<execution::local_priority_queue_os_executor>();

#if defined(HPX_HAVE_STATIC_SCHEDULER)
    spawn_test<execution::static_queue_os_executor>();
#endif

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    spawn_test<execution::static_priority_queue_os_executor>();
#endif

    return hpx::util::report_errors();
}
