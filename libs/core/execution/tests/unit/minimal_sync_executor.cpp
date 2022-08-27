//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/execution.hpp>
#include <hpx/local/future.hpp>
#include <hpx/local/init.hpp>
#include <hpx/local/thread.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
hpx::thread::id sync_test(int passed_through)
{
    HPX_TEST_EQ(passed_through, 42);
    return hpx::this_thread::get_id();
}

void sync_test_void(int passed_through)
{
    HPX_TEST_EQ(passed_through, 42);
}

hpx::thread::id sync_bulk_test(int, hpx::thread::id tid,
    int passed_through)    //-V813
{
    HPX_TEST_EQ(tid, hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);
    return hpx::this_thread::get_id();
}

void sync_bulk_test_void(
    int, hpx::thread::id tid, int passed_through)    //-V813
{
    HPX_TEST_EQ(tid, hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);
}

hpx::thread::id then_test(hpx::future<void> f, int passed_through)
{
    HPX_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    HPX_TEST_EQ(passed_through, 42);
    return hpx::this_thread::get_id();
}

void then_test_void(hpx::future<void> f, int passed_through)
{
    HPX_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    HPX_TEST_EQ(passed_through, 42);
}

hpx::thread::id then_bulk_test(int, hpx::shared_future<void> f,
    hpx::thread::id tid, int passed_through)    //-V813
{
    HPX_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    HPX_TEST_EQ(tid, hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);

    return hpx::this_thread::get_id();
}

void then_bulk_test_void(int, hpx::shared_future<void> f, hpx::thread::id tid,
    int passed_through)    //-V813
{
    HPX_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    HPX_TEST_EQ(tid, hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_sync(Executor& exec)
{
    HPX_TEST(hpx::parallel::execution::sync_execute(exec, &sync_test, 42) ==
        hpx::this_thread::get_id());

    hpx::parallel::execution::sync_execute(exec, &sync_test_void, 42);
}

template <typename Executor>
void test_async(Executor& exec)
{
    HPX_TEST(
        hpx::parallel::execution::async_execute(exec, &sync_test, 42).get() ==
        hpx::this_thread::get_id());

    hpx::parallel::execution::async_execute(exec, &sync_test_void, 42).get();
}

template <typename Executor>
void test_then(Executor& exec)
{
    hpx::future<void> f1 = hpx::make_ready_future();
    HPX_TEST(hpx::parallel::execution::then_execute(exec, &then_test, f1, 42)
                 .get() == hpx::this_thread::get_id());

    hpx::future<void> f2 = hpx::make_ready_future();
    hpx::parallel::execution::then_execute(exec, &then_test_void, f2, 42).get();
}

template <typename Executor>
void test_bulk_sync(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::placeholders::_1;
    using hpx::placeholders::_2;

    std::vector<hpx::thread::id> ids =
        hpx::parallel::execution::bulk_sync_execute(
            exec, hpx::bind(&sync_bulk_test, _1, tid, _2), v, 42);
    for (auto const& id : ids)
    {
        HPX_TEST_EQ(id, hpx::this_thread::get_id());
    }

    ids = hpx::parallel::execution::bulk_sync_execute(
        exec, &sync_bulk_test, v, tid, 42);
    for (auto const& id : ids)
    {
        HPX_TEST_EQ(id, hpx::this_thread::get_id());
    }

    hpx::parallel::execution::bulk_sync_execute(
        exec, hpx::bind(&sync_bulk_test_void, _1, tid, _2), v, 42);
    hpx::parallel::execution::bulk_sync_execute(
        exec, &sync_bulk_test_void, v, tid, 42);
}

template <typename Executor>
void test_bulk_async(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::placeholders::_1;
    using hpx::placeholders::_2;

    hpx::when_all(hpx::parallel::execution::bulk_async_execute(
                      exec, hpx::bind(&sync_bulk_test, _1, tid, _2), v, 42))
        .get();
    hpx::when_all(hpx::parallel::execution::bulk_async_execute(
                      exec, &sync_bulk_test, v, tid, 42))
        .get();

    hpx::when_all(hpx::parallel::execution::bulk_async_execute(exec,
                      hpx::bind(&sync_bulk_test_void, _1, tid, _2), v, 42))
        .get();
    hpx::when_all(hpx::parallel::execution::bulk_async_execute(
                      exec, &sync_bulk_test_void, v, tid, 42))
        .get();
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_bulk_then(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    hpx::shared_future<void> f = hpx::make_ready_future();

    std::vector<hpx::thread::id> tids =
        hpx::parallel::execution::bulk_then_execute(
            exec, &then_bulk_test, v, f, tid, 42)
            .get();

    for (auto const& tid : tids)
    {
        HPX_TEST_EQ(tid, hpx::this_thread::get_id());
    }

    hpx::parallel::execution::bulk_then_execute(
        exec, &then_bulk_test_void, v, f, tid, 42)
        .get();
}

///////////////////////////////////////////////////////////////////////////////
std::atomic<std::size_t> count_sync(0);
std::atomic<std::size_t> count_bulk_sync(0);

template <typename Executor>
void test_executor(std::array<std::size_t, 2> expected)
{
    typedef typename hpx::traits::executor_execution_category<Executor>::type
        execution_category;

    HPX_TEST((std::is_same<hpx::execution::sequenced_execution_tag,
        execution_category>::value));

    count_sync.store(0);
    count_bulk_sync.store(0);

    Executor exec;

    test_sync(exec);
    test_async(exec);
    test_then(exec);

    test_bulk_sync(exec);
    test_bulk_async(exec);
    test_bulk_then(exec);

    HPX_TEST_EQ(expected[0], count_sync.load());
    HPX_TEST_EQ(expected[1], count_bulk_sync.load());
}

///////////////////////////////////////////////////////////////////////////////
struct test_sync_executor1
{
    typedef hpx::execution::sequenced_execution_tag execution_category;

    template <typename F, typename... Ts>
    static typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type
    sync_execute(F&& f, Ts&&... ts)
    {
        ++count_sync;
        return hpx::util::invoke(std::forward<F>(f), std::forward<Ts>(ts)...);
    }
};

namespace hpx { namespace parallel { namespace execution {
    template <>
    struct is_one_way_executor<test_sync_executor1> : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution

struct test_sync_executor2 : test_sync_executor1
{
    typedef hpx::execution::sequenced_execution_tag execution_category;

    template <typename F, typename Shape, typename... Ts>
    static typename hpx::parallel::execution::detail::bulk_execute_result<F,
        Shape, Ts...>::type
    call(std::false_type, F&& f, Shape const& shape, Ts&&... ts)
    {
        typedef
            typename hpx::parallel::execution::detail::bulk_function_result<F,
                Shape, Ts...>::type result_type;

        std::vector<result_type> results;
        for (auto const& elem : shape)
        {
            results.push_back(hpx::util::invoke(f, elem, ts...));
        }
        return results;
    }

    template <typename F, typename Shape, typename... Ts>
    static void call(std::true_type, F&& f, Shape const& shape, Ts&&... ts)
    {
        for (auto const& elem : shape)
        {
            hpx::util::invoke(f, elem, ts...);
        }
    }

    template <typename F, typename Shape, typename... Ts>
    static typename hpx::parallel::execution::detail::bulk_execute_result<F,
        Shape, Ts...>::type
    bulk_sync_execute(F&& f, Shape const& shape, Ts&&... ts)
    {
        ++count_bulk_sync;

        typedef
            typename std::is_void<typename hpx::parallel::execution::detail::
                    bulk_function_result<F, Shape, Ts...>::type>::type is_void;

        return call(
            is_void(), std::forward<F>(f), shape, std::forward<Ts>(ts)...);
    }
};

namespace hpx { namespace parallel { namespace execution {
    template <>
    struct is_one_way_executor<test_sync_executor2> : std::true_type
    {
    };

    template <>
    struct is_bulk_one_way_executor<test_sync_executor2> : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution

template <typename Executor, typename B1, typename B2>
void static_check_executor(B1, B2)
{
    using namespace hpx::traits;

    static_assert(has_sync_execute_member<Executor>::value == B1::value,
        "check has_sync_execute_member<Executor>::value");
    static_assert(has_bulk_sync_execute_member<Executor>::value == B2::value,
        "check has_bulk_sync_execute_member<Executor>::value");

    static_assert(!has_async_execute_member<Executor>::value,
        "!has_async_execute_member<Executor>::value");
    static_assert(!has_bulk_async_execute_member<Executor>::value,
        "!has_bulk_async_execute_member<Executor>::value");
    static_assert(!has_then_execute_member<Executor>::value,
        "!has_then_execute_member<Executor>::value");
    static_assert(!has_bulk_then_execute_member<Executor>::value,
        "!has_bulk_then_execute_member<Executor>::value");
    static_assert(
        !has_post_member<Executor>::value, "!has_post_member<Executor>::value");
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    std::false_type f;
    std::true_type t;

    static_check_executor<test_sync_executor1>(t, f);
    static_check_executor<test_sync_executor2>(t, t);

    test_executor<test_sync_executor1>({{1078, 0}});
    test_executor<test_sync_executor2>({{436, 6}});

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
