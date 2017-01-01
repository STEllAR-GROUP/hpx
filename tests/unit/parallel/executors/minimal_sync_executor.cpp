//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/atomic.hpp>
#include <boost/range/functions.hpp>

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

hpx::thread::id sync_bulk_test(int value, hpx::thread::id tid,
    int passed_through) //-V813
{
    HPX_TEST_EQ(passed_through, 42);
    return hpx::this_thread::get_id();
}

void sync_bulk_test_void(int value, hpx::thread::id tid, int passed_through) //-V813
{
    HPX_TEST(tid == hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);
}

hpx::thread::id then_test(hpx::future<void> f, int passed_through)
{
    HPX_ASSERT(f.is_ready());   // make sure, future is ready

    f.get();                    // propagate exceptions

    HPX_TEST_EQ(passed_through, 42);
    return hpx::this_thread::get_id();
}

void then_test_void(hpx::future<void> f, int passed_through)
{
    HPX_ASSERT(f.is_ready());   // make sure, future is ready

    f.get();                    // propagate exceptions

    HPX_TEST_EQ(passed_through, 42);
}

hpx::thread::id then_bulk_test(int value, hpx::shared_future<void> f,
    hpx::thread::id tid, int passed_through) //-V813
{
    HPX_ASSERT(f.is_ready());   // make sure, future is ready

    f.get();                    // propagate exceptions

    HPX_TEST(tid == hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);

    return hpx::this_thread::get_id();
}

void then_bulk_test_void(int value, hpx::shared_future<void> f,
    hpx::thread::id tid, int passed_through) //-V813
{
    HPX_ASSERT(f.is_ready());   // make sure, future is ready

    f.get();                    // propagate exceptions

    HPX_TEST(tid == hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_sync(Executor& exec)
{
    HPX_TEST(
        hpx::parallel::execution::sync_execute(exec, &sync_test, 42) ==
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
    HPX_TEST(
        hpx::parallel::execution::then_execute(exec, &then_test, f1, 42).get() ==
        hpx::this_thread::get_id());

    hpx::future<void> f2 = hpx::make_ready_future();
    hpx::parallel::execution::then_execute(exec, &then_test_void, f2, 42).get();
}

template <typename Executor>
void test_bulk_sync(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    std::vector<hpx::thread::id> ids =
        hpx::parallel::execution::sync_bulk_execute(
            exec, hpx::util::bind(&sync_bulk_test, _1, tid, _2), v, 42);
    for (auto const& id : ids)
    {
        HPX_TEST(id == hpx::this_thread::get_id());
    }

    ids = hpx::parallel::execution::sync_bulk_execute(
        exec, &sync_bulk_test, v, tid, 42);
    for (auto const& id : ids)
    {
        HPX_TEST(id == hpx::this_thread::get_id());
    }

    hpx::parallel::execution::sync_bulk_execute(
        exec, hpx::util::bind(&sync_bulk_test_void, _1, tid, _2), v, 42);
    hpx::parallel::execution::sync_bulk_execute(
        exec, &sync_bulk_test_void, v, tid, 42);
}

template <typename Executor>
void test_bulk_async(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    hpx::when_all(
        hpx::parallel::execution::async_bulk_execute(
            exec, hpx::util::bind(&sync_bulk_test, _1, tid, _2), v, 42)
    ).get();
    hpx::when_all(
        hpx::parallel::execution::async_bulk_execute(
            exec, &sync_bulk_test, v, tid, 42)
    ).get();

    hpx::when_all(
        hpx::parallel::execution::async_bulk_execute(
            exec, hpx::util::bind(&sync_bulk_test_void, _1, tid, _2), v, 42)
    ).get();
    hpx::when_all(
        hpx::parallel::execution::async_bulk_execute(
            exec, &sync_bulk_test_void, v, tid, 42)
    ).get();
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_bulk_then(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    hpx::shared_future<void> f = hpx::make_ready_future();

    std::vector<hpx::thread::id> tids =
        hpx::parallel::execution::then_bulk_execute(
            exec, &then_bulk_test, v, f, tid, 42).get();

    for (auto const& tid : tids)
    {
        HPX_TEST(tid == hpx::this_thread::get_id());
    }

    hpx::parallel::execution::then_bulk_execute(
        exec, &then_bulk_test_void, v, f, tid, 42).get();
}

///////////////////////////////////////////////////////////////////////////////
boost::atomic<std::size_t> count_sync(0);
boost::atomic<std::size_t> count_bulk_sync(0);

template <typename Executor>
void test_executor(std::array<std::size_t, 2> expected)
{
    typedef typename hpx::parallel::execution::executor_execution_category<
            Executor
        >::type execution_category;

    HPX_TEST((std::is_same<
            hpx::parallel::execution::sequenced_execution_tag,
            execution_category
        >::value));

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
    typedef hpx::parallel::execution::sequenced_execution_tag execution_category;

    template <typename F, typename ... Ts>
    static typename hpx::util::detail::deferred_result_of<F(Ts&&...)>::type
    sync_execute(F && f, Ts &&... ts)
    {
        ++count_sync;
        return hpx::util::invoke(std::forward<F>(f), std::forward<Ts>(ts)...);
    }
};

namespace hpx { namespace traits
{
    template <>
    struct is_one_way_executor<test_sync_executor1>
      : std::true_type
    {};
}}

struct test_sync_executor2 : test_sync_executor1
{
    typedef hpx::parallel::execution::sequenced_execution_tag execution_category;

    template <typename F, typename Shape, typename ... Ts>
    static typename hpx::parallel::execution::detail::bulk_execute_result<
        F, Shape, Ts...
    >::type
    call(std::false_type, F && f, Shape const& shape, Ts &&... ts)
    {
        typedef typename hpx::parallel::execution::detail::bulk_function_result<
                    F, Shape, Ts...
                >::type result_type;

        std::vector<result_type> results;
        for (auto const& elem: shape)
        {
            results.push_back(hpx::util::invoke(f, elem, ts...));
        }
        return results;
    }

    template <typename F, typename Shape, typename ... Ts>
    static void
    call(std::true_type, F && f, Shape const& shape, Ts &&... ts)
    {
        for (auto const& elem: shape)
        {
            hpx::util::invoke(f, elem, ts...);
        }
    }

    template <typename F, typename Shape, typename ... Ts>
    static typename hpx::parallel::execution::detail::bulk_execute_result<
        F, Shape, Ts...
    >::type
    sync_bulk_execute(F && f, Shape const& shape, Ts &&... ts)
    {
        ++count_bulk_sync;

        typedef typename std::is_void<
                typename hpx::parallel::execution::detail::bulk_function_result<
                    F, Shape, Ts...
                >::type
            >::type is_void;

        return call(is_void(), std::forward<F>(f), shape,
            std::forward<Ts>(ts)...);
    }
};

namespace hpx { namespace traits
{
    template <>
    struct is_one_way_executor<test_sync_executor2>
      : std::true_type
    {};

    template <>
    struct is_bulk_one_way_executor<test_sync_executor2>
      : std::true_type
    {};
}}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[])
{
    test_executor<test_sync_executor1>({{ 1078, 0 }});
    test_executor<test_sync_executor2>({{ 436, 6 }});

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
