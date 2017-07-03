//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <array>
#include <algorithm>
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
hpx::thread::id async_test(int passed_through)
{
    HPX_TEST_EQ(passed_through, 42);
    return hpx::this_thread::get_id();
}

void apply_test(hpx::lcos::local::latch& l, hpx::thread::id& id,
    int passed_through)
{
    HPX_TEST_EQ(passed_through, 42);
    id = hpx::this_thread::get_id();
    l.count_down(1);
}

void async_bulk_test(int value, hpx::thread::id tid, int passed_through) //-V813
{
    HPX_TEST(tid != hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_apply(Executor& exec)
{
    hpx::lcos::local::latch l(2);
    hpx::thread::id id;

    hpx::parallel::execution::post(
        exec, &apply_test, std::ref(l), std::ref(id), 42);
    l.count_down_and_wait();

    HPX_TEST(id != hpx::this_thread::get_id());
}

template <typename Executor>
void test_sync(Executor& exec)
{
    HPX_TEST(
        hpx::parallel::execution::sync_execute(exec, &async_test, 42) !=
        hpx::this_thread::get_id());
}

template <typename Executor>
void test_async(Executor& exec)
{
    HPX_TEST(
        hpx::parallel::execution::async_execute(exec, &async_test, 42).get() !=
        hpx::this_thread::get_id());
}

template <typename Executor>
void test_bulk_sync(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    hpx::parallel::execution::bulk_sync_execute(exec,
        hpx::util::bind(&async_bulk_test, _1, tid, _2), v, 42);
    hpx::parallel::execution::bulk_sync_execute(
        exec, &async_bulk_test, v, tid, 42);
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
        hpx::parallel::execution::bulk_async_execute(
            exec, hpx::util::bind(&async_bulk_test, _1, tid, _2), v, 42)
    ).get();
    hpx::when_all(
        hpx::parallel::execution::bulk_async_execute(
            exec, &async_bulk_test, v, tid, 42)
    ).get();
}

boost::atomic<std::size_t> count_apply(0);
boost::atomic<std::size_t> count_sync(0);
boost::atomic<std::size_t> count_async(0);
boost::atomic<std::size_t> count_bulk_sync(0);
boost::atomic<std::size_t> count_bulk_async(0);

template <typename Executor>
void test_executor(std::array<std::size_t, 5> expected)
{
    typedef typename hpx::traits::executor_execution_category<
            Executor
        >::type execution_category;

    HPX_TEST((std::is_same<
            hpx::parallel::execution::parallel_execution_tag,
            execution_category
        >::value));

    count_apply.store(0);
    count_sync.store(0);
    count_async.store(0);
    count_bulk_sync.store(0);
    count_bulk_async.store(0);

    Executor exec;

    test_apply(exec);
    test_sync(exec);
    test_async(exec);
    test_bulk_sync(exec);
    test_bulk_async(exec);

    HPX_TEST_EQ(expected[0], count_apply.load());
    HPX_TEST_EQ(expected[1], count_sync.load());
    HPX_TEST_EQ(expected[2], count_async.load());
    HPX_TEST_EQ(expected[3], count_bulk_sync.load());
    HPX_TEST_EQ(expected[4], count_bulk_async.load());
}

///////////////////////////////////////////////////////////////////////////////
struct test_async_executor1
{
    typedef hpx::parallel::execution::parallel_execution_tag execution_category;

    template <typename F, typename ... Ts>
    static hpx::future<typename hpx::util::invoke_result<F, Ts...>::type>
    async_execute(F && f, Ts &&... ts)
    {
        ++count_async;
        return hpx::async(hpx::launch::async, std::forward<F>(f),
            std::forward<Ts>(ts)...);
    }
};

namespace hpx { namespace traits
{
    template <>
    struct is_two_way_executor<test_async_executor1>
      : std::true_type
    {};
}}

struct test_async_executor2 : test_async_executor1
{
    typedef hpx::parallel::execution::parallel_execution_tag execution_category;

    template <typename F, typename ... Ts>
    static typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type
    sync_execute(F && f, Ts &&... ts)
    {
        ++count_sync;
        return hpx::async(hpx::launch::async, std::forward<F>(f),
            std::forward<Ts>(ts)...).get();
    }
};

namespace hpx { namespace traits
{
    template <>
    struct is_two_way_executor<test_async_executor2>
      : std::true_type
    {};
}}

struct test_async_executor3 : test_async_executor1
{
    typedef hpx::parallel::execution::parallel_execution_tag execution_category;

    template <typename F, typename Shape, typename ... Ts>
    static void bulk_sync_execute(F f, Shape const& shape, Ts &&... ts)
    {
        ++count_bulk_sync;
        std::vector<hpx::future<void> > results;
        for (auto const& elem: shape)
        {
            results.push_back(hpx::async(hpx::launch::async, f, elem, ts...));
        }
        hpx::when_all(results).get();
    }
};

namespace hpx { namespace traits
{
    template <>
    struct is_two_way_executor<test_async_executor3>
      : std::true_type
    {};
}}

struct test_async_executor4 : test_async_executor1
{
    typedef hpx::parallel::execution::parallel_execution_tag execution_category;

    template <typename F, typename Shape, typename ... Ts>
    static std::vector<hpx::future<void> >
    bulk_async_execute(F f, Shape const& shape, Ts &&... ts)
    {
        ++count_bulk_async;
        std::vector<hpx::future<void> > results;
        for (auto const& elem: shape)
        {
            results.push_back(hpx::async(hpx::launch::async, f, elem, ts...));
        }
        return results;
    }
};

namespace hpx { namespace traits
{
    template <>
    struct is_two_way_executor<test_async_executor4>
      : std::true_type
    {};

    template <>
    struct is_bulk_two_way_executor<test_async_executor4>
      : std::true_type
    {};
}}

struct test_async_executor5 : test_async_executor1
{
    typedef hpx::parallel::execution::parallel_execution_tag execution_category;

    template <typename F, typename ... Ts>
    static void post(F && f, Ts &&... ts)
    {
        ++count_apply;
        hpx::apply(std::forward<F>(f), std::forward<Ts>(ts)...);
    }
};

namespace hpx { namespace traits
{
    template <>
    struct is_two_way_executor<test_async_executor5>
      : std::true_type
    {};
}}

template <typename Executor,
    typename B1, typename B2, typename B3, typename B4, typename B5>
constexpr void static_check_executor(B1, B2, B3, B4, B5)
{
    using namespace hpx::traits;

    static_assert(
        has_async_execute_member<Executor>::value == B1::value,
        "check has_async_execute_member<Executor>::value");
    static_assert(
        has_sync_execute_member<Executor>::value == B2::value,
        "check has_sync_execute_member<Executor>::value");
    static_assert(
        has_bulk_sync_execute_member<Executor>::value == B3::value,
        "check has_bulk_sync_execute_member<Executor>::value");
    static_assert(
        has_bulk_async_execute_member<Executor>::value == B4::value,
        "check has_bulk_async_execute_member<Executor>::value");
    static_assert(
        has_post_member<Executor>::value == B5::value,
        "check has_post_member<Executor>::value");

    static_assert(
        !has_then_execute_member<Executor>::value,
        "!has_then_execute_member<Executor>::value");
    static_assert(
        !has_bulk_then_execute_member<Executor>::value,
        "!has_bulk_then_execute_member<Executor>::value");
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[])
{
    std::false_type f;
    std::true_type t;

    static_check_executor<test_async_executor1>(t, f, f, f, f);
    static_check_executor<test_async_executor2>(t, t, f, f, f);
    static_check_executor<test_async_executor3>(t, f, t, f, f);
    static_check_executor<test_async_executor4>(t, f, f, t, f);
    static_check_executor<test_async_executor5>(t, f, f, f, t);

    test_executor<test_async_executor1>({{ 0, 0, 431, 0, 0 }});
    test_executor<test_async_executor2>({{ 0, 1, 430, 0, 0 }});
    test_executor<test_async_executor3>({{ 0, 0, 217, 2, 0 }});
    test_executor<test_async_executor4>({{ 0, 0, 217, 0, 2 }});
    test_executor<test_async_executor5>({{ 1, 0, 430, 0, 0 }});

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
