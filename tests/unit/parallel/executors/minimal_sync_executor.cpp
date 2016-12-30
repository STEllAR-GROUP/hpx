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

void sync_bulk_test(int value, hpx::thread::id tid, int passed_through) //-V813
{
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
}

template <typename Executor>
void test_bulk_sync(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    hpx::parallel::execution::sync_bulk_execute(
        exec, hpx::util::bind(&sync_bulk_test, _1, tid, _2), v, 42);
    hpx::parallel::execution::sync_bulk_execute(
        exec, &sync_bulk_test, v, tid, 42);
}

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
    test_bulk_sync(exec);

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

    std::size_t processing_units_count()
    {
        return 1;
    }
};

namespace hpx { namespace traits
{
    template <>
    struct is_one_way_executor<test_sync_executor1>
      : std::true_type
    {};

    template <>
    struct is_bulk_one_way_executor<test_sync_executor1>
      : std::true_type
    {};
}}

struct test_sync_executor2 : test_sync_executor1
{
    typedef hpx::parallel::execution::sequenced_execution_tag execution_category;

    template <typename F, typename Shape, typename ... Ts>
    static void sync_bulk_execute(F f, Shape const& shape, Ts &&... ts)
    {
        ++count_bulk_sync;
        for (auto const& elem: shape)
        {
            hpx::util::invoke(f, elem, ts...);
        }
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
    test_executor<test_sync_executor1>({{ 215, 0 }});
    test_executor<test_sync_executor2>({{ 1, 2 }});

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
