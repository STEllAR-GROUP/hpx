//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

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

    typedef hpx::parallel::executor_traits<Executor> traits;
    traits::apply_execute(exec, &apply_test, std::ref(l), std::ref(id), 42);
    l.count_down_and_wait();

    HPX_TEST(id != hpx::this_thread::get_id());
}

template <typename Executor>
void test_sync(Executor& exec)
{
    typedef hpx::parallel::executor_traits<Executor> traits;
    HPX_TEST(traits::execute(exec, &async_test, 42) != hpx::this_thread::get_id());
}

template <typename Executor>
void test_async(Executor& exec)
{
    typedef hpx::parallel::executor_traits<Executor> traits;
    HPX_TEST(
        traits::async_execute(exec, &async_test, 42).get() !=
        hpx::this_thread::get_id());
}

template <typename Executor>
void test_bulk_sync(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    typedef hpx::parallel::executor_traits<Executor> traits;
    traits::bulk_execute(exec,
        hpx::util::bind(&async_bulk_test, _1, tid, _2), v, 42);
    traits::bulk_execute(exec, &async_bulk_test, v, tid, 42);
}

template <typename Executor>
void test_bulk_async(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    typedef hpx::parallel::executor_traits<Executor> traits;
    hpx::when_all(
        traits::bulk_async_execute(
            exec, hpx::util::bind(&async_bulk_test, _1, tid, _2), v, 42)
    ).get();
    hpx::when_all(
        traits::bulk_async_execute(exec, &async_bulk_test, v, tid, 42)
    ).get();
}

template <typename Executor>
void test_executor()
{
    typedef typename hpx::parallel::executor_traits<
            Executor
        >::execution_category execution_category;

    HPX_TEST((std::is_same<
            hpx::parallel::parallel_execution_tag, execution_category
        >::value));

    Executor exec;

    test_apply(exec);
    test_sync(exec);
    test_async(exec);
    test_bulk_sync(exec);
    test_bulk_async(exec);
}

///////////////////////////////////////////////////////////////////////////////
struct test_async_executor2 : hpx::parallel::executor_tag
{
    template <typename F, typename ... Ts>
    hpx::future<typename hpx::util::invoke_result<F, Ts...>::type>
    async_execute(F && f, Ts &&... ts)
    {
        return hpx::async(hpx::launch::async, std::forward<F>(f),
            std::forward<Ts>(ts)...);
    }
};

struct test_async_executor1 : test_async_executor2
{
    template <typename F, typename ... Ts>
    typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type
    execute(F && f, Ts &&... ts)
    {
        return hpx::async(hpx::launch::async, std::forward<F>(f),
            std::forward<Ts>(ts)...).get();
    }
};

struct test_async_executor3 : test_async_executor2
{
    template <typename F, typename Shape, typename ... Ts>
    void bulk_execute(F f, Shape const& shape, Ts &&... ts)
    {
        std::vector<hpx::future<void> > results;
        for (auto const& elem: shape)
        {
            results.push_back(hpx::async(hpx::launch::async, f, elem, ts...));
        }
        hpx::when_all(results).get();
    }
};

struct test_async_executor4 : test_async_executor2
{
    template <typename F, typename Shape, typename ... Ts>
    std::vector<hpx::future<void> >
    bulk_async_execute(F f, Shape const& shape, Ts &&... ts)
    {
        std::vector<hpx::future<void> > results;
        for (auto const& elem: shape)
        {
            results.push_back(hpx::async(hpx::launch::async, f, elem, ts...));
        }
        return results;
    }
};

struct test_async_executor5 : test_async_executor2
{
    template <typename F, typename ... Ts>
    void post(F && f, Ts &&... ts)
    {
        hpx::apply(std::forward<F>(f), std::forward<Ts>(ts)...);
    }
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[])
{
    test_executor<test_async_executor1>();
    test_executor<test_async_executor2>();
    test_executor<test_async_executor3>();
    test_executor<test_async_executor4>();
    test_executor<test_async_executor5>();

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
