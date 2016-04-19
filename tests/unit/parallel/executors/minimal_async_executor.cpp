//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/decay.hpp>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include <boost/range/functions.hpp>
#include <boost/type_traits/is_same.hpp>

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

void async_bulk_test(hpx::thread::id tid, int value, int passed_through)
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
    traits::apply_execute(exec, &apply_test, boost::ref(l), boost::ref(id), 42);
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
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    typedef hpx::parallel::executor_traits<Executor> traits;
    traits::bulk_execute(exec,
        hpx::util::bind(&async_bulk_test, tid, _1, _2), v, 42);
}

template <typename Executor>
void test_bulk_async(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    typedef hpx::parallel::executor_traits<Executor> traits;
    hpx::when_all(traits::bulk_async_execute(
        exec, hpx::util::bind(&async_bulk_test, tid, _1, _2), v, 42)).get();
}

template <typename Executor>
void test_executor()
{
    typedef typename hpx::parallel::executor_traits<
            Executor
        >::execution_category execution_category;

    HPX_TEST((boost::is_same<
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
    hpx::future<typename hpx::util::result_of<F&&(Ts&&...)>::type>
    async_execute(F && f, Ts &&... ts)
    {
        return hpx::async(hpx::launch::async, std::forward<F>(f),
            std::forward<Ts>(ts)...);
    }
};

struct test_async_executor1 : test_async_executor2
{
    template <typename F, typename ... Ts>
    typename hpx::util::result_of<F&&(Ts&&...)>::type
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
    void apply_execute(F && f, Ts &&... ts)
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
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        std::to_string(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
