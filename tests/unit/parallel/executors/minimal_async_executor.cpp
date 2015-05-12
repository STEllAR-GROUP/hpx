//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <algorithm>
#include <cstdlib>
#include <vector>

#include <boost/range/functions.hpp>

///////////////////////////////////////////////////////////////////////////////
hpx::thread::id async_test()
{
    return hpx::this_thread::get_id();
}

void async_bulk_test(hpx::thread::id tid, int value)
{
    HPX_TEST(tid != hpx::this_thread::get_id());
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_sync(Executor& exec)
{
    typedef hpx::parallel::executor_traits<Executor> traits;
    HPX_TEST(traits::execute(exec, &async_test) != hpx::this_thread::get_id());
}

template <typename Executor>
void test_async(Executor& exec)
{
    typedef hpx::parallel::executor_traits<Executor> traits;
    HPX_TEST(
        traits::async_execute(exec, &async_test).get() !=
        hpx::this_thread::get_id());
}

template <typename Executor>
void test_bulk_sync(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;

    typedef hpx::parallel::executor_traits<Executor> traits;
    traits::execute(exec, hpx::util::bind(&async_bulk_test, tid, _1), v);
}

template <typename Executor>
void test_bulk_async(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;

    typedef hpx::parallel::executor_traits<Executor> traits;
    traits::async_execute(exec, hpx::util::bind(&async_bulk_test, tid, _1), v).get();
}

template <typename Executor>
void test_executor()
{
    typedef typename hpx::parallel::executor_traits<
            Executor
        >::execution_category execution_category;
    HPX_TEST(
        typeid(hpx::parallel::parallel_execution_tag) ==
        typeid(execution_category));

    Executor exec;
    test_sync(exec);
    test_async(exec);
    test_bulk_sync(exec);
    test_bulk_async(exec);
}

///////////////////////////////////////////////////////////////////////////////
struct test_empty_async_executor
{
};

struct test_async_executor1
{
    template <typename F>
    typename hpx::util::result_of<F()>::type
    execute(F f)
    {
        return hpx::async(hpx::launch::async, f).get();
    }
};

struct test_async_executor2
{
    template <typename F>
    hpx::future<typename hpx::util::result_of<F()>::type>
    async_execute(F f)
    {
        return hpx::async(hpx::launch::async, f);
    }
};

struct test_async_executor3
{
    template <typename F, typename Shape>
    void bulk_execute(F f, Shape const& shape)
    {
        std::vector<hpx::future<void> > results;
        for (auto const& elem: shape)
        {
            results.push_back(hpx::async(hpx::launch::async, f, elem));
        }
        hpx::when_all(results).get();
    }
};

struct test_async_executor4
{
    template <typename F, typename Shape>
    hpx::future<void>
    bulk_async_execute(F f, Shape const& shape)
    {
        std::vector<hpx::future<void> > results;
        for (auto const& elem: shape)
        {
            results.push_back(hpx::async(hpx::launch::async, f, elem));
        }
        return hpx::when_all(results);
    }
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[])
{
    test_executor<test_empty_async_executor>();
    test_executor<test_async_executor1>();
    test_executor<test_async_executor2>();
    test_executor<test_async_executor3>();
    test_executor<test_async_executor4>();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

