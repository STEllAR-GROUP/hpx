//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/util.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <algorithm>
#include <cstdlib>
#include <vector>
#include <utility>

#include <boost/range/functions.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/chrono.hpp>

using namespace boost::chrono;

///////////////////////////////////////////////////////////////////////////////
hpx::thread::id sync_test()
{
    return hpx::this_thread::get_id();
}

void apply_test(hpx::lcos::local::latch& l, hpx::thread::id& id)
{
    id = hpx::this_thread::get_id();
    l.count_down(1);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_timed_apply(Executor& exec)
{
    {
        hpx::lcos::local::latch l(2);
        hpx::thread::id id;

        typedef hpx::parallel::timed_executor_traits<Executor> traits;
        traits::apply_execute_after(exec, milliseconds(10),
            hpx::util::bind(&apply_test, boost::ref(l), boost::ref(id))
        );
        l.count_down_and_wait();

        HPX_TEST(id == hpx::this_thread::get_id());
    }

    {
        hpx::lcos::local::latch l(2);
        hpx::thread::id id;

        typedef hpx::parallel::timed_executor_traits<Executor> traits;
        traits::apply_execute_at(exec, steady_clock::now()+milliseconds(10),
            hpx::util::bind(&apply_test, boost::ref(l), boost::ref(id))
        );
        l.count_down_and_wait();

        HPX_TEST(id == hpx::this_thread::get_id());
    }
}

template <typename Executor>
void test_timed_sync(Executor& exec)
{
    typedef hpx::parallel::timed_executor_traits<Executor> traits;
    HPX_TEST(traits::execute_after(exec, milliseconds(10), &sync_test) ==
        hpx::this_thread::get_id());
    HPX_TEST(traits::execute_at(
            exec, steady_clock::now()+milliseconds(10), &sync_test) ==
        hpx::this_thread::get_id());
}

template <typename Executor>
void test_timed_async(Executor& exec)
{
    typedef hpx::parallel::timed_executor_traits<Executor> traits;
    HPX_TEST(
        traits::async_execute_after(exec, milliseconds(10), &sync_test).get() ==
        hpx::this_thread::get_id());
    HPX_TEST(
        traits::async_execute_at(
            exec, steady_clock::now()+milliseconds(10), &sync_test
        ).get() == hpx::this_thread::get_id());
}

template <typename Executor>
void test_timed_executor()
{
    typedef typename hpx::parallel::timed_executor_traits<
            Executor
        >::execution_category execution_category;

    HPX_TEST((boost::is_same<
            hpx::parallel::sequential_execution_tag, execution_category
        >::value));

    Executor exec;

    test_timed_apply(exec);
    test_timed_sync(exec);
    test_timed_async(exec);
}

///////////////////////////////////////////////////////////////////////////////
struct test_timed_async_executor2 : hpx::parallel::timed_executor_tag
{
    typedef hpx::parallel::sequential_execution_tag execution_category;

    template <typename F>
    hpx::future<typename hpx::util::result_of<
        typename hpx::util::decay<F>::type()
    >::type>
    async_execute(F && f)
    {
        return hpx::async(hpx::launch::sync, std::forward<F>(f));
    }

    template <typename F>
    hpx::future<typename hpx::util::result_of<
        typename hpx::util::decay<F>::type()
    >::type>
    async_execute_at(hpx::util::steady_time_point const& abs_time, F && f)
    {
        hpx::this_thread::sleep_until(abs_time);
        return hpx::async(hpx::launch::sync, std::forward<F>(f));
    }

    std::size_t os_thread_count()
    {
        return 1;
    }
};

struct test_timed_async_executor1 : test_timed_async_executor2
{
    typedef hpx::parallel::sequential_execution_tag execution_category;

    template <typename F>
    typename hpx::util::result_of<F()>::type
    execute_at(hpx::util::steady_time_point const& abs_time, F f)
    {
        hpx::this_thread::sleep_until(abs_time);
        return f();
    }
};

struct test_timed_async_executor3 : test_timed_async_executor1
{
    typedef hpx::parallel::sequential_execution_tag execution_category;

    template <typename F>
    void apply_execute_at(hpx::util::steady_time_point const& abs_time, F && f)
    {
        this->execute_at(abs_time, std::forward<F>(f));
    }
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[])
{
    test_timed_executor<test_timed_async_executor1>();
    test_timed_executor<test_timed_async_executor2>();
    test_timed_executor<test_timed_async_executor3>();

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
