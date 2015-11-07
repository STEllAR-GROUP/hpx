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
#include <utility>

#include <boost/range/functions.hpp>
#include <boost/type_traits/is_same.hpp>

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

void sync_bulk_test(hpx::thread::id tid, int value)
{
    HPX_TEST(tid == hpx::this_thread::get_id());
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_apply(Executor& exec)
{
    hpx::lcos::local::latch l(2);
    hpx::thread::id id;

    typedef hpx::parallel::executor_traits<Executor> traits;
    traits::apply_execute(exec,
        hpx::util::bind(&apply_test, boost::ref(l), boost::ref(id))
    );
    l.count_down_and_wait();

    HPX_TEST(id == hpx::this_thread::get_id());
}

template <typename Executor>
void test_sync(Executor& exec)
{
    typedef hpx::parallel::executor_traits<Executor> traits;
    HPX_TEST(traits::execute(exec, &sync_test) == hpx::this_thread::get_id());
}

template <typename Executor>
void test_async(Executor& exec)
{
    typedef hpx::parallel::executor_traits<Executor> traits;
    HPX_TEST(
        traits::async_execute(exec, &sync_test).get() ==
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
    traits::execute(exec, hpx::util::bind(&sync_bulk_test, tid, _1), v);
}

template <typename Executor>
void test_bulk_async(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;

    typedef hpx::parallel::executor_traits<Executor> traits;
    hpx::when_all(traits::async_execute(
        exec, hpx::util::bind(&sync_bulk_test, tid, _1), v)).get();
}

template <typename Executor>
void test_executor()
{
    typedef typename hpx::parallel::executor_traits<
            Executor
        >::execution_category execution_category;

    HPX_TEST((boost::is_same<
            hpx::parallel::sequential_execution_tag, execution_category
        >::value));

    Executor exec;

    test_apply(exec);
    test_sync(exec);
    test_async(exec);
    test_bulk_sync(exec);
    test_bulk_async(exec);
}

///////////////////////////////////////////////////////////////////////////////
struct test_sync_executor2 : hpx::parallel::executor_tag
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

    std::size_t processing_units_count()
    {
        return 1;
    }
};

struct test_sync_executor1 : test_sync_executor2
{
    typedef hpx::parallel::sequential_execution_tag execution_category;

    template <typename F>
    typename hpx::util::result_of<F()>::type
    execute(F f)
    {
        return f();
    }
};

struct test_sync_executor3 : test_sync_executor2
{
    typedef hpx::parallel::sequential_execution_tag execution_category;

    template <typename F, typename Shape>
    void bulk_execute(F f, Shape const& shape)
    {
        for (auto const& elem: shape)
            f(elem);
    }
};

struct test_sync_executor4 : test_sync_executor2
{
    typedef hpx::parallel::sequential_execution_tag execution_category;

    template <typename F, typename Shape>
    std::vector<hpx::future<void> >
    bulk_async_execute(F f, Shape const& shape)
    {
        std::vector<hpx::future<void> > results;
        for (auto const& elem: shape)
            results.push_back(
                async_execute(hpx::util::deferred_call(f, elem))
            );
        return std::move(results);
    }
};

struct test_sync_executor5 : test_sync_executor1  // derive from sync execute
{
    typedef hpx::parallel::sequential_execution_tag execution_category;

    template <typename F>
    void apply_execute(F && f)
    {
        this->execute(std::forward<F>(f));
    }
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[])
{
    test_executor<test_sync_executor1>();
    test_executor<test_sync_executor2>();
    test_executor<test_sync_executor3>();
    test_executor<test_sync_executor4>();
    test_executor<test_sync_executor5>();

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
