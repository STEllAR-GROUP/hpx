//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/deferred_call.hpp>

#include <boost/range/functions.hpp>

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
int bulk_test(hpx::thread::id tid, int value, bool is_par, int passed_through) //-V813
{
    HPX_TEST(is_par == (tid != hpx::this_thread::get_id()));
    HPX_TEST_EQ(passed_through, 42);
    return value;
}

template <typename Executor>
void test_bulk_async(Executor& exec, bool is_par = true)
{
    typedef hpx::parallel::executor_traits<Executor> traits;

    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), 0);

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    std::vector<hpx::future<int> > results = traits::bulk_async_execute(
        exec, hpx::util::bind(&bulk_test, tid, _1, is_par, _2), v, 42);

    HPX_TEST(std::equal(
        boost::begin(results), boost::end(results), boost::begin(v),
        [](hpx::future<int>& lhs, const int& rhs)
        {
            return lhs.get() == rhs;
        }));
}

////////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[])
{
    using namespace hpx::parallel;

    sequential_executor seq_exec;
    parallel_executor par_exec;
    parallel_executor par_fork_exec(hpx::launch::fork);

    test_bulk_async(seq_exec, false);
    test_bulk_async(par_exec);
    test_bulk_async(par_fork_exec);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {
        "hpx.os_threads=" +
            std::to_string(hpx::threads::hardware_concurrency())
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
