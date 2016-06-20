//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/parallel_execution.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/deferred_call.hpp>

#include <algorithm>
#include <iterator>
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
void test_bulk_sync(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), 0);

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    std::vector<int> results =
        hpx::parallel::execution::bulk_sync_execute(exec,
            hpx::util::bind(&bulk_test, tid, _1, false, _2), v, 42);

    HPX_TEST(std::equal(
        std::begin(results), std::end(results), std::begin(v))
    );
}

template <typename Executor>
void test_bulk_async(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), 0);

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    std::vector<hpx::future<int> > results =
        hpx::parallel::execution::bulk_async_execute(exec,
            hpx::util::bind(&bulk_test, tid, _1, true, _2), v, 42);

    HPX_TEST(std::equal(
        std::begin(results), std::end(results), std::begin(v),
        [](hpx::future<int>& lhs, const int& rhs)
        {
            return lhs.get() == rhs;
        }));
}

////////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[])
{
    using namespace hpx::parallel;

    execution::sequenced_executor seq_exec;
    test_bulk_sync(seq_exec);

    execution::parallel_executor par_exec;
    execution::parallel_executor par_fork_exec(hpx::launch::fork);
    test_bulk_async(par_exec);
    test_bulk_async(par_fork_exec);

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
