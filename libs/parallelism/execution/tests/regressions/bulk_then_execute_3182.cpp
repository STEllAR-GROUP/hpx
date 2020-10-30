//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// #3182: bulk_then_execute has unexpected return type/does not compile

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>

#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/parallel_executors.hpp>

#include <algorithm>
#include <atomic>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
std::atomic<int> void_count(0);
void fun1(int, hpx::shared_future<int> f)
{
    HPX_TEST(f.is_ready());
    HPX_TEST_EQ(f.get(), 42);

    ++void_count;
}

std::atomic<int> int_count(0);
int fun2(int i, hpx::shared_future<int> f)
{
    HPX_TEST(f.is_ready());
    HPX_TEST_EQ(f.get(), 42);

    ++int_count;
    return i;
}

template <typename Executor>
void test_bulk_then_execute(Executor&& exec)
{
    hpx::shared_future<int> f = hpx::make_ready_future(42);
    std::vector<int> v(100);
    std::iota(v.begin(), v.end(), 0);

    {
        hpx::future<void> fut =
            hpx::parallel::execution::bulk_then_execute(exec, &fun1, v, f);
        fut.get();

        HPX_TEST_EQ(void_count.load(), 100);
    }

    {
        hpx::future<std::vector<int>> fut =
            hpx::parallel::execution::bulk_then_execute(exec, &fun2, v, f);
        auto result = fut.get();

        HPX_TEST_EQ(int_count.load(), 100);
        HPX_TEST(result == v);
    }
}

int hpx_main()
{
    {
        void_count.store(0);
        int_count.store(0);

        hpx::execution::parallel_executor exec;
        test_bulk_then_execute(exec);
    }

    {
        void_count.store(0);
        int_count.store(0);

        hpx::parallel::execution::pool_executor exec{"default"};
        test_bulk_then_execute(exec);
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}
