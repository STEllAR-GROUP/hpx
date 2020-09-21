//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>

///////////////////////////////////////////////////////////////////////////////
std::atomic<std::int32_t> accumulator;
hpx::lcos::local::condition_variable_any result_cv;

void increment(std::int32_t i)
{
    accumulator += i;
    result_cv.notify_one();
}

void increment_with_future(hpx::shared_future<std::int32_t> fi)
{
    accumulator += fi.get();
    result_cv.notify_one();
}

///////////////////////////////////////////////////////////////////////////////
struct increment_function_object
{
    void operator()(std::int32_t i) const
    {
        accumulator += i;
    }
};

///////////////////////////////////////////////////////////////////////////////
struct increment_type
{
    void call(std::int32_t i) const
    {
        accumulator += i;
    }
};

auto increment_lambda = [](std::int32_t i) { accumulator += i; };

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_apply_with_executor(Executor& exec)
{
    accumulator.store(0);

    {
        using hpx::util::placeholders::_1;

        hpx::apply(exec, &increment, 1);
        hpx::apply(exec, hpx::util::bind(&increment, 1));
        hpx::apply(exec, hpx::util::bind(&increment, _1), 1);
    }

    {
        hpx::lcos::local::promise<std::int32_t> p;
        hpx::shared_future<std::int32_t> f = p.get_future();

        p.set_value(1);

        using hpx::util::placeholders::_1;

        hpx::apply(exec, &increment_with_future, f);
        hpx::apply(exec, hpx::util::bind(&increment_with_future, f));
        hpx::apply(exec, hpx::util::bind(&increment_with_future, _1), f);
    }

    {
        using hpx::util::placeholders::_1;

        hpx::apply(exec, increment, 1);
        hpx::apply(exec, hpx::util::bind(increment, 1));
        hpx::apply(exec, hpx::util::bind(increment, _1), 1);
    }

    {
        increment_type inc;

        using hpx::util::placeholders::_1;
        using hpx::util::placeholders::_2;

        hpx::apply(exec, &increment_type::call, inc, 1);
        hpx::apply(exec, hpx::util::bind(&increment_type::call, inc, 1));
        hpx::apply(exec, hpx::util::bind(&increment_type::call, inc, _1), 1);
    }

    {
        increment_function_object obj;

        using hpx::util::placeholders::_1;
        using hpx::util::placeholders::_2;

        hpx::apply(exec, obj, 1);
        hpx::apply(exec, hpx::util::bind(obj, 1));
        hpx::apply(exec, hpx::util::bind(obj, _1), 1);
    }

    {
        using hpx::util::placeholders::_1;
        using hpx::util::placeholders::_2;

        hpx::apply(exec, increment_lambda, 1);
        hpx::apply(exec, hpx::util::bind(increment_lambda, 1));
        hpx::apply(exec, hpx::util::bind(increment_lambda, _1), 1);
    }

    hpx::lcos::local::no_mutex result_mutex;
    std::unique_lock<hpx::lcos::local::no_mutex> l(result_mutex);
    result_cv.wait_for(l, std::chrono::seconds(1),
        hpx::util::bind(
            std::equal_to<std::int32_t>(), std::ref(accumulator), 18));

    HPX_TEST_EQ(accumulator.load(), 18);
}

int hpx_main()
{
    {
        hpx::execution::sequenced_executor exec;
        test_apply_with_executor(exec);
    }

    {
        hpx::execution::parallel_executor exec;
        test_apply_with_executor(exec);
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(
        hpx::init(argc, argv), 0, "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
