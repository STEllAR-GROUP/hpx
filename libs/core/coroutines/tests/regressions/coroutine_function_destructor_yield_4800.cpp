//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test verifies that the destructor of a thread function may yield.

#include <hpx/local/future.hpp>
#include <hpx/local/init.hpp>
#include <hpx/local/thread.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/threading.hpp>

#include <utility>

struct thread_function_yield_destructor
{
    hpx::threads::thread_result_type operator()(hpx::threads::thread_arg_type)
    {
        return {hpx::threads::thread_schedule_state::terminated,
            hpx::threads::invalid_thread_id};
    }

    ~thread_function_yield_destructor()
    {
        hpx::this_thread::yield();
    }
};

struct yielder
{
    ~yielder()
    {
        hpx::this_thread::yield();
    }
};

int hpx_main()
{
    // We supply the thread function ourselves which means that the destructor
    // will be called late in the coroutine call operator.
    {
        hpx::threads::thread_init_data data{thread_function_yield_destructor{},
            "thread_function_yield_destructor"};
        hpx::threads::register_thread(data);
    }

    // This is a more complicated example which sometimes leads to the yielder
    // destructor being called late in the coroutine call operator.
    for (int i = 0; i < 1000; ++i)
    {
        hpx::promise<yielder> p;
        hpx::future<yielder> f = p.get_future();
        hpx::dataflow([](auto&&) {}, f);
        p.set_value(yielder{});
    }

    // In the following two cases the yielder instance gets destructed earlier
    // in the coroutine call operator (before the thread function returns), so
    // these cases should never fail, even when the above two cases may fail.
    for (int i = 0; i < 1000; ++i)
    {
        hpx::promise<yielder> p;
        hpx::future<yielder> f = p.get_future();
        f.then([](auto&&) {});
        p.set_value(yielder{});
    }

    for (int i = 0; i < 1000; ++i)
    {
        yielder y;
        hpx::apply([y = std::move(y)]() {});
    }

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
