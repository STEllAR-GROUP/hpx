//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <system_error>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
std::int32_t increment(std::int32_t i)
{
    return i + 1;
}
HPX_PLAIN_ACTION(increment);    // defines increment_action

std::int32_t increment_with_future(hpx::shared_future<std::int32_t> fi)
{
    return fi.get() + 1;
}
HPX_PLAIN_ACTION(increment_with_future);

///////////////////////////////////////////////////////////////////////////////
std::int32_t mult2(std::int32_t i)
{
    return i * 2;
}
HPX_PLAIN_ACTION(mult2);    // defines mult2_action

///////////////////////////////////////////////////////////////////////////////
std::atomic<int> callback_called(0);

#if defined(HPX_HAVE_NETWORKING)
void cb(std::error_code const&, hpx::parcelset::parcel const&)
{
    ++callback_called;
}
#else
void cb()
{
    ++callback_called;
}
#endif

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    using hpx::make_continuation;

    increment_action inc;
    increment_with_future_action inc_f;
    mult2_action mult;

    // test locally, fully equivalent to plain hpx::async
    {
        callback_called.store(0);
        hpx::future<int> f1 = hpx::async_continue_cb(
            inc, make_continuation(), hpx::find_here(), &cb, 42);
        HPX_TEST_EQ(f1.get(), 43);

        hpx::lcos::promise<std::int32_t> p;
        hpx::shared_future<std::int32_t> f = p.get_future();

        hpx::future<int> f2 = hpx::async_continue_cb(
            inc_f, make_continuation(), hpx::find_here(), &cb, f);

        p.set_value(42);
        HPX_TEST_EQ(f2.get(), 43);

        // The callback should have been called 2 times. wait for a short period
        // of time, to allow it for it to be fully executed
        hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
        HPX_TEST_EQ(callback_called.load(), 2);
    }

    // test remotely, if possible, fully equivalent to plain hpx::async
    std::vector<hpx::id_type> localities = hpx::find_remote_localities();
    if (!localities.empty())
    {
        callback_called.store(0);
        hpx::future<int> f1 = hpx::async_continue_cb(
            inc, make_continuation(), localities[0], &cb, 42);
        HPX_TEST_EQ(f1.get(), 43);

        hpx::lcos::promise<std::int32_t> p;
        hpx::shared_future<std::int32_t> f = p.get_future();

        hpx::future<int> f2 = hpx::async_continue_cb(
            inc_f, make_continuation(), localities[0], &cb, f);

        p.set_value(42);
        HPX_TEST_EQ(f2.get(), 43);

        // The callback should have been called 2 times. wait for a short period
        // of time, to allow it for it to be fully executed
        hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
        HPX_TEST_EQ(callback_called.load(), 2);
    }

    // test chaining locally
    {
        callback_called.store(0);
        hpx::future<int> f = hpx::async_continue_cb(
            inc, make_continuation(mult), hpx::find_here(), &cb, 42);
        HPX_TEST_EQ(f.get(), 86);

        f = hpx::async_continue_cb(inc,
            make_continuation(mult, make_continuation()), hpx::find_here(), &cb,
            42);
        HPX_TEST_EQ(f.get(), 86);

        f = hpx::async_continue_cb(inc,
            make_continuation(mult, make_continuation(inc)), hpx::find_here(),
            &cb, 42);
        HPX_TEST_EQ(f.get(), 87);

        f = hpx::async_continue_cb(inc,
            make_continuation(
                mult, make_continuation(inc, make_continuation())),
            hpx::find_here(), &cb, 42);
        HPX_TEST_EQ(f.get(), 87);

        // The callback should have been called 4 times. wait for a short period
        // of time, to allow it for it to be fully executed
        hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
        HPX_TEST_EQ(callback_called.load(), 4);
    }

    // test chaining remotely, if possible
    if (!localities.empty())
    {
        callback_called.store(0);
        hpx::future<int> f = hpx::async_continue_cb(inc,
            make_continuation(mult, localities[0]), localities[0], &cb, 42);
        HPX_TEST_EQ(f.get(), 86);

        f = hpx::async_continue_cb(inc,
            make_continuation(mult, localities[0], make_continuation()),
            localities[0], &cb, 42);
        HPX_TEST_EQ(f.get(), 86);

        f = hpx::async_continue_cb(inc,
            make_continuation(mult, localities[0], make_continuation(inc)),
            localities[0], &cb, 42);
        HPX_TEST_EQ(f.get(), 87);

        f = hpx::async_continue_cb(inc,
            make_continuation(mult, localities[0],
                make_continuation(inc, make_continuation())),
            localities[0], &cb, 42);
        HPX_TEST_EQ(f.get(), 87);

        f = hpx::async_continue_cb(inc,
            make_continuation(
                mult, localities[0], make_continuation(inc, localities[0])),
            localities[0], &cb, 42);
        HPX_TEST_EQ(f.get(), 87);

        f = hpx::async_continue_cb(inc,
            make_continuation(mult, localities[0],
                make_continuation(inc, localities[0], make_continuation())),
            localities[0], &cb, 42);
        HPX_TEST_EQ(f.get(), 87);

        f = hpx::async_continue_cb(
            inc, make_continuation(mult), localities[0], &cb, 42);
        HPX_TEST_EQ(f.get(), 86);

        f = hpx::async_continue_cb(inc,
            make_continuation(mult, make_continuation()), localities[0], &cb,
            42);
        HPX_TEST_EQ(f.get(), 86);

        // The callback should have been called 8 times. wait for a short period
        // of time, to allow it for it to be fully executed
        hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
        HPX_TEST_EQ(callback_called.load(), 8);
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
#endif
