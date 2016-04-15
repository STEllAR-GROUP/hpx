//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/async.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
boost::int32_t increment(boost::int32_t i)
{
    return i + 1;
}
HPX_PLAIN_ACTION(increment);  // defines increment_action

boost::int32_t increment_with_future(hpx::shared_future<boost::int32_t> fi)
{
    return fi.get() + 1;
}
HPX_PLAIN_ACTION(increment_with_future);

///////////////////////////////////////////////////////////////////////////////
boost::int32_t mult2(boost::int32_t i)
{
    return i * 2;
}
HPX_PLAIN_ACTION(mult2);      // defines mult2_action

///////////////////////////////////////////////////////////////////////////////
boost::atomic<int> callback_called(0);

void cb(boost::system::error_code const& ec,
    hpx::parcelset::parcel const& p)
{
    ++callback_called;
}

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
        HPX_TEST_EQ(callback_called.load(), 1);

        hpx::promise<boost::int32_t> p;
        hpx::shared_future<boost::int32_t> f = p.get_future();

        callback_called.store(0);
        hpx::future<int> f2 = hpx::async_continue_cb(
            inc_f, make_continuation(), hpx::find_here(), &cb, f);

        p.set_value(42);
        HPX_TEST_EQ(f2.get(), 43);
        HPX_TEST_EQ(callback_called.load(), 1);
    }

    // test remotely, if possible, fully equivalent to plain hpx::async
    std::vector<hpx::id_type> localities = hpx::find_remote_localities();
    if (!localities.empty())
    {
        callback_called.store(0);
        hpx::future<int> f1 = hpx::async_continue_cb(
            inc, make_continuation(), localities[0], &cb, 42);
        HPX_TEST_EQ(f1.get(), 43);
        HPX_TEST_EQ(callback_called.load(), 1);

        hpx::promise<boost::int32_t> p;
        hpx::shared_future<boost::int32_t> f = p.get_future();

        callback_called.store(0);
        hpx::future<int> f2 = hpx::async_continue_cb(
            inc_f, make_continuation(), localities[0], &cb, f);

        p.set_value(42);
        HPX_TEST_EQ(f2.get(), 43);
        HPX_TEST_EQ(callback_called.load(), 1);
    }

    // test chaining locally
    {
        callback_called.store(0);
        hpx::future<int> f = hpx::async_continue_cb(
            inc, make_continuation(mult), hpx::find_here(), &cb, 42);
        HPX_TEST_EQ(f.get(), 86);
        HPX_TEST_EQ(callback_called.load(), 1);

        callback_called.store(0);
        f = hpx::async_continue_cb(inc,
            make_continuation(mult, make_continuation()), hpx::find_here(),
            &cb, 42);
        HPX_TEST_EQ(f.get(), 86);
        HPX_TEST_EQ(callback_called.load(), 1);

        callback_called.store(0);
        f = hpx::async_continue_cb(inc,
            make_continuation(mult, make_continuation(inc)), hpx::find_here(),
            &cb, 42);
        HPX_TEST_EQ(f.get(), 87);
        HPX_TEST_EQ(callback_called.load(), 1);

        callback_called.store(0);
        f = hpx::async_continue_cb(inc,
            make_continuation(mult, make_continuation(inc, make_continuation())),
            hpx::find_here(), &cb, 42);
        HPX_TEST_EQ(f.get(), 87);
        HPX_TEST_EQ(callback_called.load(), 1);
    }

    // test chaining remotely, if possible
    if (!localities.empty())
    {
        callback_called.store(0);
        hpx::future<int> f = hpx::async_continue_cb(inc,
            make_continuation(mult, localities[0]), localities[0], &cb, 42);
        HPX_TEST_EQ(f.get(), 86);
        HPX_TEST_EQ(callback_called.load(), 1);

        callback_called.store(0);
        f = hpx::async_continue_cb(inc,
            make_continuation(mult, localities[0], make_continuation()),
            localities[0], &cb, 42);
        HPX_TEST_EQ(f.get(), 86);
        HPX_TEST_EQ(callback_called.load(), 1);

        callback_called.store(0);
        f = hpx::async_continue_cb(inc,
            make_continuation(mult, localities[0],
                make_continuation(inc)), localities[0], &cb, 42);
        HPX_TEST_EQ(f.get(), 87);
        HPX_TEST_EQ(callback_called.load(), 1);

        callback_called.store(0);
        f = hpx::async_continue_cb(inc,
            make_continuation(mult, localities[0],
                make_continuation(inc, make_continuation())), localities[0],
            &cb, 42);
        HPX_TEST_EQ(f.get(), 87);
        HPX_TEST_EQ(callback_called.load(), 1);

        callback_called.store(0);
        f = hpx::async_continue_cb(inc,
            make_continuation(mult, localities[0],
                make_continuation(inc, localities[0])), localities[0], &cb, 42);
        HPX_TEST_EQ(f.get(), 87);
        HPX_TEST_EQ(callback_called.load(), 1);

        callback_called.store(0);
        f = hpx::async_continue_cb(inc,
            make_continuation(mult, localities[0],
                make_continuation(inc, localities[0], make_continuation())),
                localities[0], &cb, 42);
        HPX_TEST_EQ(f.get(), 87);
        HPX_TEST_EQ(callback_called.load(), 1);

        callback_called.store(0);
        f = hpx::async_continue_cb(inc,
            make_continuation(mult), localities[0], &cb, 42);
        HPX_TEST_EQ(f.get(), 86);
        HPX_TEST_EQ(callback_called.load(), 1);

        callback_called.store(0);
        f = hpx::async_continue_cb(inc,
            make_continuation(mult, make_continuation()), localities[0],
            &cb, 42);
        HPX_TEST_EQ(f.get(), 86);
        HPX_TEST_EQ(callback_called.load(), 1);
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

