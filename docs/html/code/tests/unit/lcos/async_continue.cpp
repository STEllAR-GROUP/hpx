//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/async.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
boost::int32_t increment(boost::int32_t i)
{
    return i + 1;
}
HPX_PLAIN_ACTION(increment);  // defines increment_action

///////////////////////////////////////////////////////////////////////////////
boost::int32_t mult2(boost::int32_t i)
{
    return i * 2;
}
HPX_PLAIN_ACTION(mult2);      // defines mult2_action

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    using hpx::make_continuation;

    increment_action inc;
    mult2_action mult;

    // test locally, fully equivalent to plain hpx::async
    {
        hpx::future<int> f =
            hpx::async_continue(inc, hpx::find_here(), 42, make_continuation());
        HPX_TEST_EQ(f.get(), 43);
    }

    // test remotely, if possible, fully equivalent to plain hpx::async
    std::vector<hpx::id_type> localities = hpx::find_remote_localities();
    if (!localities.empty())
    {
        hpx::future<int> f =
            hpx::async_continue(inc, localities[0], 42, make_continuation());
        HPX_TEST_EQ(f.get(), 43);
    }

    // test chaining locally
    {
        hpx::future<int> f = hpx::async_continue(inc, hpx::find_here(), 42,
            make_continuation(mult));
        HPX_TEST_EQ(f.get(), 86);

        f = hpx::async_continue(inc, hpx::find_here(), 42,
            make_continuation(mult, make_continuation()));
        HPX_TEST_EQ(f.get(), 86);

        f = hpx::async_continue(inc, hpx::find_here(), 42,
            make_continuation(mult, make_continuation(inc)));
        HPX_TEST_EQ(f.get(), 87);

        f = hpx::async_continue(inc, hpx::find_here(), 42,
            make_continuation(mult, make_continuation(inc, make_continuation())));
        HPX_TEST_EQ(f.get(), 87);
    }

    // test chaining remotely, if possible
    if (!localities.empty())
    {
        hpx::future<int> f = hpx::async_continue(inc, localities[0], 42,
            make_continuation(mult, localities[0]));
        HPX_TEST_EQ(f.get(), 86);

        f = hpx::async_continue(inc, localities[0], 42,
            make_continuation(mult, localities[0], make_continuation()));
        HPX_TEST_EQ(f.get(), 86);

        f = hpx::async_continue(inc, localities[0], 42,
            make_continuation(mult, localities[0],
                make_continuation(inc)));
        HPX_TEST_EQ(f.get(), 87);

        f = hpx::async_continue(inc, localities[0], 42,
            make_continuation(mult, localities[0],
                make_continuation(inc, make_continuation())));
        HPX_TEST_EQ(f.get(), 87);

        f = hpx::async_continue(inc, localities[0], 42,
            make_continuation(mult, localities[0],
                make_continuation(inc, localities[0])));
        HPX_TEST_EQ(f.get(), 87);

        f = hpx::async_continue(inc, localities[0], 42,
            make_continuation(mult, localities[0],
                make_continuation(inc, localities[0], make_continuation())));
        HPX_TEST_EQ(f.get(), 87);

        f = hpx::async_continue(inc, localities[0], 42,
            make_continuation(mult));
        HPX_TEST_EQ(f.get(), 86);

        f = hpx::async_continue(inc, localities[0], 42,
            make_continuation(mult, make_continuation()));
        HPX_TEST_EQ(f.get(), 86);
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

