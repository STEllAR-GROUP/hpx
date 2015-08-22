//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/local_lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#define NUM_VALUES 100

void send_values(hpx::lcos::local::channel<int> &c)
{
    for (int i = 0; i != NUM_VALUES; ++i)
    {
        c.set_value(i);
    }
}

void receive_values(hpx::lcos::local::channel<int> &c)
{
    for (int i = 0; i != NUM_VALUES; ++i)
    {
        hpx::future<int> f = c.get_future();
        HPX_TEST_EQ(i, f.get());
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    {
        hpx::lcos::local::channel<int> c;

        hpx::future<void> g = hpx::async(&send_values, boost::ref(c));
        receive_values(c);

        g.get();
    }

    {
        hpx::lcos::local::channel<int> c;

        hpx::future<void> g = hpx::async(&receive_values, boost::ref(c));
        send_values(c);

        g.get();
    }

    // two calls to get_future() without waiting for the first one to become
    // ready should throw
    {
        hpx::lcos::local::channel<int> c;
        hpx::future<int> f1 = c.get_future();

        bool exception_thrown = false;
        try {
            hpx::future<int> f2 = c.get_future();
            HPX_TEST(false);
        }
        catch (hpx::exception const&) {
            exception_thrown = true;
        }
        HPX_TEST(exception_thrown);
    }

    HPX_TEST_EQ(hpx::finalize(), 0);
    return 0;
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
