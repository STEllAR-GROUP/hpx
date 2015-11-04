//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2012      Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  http://gcc.gnu.org/bugzilla/show_bug.cgi?id=52072

#include <hpx/hpx_init.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
int test()
{
    return 42;
}
HPX_PLAIN_ACTION(test, test_action);

///////////////////////////////////////////////////////////////////////////////
int future_callback(hpx::lcos::future<int> p)
{
    HPX_TEST(p.has_value());
    int result = p.get();
    HPX_TEST_EQ(result, 42);
    return result;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    using hpx::lcos::future;
    using hpx::async;

    {
        test_action do_test;

        future<int> f = async(do_test, hpx::find_here());
        future<int> p = f.then(hpx::util::bind(future_callback,
            hpx::util::placeholders::_1));

        HPX_TEST_EQ(p.get(), 42);
    }

    hpx::finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    boost::program_options::options_description cmdline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    return hpx::init(cmdline, argc, argv);
}

