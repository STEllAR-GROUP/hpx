//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2012      Bryce Adelstein-Lelbach 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
hpx::future<void> test_async0(/* ... */)
{
    // if (... some logic ...)
        return hpx::make_error_future<void>(HPX_GET_EXCEPTION(
            hpx::unknown_error, "test", "Something horrible happened"));
    // else
    //    return hpx::async
}

void test1()
{
    HPX_THROW_EXCEPTION(
        hpx::unknown_error, "test", "Something horrible happened");
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    using hpx::lcos::future;
    using hpx::async;

    {
        bool caught_exception = false;

        try
        {
            {
                future<void> f = test_async0();
            }

        }

        catch (hpx::exception&) 
        {
            caught_exception = true;
        }

        HPX_TEST_MSG(caught_exception, "test_async0 exception wasn't caught");

    }

    {
        bool caught_exception = false;


        try
        {

            {
                future<void> f = async(test1);
            }

        }

        catch (hpx::exception&) 
        {
            caught_exception = true;
        }


        HPX_TEST_MSG(caught_exception, "async(test1) exception wasn't caught");
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

