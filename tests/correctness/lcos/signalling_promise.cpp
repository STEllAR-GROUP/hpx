//  Copyright (c) 2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/include/lcos.hpp>

#include <boost/detail/lightweight_test.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

///////////////////////////////////////////////////////////////////////////////
int test()
{
    return 42;
}
typedef hpx::actions::plain_result_action0<int, test> test_action;

HPX_REGISTER_PLAIN_ACTION(test_action);

///////////////////////////////////////////////////////////////////////////////
int test_error()
{
    HPX_THROW_EXCEPTION(hpx::not_implemented, "test_error",
        "throwing test exception");
    return 42;
}
typedef hpx::actions::plain_result_action0<int, test_error> test_error_action;

HPX_REGISTER_PLAIN_ACTION(test_error_action);

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    using hpx::lcos::promise;
    using hpx::lcos::async_callback;

    {
        bool cb_called = false;

        promise<int> p = async_callback<test_action>(
            [&](int const& i) {                       // data callback
                cb_called = true;
                BOOST_TEST(i == 42);
            },
            hpx::find_here()
        );

        BOOST_TEST(p.get() == 42);
        BOOST_TEST(cb_called);
    }


    {
        std::string error_msg = "throwing test exception: HPX(not_implemented)";
        bool data_cb_called = false;
        bool error_cb_called = false;

        promise<int> p = async_callback<test_error_action>(
            [&](int const& i) {                       // data callback
                data_cb_called = true;
            },
            [&](boost::exception_ptr const& be) {     // error callback
                error_cb_called = true;
                std::string what_msg;

                try {
                    boost::rethrow_exception(be);
                }
                catch (std::exception const& e) {
                    what_msg = e.what();
                }

                BOOST_TEST(what_msg == error_msg);
            },
            hpx::find_here()
        );

        std::string what_msg;
        bool exception_caught = false;

        try {
            p.get();      // throws
        }
        catch (std::exception const& e) {
            exception_caught = true;
            what_msg = e.what();
        }

        BOOST_TEST(what_msg == error_msg);
        BOOST_TEST(exception_caught);
        BOOST_TEST(!data_cb_called);
        BOOST_TEST(error_cb_called);
    }

    hpx::finalize();
    return boost::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}

