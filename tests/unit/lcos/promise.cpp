//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/ref.hpp>

///////////////////////////////////////////////////////////////////////////////
int test()
{
    return 42;
}
HPX_PLAIN_ACTION(test, test_action);

///////////////////////////////////////////////////////////////////////////////
int test_error()
{
    HPX_THROW_EXCEPTION(hpx::not_implemented, "test_error",
        "throwing test exception");
    return 42;
}
HPX_PLAIN_ACTION(test_error, test_error_action);

char const* const error_msg = "throwing test exception: HPX(not_implemented)";

///////////////////////////////////////////////////////////////////////////////
int future_callback(
    bool& data_cb_called
  , bool& error_cb_called
  , hpx::lcos::shared_future<int> f
    )
{
    if (f.has_value()) {
        data_cb_called = true;
        int result = f.get();
        HPX_TEST_EQ(result, 42);
        return result;
    }

    error_cb_called = true;
    HPX_TEST(f.has_exception());

    std::string what_msg;

    try {
        f.get();          // should rethrow
        HPX_TEST(false);
    }
    catch (std::exception const& e) {
        what_msg = e.what();
        HPX_TEST(true);
    }

    HPX_TEST_EQ(what_msg, error_msg);
    return -1;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map&)
{
    {
        hpx::lcos::future<int> p = hpx::async<test_action>(hpx::find_here());
        HPX_TEST_EQ(p.get(), 42);
    }

    {
        test_action do_test;
        hpx::lcos::future<int> p = hpx::async(do_test, hpx::find_here());
        HPX_TEST_EQ(p.get(), 42);
    }

    {
        bool data_cb_called = false;
        bool error_cb_called = false;

        hpx::lcos::shared_future<int> f = hpx::async<test_action>(hpx::find_here());
        hpx::lcos::future<int> p = f.then(hpx::util::bind(future_callback,
            boost::ref(data_cb_called), boost::ref(error_cb_called),
            hpx::util::placeholders::_1));

        HPX_TEST_EQ(p.get(), 42);
        HPX_TEST(data_cb_called);
        HPX_TEST(!error_cb_called);
    }

    {
        bool data_cb_called = false;
        bool error_cb_called = false;
        test_action do_test;

        hpx::lcos::shared_future<int> f = hpx::async(do_test, hpx::find_here());

        hpx::lcos::future<int> p = f.then(
                hpx::util::bind(future_callback, boost::ref(data_cb_called),
                    boost::ref(error_cb_called), hpx::util::placeholders::_1));

        HPX_TEST_EQ(p.get(), 42);
        HPX_TEST(data_cb_called);
        HPX_TEST(!error_cb_called);
    }

    {
        hpx::lcos::future<int> p = hpx::async<test_error_action>(hpx::find_here());

        std::string what_msg;

        try {
            p.get();      // throws
            HPX_TEST(false);
        }
        catch (std::exception const& e) {
            HPX_TEST(true);
            what_msg = e.what();
        }

        HPX_TEST_EQ(what_msg, error_msg);
    }

    {
        test_error_action do_error;
        hpx::lcos::future<int> p = hpx::async(do_error, hpx::find_here());

        std::string what_msg;

        try {
            p.get();      // throws
            HPX_TEST(false);
        }
        catch (std::exception const& e) {
            HPX_TEST(true);
            what_msg = e.what();
        }

        HPX_TEST_EQ(what_msg, error_msg);
    }

    {
        bool data_cb_called = false;
        bool error_cb_called = false;

        hpx::lcos::shared_future<int> f = hpx::async<test_error_action>(hpx::find_here());
        hpx::lcos::future<int> p = f.then(hpx::util::bind(future_callback,
            boost::ref(data_cb_called), boost::ref(error_cb_called),
            hpx::util::placeholders::_1));

        std::string what_msg;

        try {
            f.get();      // throws
            p.get();
            HPX_TEST(false);
        }
        catch (std::exception const& e) {
            HPX_TEST(true);
            what_msg = e.what();
        }

        HPX_TEST_EQ(what_msg, error_msg);
        HPX_TEST(!data_cb_called);
        HPX_TEST(error_cb_called);
    }

    {
        bool data_cb_called = false;
        bool error_cb_called = false;
        test_error_action do_test_error;

        hpx::lcos::shared_future<int> f = hpx::async(do_test_error, hpx::find_here());

        hpx::lcos::future<int> p = f.then(
                hpx::util::bind(future_callback, boost::ref(data_cb_called),
                    boost::ref(error_cb_called), hpx::util::placeholders::_1));

        std::string what_msg;

        try {
            f.get();      // throws
            p.get();
            HPX_TEST(false);
        }
        catch (std::exception const& e) {
            HPX_TEST(true);
            what_msg = e.what();
        }

        HPX_TEST_EQ(what_msg, error_msg);
        HPX_TEST(!data_cb_called);
        HPX_TEST(error_cb_called);
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

    // We force this test to use several threads by default.
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency());

    // Initialize and run HPX
    return hpx::init(cmdline, argc, argv, cfg);
}

