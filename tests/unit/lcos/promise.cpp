//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/modules/testing.hpp>

#include <functional>
#include <string>
#include <vector>

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
    bool caught_exception = false;

    try {
        f.get();          // should rethrow
        HPX_TEST(false);
    }
    catch (std::exception const& e) {
        what_msg = e.what();
        caught_exception = true;
    }
    catch (...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST_EQ(what_msg, error_msg);
    return -1;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map&)
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
            std::ref(data_cb_called), std::ref(error_cb_called),
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
                hpx::util::bind(future_callback, std::ref(data_cb_called),
                    std::ref(error_cb_called), hpx::util::placeholders::_1));

        HPX_TEST_EQ(p.get(), 42);
        HPX_TEST(data_cb_called);
        HPX_TEST(!error_cb_called);
    }

    {
        hpx::lcos::future<int> p =
            hpx::async<test_error_action>(hpx::find_here());

        std::string what_msg;
        bool caught_exception = false;

        try {
            p.get();      // throws
            HPX_TEST(false);
        }
        catch (std::exception const& e) {
            what_msg = e.what();
            caught_exception = true;
        }
        catch (...) {
            HPX_TEST(false);
        }

        HPX_TEST(caught_exception);
        HPX_TEST_EQ(what_msg, error_msg);
    }

    {
        test_error_action do_error;
        hpx::lcos::future<int> p = hpx::async(do_error, hpx::find_here());

        std::string what_msg;
        bool caught_exception = false;

        try {
            p.get();      // throws
            HPX_TEST(false);
        }
        catch (std::exception const& e) {
            what_msg = e.what();
            caught_exception = true;
        }

        HPX_TEST(caught_exception);
        HPX_TEST_EQ(what_msg, error_msg);
    }

    {
        bool data_cb_called = false;
        bool error_cb_called = false;

        hpx::lcos::shared_future<int> f =
            hpx::async<test_error_action>(hpx::find_here());
        hpx::lcos::future<int> p = f.then(hpx::util::bind(future_callback,
            std::ref(data_cb_called), std::ref(error_cb_called),
            hpx::util::placeholders::_1));

        std::string what_msg;
        bool caught_exception = false;

        try {
            p.get();      // guarantee for callback to have finished executing
            f.get();      // throws
            HPX_TEST(false);
        }
        catch (std::exception const& e) {
            what_msg = e.what();
            caught_exception = true;
        }
        catch (...) {
            HPX_TEST(false);
        }

        HPX_TEST(caught_exception);
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
                hpx::util::bind(future_callback, std::ref(data_cb_called),
                    std::ref(error_cb_called), hpx::util::placeholders::_1));

        std::string what_msg;
        bool caught_exception = false;

        try {
            p.get();      // guarantee for callback to have finished executing
            f.get();      // throws
            HPX_TEST(false);
        }
        catch (std::exception const& e) {
            what_msg = e.what();
            caught_exception = true;
        }
        catch (...) {
            HPX_TEST(false);
        }

        HPX_TEST(caught_exception);
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
    hpx::program_options::options_description cmdline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}
#endif
