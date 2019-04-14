//  Copyright (c) 2019 Thomas Heller
//  Copyright (c) 2017 Agustin Berge
//  Copyright (c) 2017 Google
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/exception.hpp>
#include <hpx/throw_exception.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <exception>
#include <thread>

void throw_always()
{
    HPX_THROW_EXCEPTION(hpx::no_success, "throw_always", "simulated error");
}

std::exception_ptr test_transport()
{
    std::exception_ptr ptr;
    try
    {
        throw_always();
    }
    catch (...)
    {
        ptr = std::current_exception();
        HPX_TEST_EQ(
            hpx::get_error_what(ptr), "simulated error: HPX(no_success)");
        HPX_TEST_EQ(hpx::get_error_function_name(ptr), "throw_always");
    }

    return ptr;
}

int main()
{
    bool exception_caught = false;

    try
    {
        throw_always();
    }
    catch (...)
    {
        exception_caught = true;
        auto ptr = std::current_exception();
        HPX_TEST_EQ(
            hpx::get_error_what(ptr), "simulated error: HPX(no_success)");
        HPX_TEST_EQ(hpx::get_error_function_name(ptr), "throw_always");
    }
    HPX_TEST(exception_caught);

    exception_caught = false;
    try
    {
        throw_always();
    }
    catch (hpx::exception& e)
    {
        exception_caught = true;
        HPX_TEST_EQ(hpx::get_error_what(e), "simulated error: HPX(no_success)");
        HPX_TEST_EQ(hpx::get_error_function_name(e), "throw_always");
    }
    HPX_TEST(exception_caught);

    exception_caught = false;
    try
    {
        throw_always();
    }
    catch (hpx::exception_info& e)
    {
        exception_caught = true;
        HPX_TEST_EQ(hpx::get_error_what(e), "simulated error: HPX(no_success)");
        HPX_TEST_EQ(hpx::get_error_function_name(e), "throw_always");
    }
    HPX_TEST(exception_caught);

    {
        std::exception_ptr ptr = test_transport();
        HPX_TEST(ptr);
        HPX_TEST_EQ(
            hpx::get_error_what(ptr), "simulated error: HPX(no_success)");
        HPX_TEST_EQ(hpx::get_error_function_name(ptr), "throw_always");
    }

    {
        std::exception_ptr ptr;
        std::thread t([&ptr]() { ptr = test_transport(); });
        t.join();
        HPX_TEST(ptr);
        HPX_TEST_EQ(
            hpx::get_error_what(ptr), "simulated error: HPX(no_success)");
        HPX_TEST_EQ(hpx::get_error_function_name(ptr), "throw_always");
    }

    return hpx::util::report_errors();
}
