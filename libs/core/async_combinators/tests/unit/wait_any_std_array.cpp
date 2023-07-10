//  Copyright (c) 2017-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <array>
#include <chrono>
#include <stdexcept>

hpx::future<int> make_future()
{
    return hpx::make_ready_future_after(std::chrono::milliseconds(100), 42);
}

void test_wait_any()
{
    {
        std::array<hpx::future<int>, 2> future_array;
        future_array[0] = make_future();
        future_array[1] = make_future();

        HPX_TEST(!hpx::wait_any_nothrow(future_array));

        int count = 0;
        for (auto& f : future_array)
        {
            if (f.is_ready())
            {
                ++count;
            }
        }
        HPX_TEST_NEQ(count, 0);
    }
    {
        std::array<hpx::future<int>, 2> future_array;
        future_array[0] = make_future();
        future_array[1] =
            hpx::make_exceptional_future<int>(std::runtime_error(""));

        bool caught_exception = false;
        try
        {
            HPX_TEST(hpx::wait_any_nothrow(future_array));

            int count = 0;
            for (auto& f : future_array)
            {
                if (f.is_ready())
                {
                    ++count;
                }
            }
            HPX_TEST_NEQ(count, 0);
        }
        catch (std::runtime_error const&)
        {
            caught_exception = true;
        }
        catch (...)
        {
            HPX_TEST(false);
        }
        HPX_TEST(!caught_exception);
    }
    {
        std::array<hpx::future<int>, 2> future_array;
        future_array[0] = make_future();
        future_array[1] =
            hpx::make_exceptional_future<int>(std::runtime_error(""));

        bool caught_exception = false;
        try
        {
            hpx::wait_any(future_array);
            HPX_TEST(false);
        }
        catch (std::runtime_error const&)
        {
            caught_exception = true;
        }
        catch (...)
        {
            HPX_TEST(false);
        }
        HPX_TEST(caught_exception);
    }
}

void test_wait_any_n()
{
    {
        std::array<hpx::future<int>, 2> future_array;
        future_array[0] = make_future();
        future_array[1] = make_future();

        HPX_TEST(!hpx::wait_any_n_nothrow(future_array.begin(), 2));

        int count = 0;
        for (auto& f : future_array)
        {
            if (f.is_ready())
            {
                ++count;
            }
        }
        HPX_TEST_NEQ(count, 0);
    }
    {
        std::array<hpx::future<int>, 2> future_array;
        future_array[0] = make_future();
        future_array[1] =
            hpx::make_exceptional_future<int>(std::runtime_error(""));

        bool caught_exception = false;
        try
        {
            HPX_TEST(hpx::wait_any_n_nothrow(future_array.begin(), 2));

            int count = 0;
            for (auto& f : future_array)
            {
                if (f.is_ready())
                {
                    ++count;
                }
            }
            HPX_TEST_NEQ(count, 0);
        }
        catch (std::runtime_error const&)
        {
            caught_exception = true;
        }
        catch (...)
        {
            HPX_TEST(false);
        }
        HPX_TEST(!caught_exception);
    }
    {
        std::array<hpx::future<int>, 2> future_array;
        future_array[0] = make_future();
        future_array[1] =
            hpx::make_exceptional_future<int>(std::runtime_error(""));

        bool caught_exception = false;
        try
        {
            hpx::wait_any_n(future_array.begin(), 2);
            HPX_TEST(false);
        }
        catch (std::runtime_error const&)
        {
            caught_exception = true;
        }
        catch (...)
        {
            HPX_TEST(false);
        }
        HPX_TEST(caught_exception);
    }
}

int hpx_main()
{
    test_wait_any();
    test_wait_any_n();
    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::local::init(hpx_main, argc, argv), 0);
    return hpx::util::report_errors();
}
