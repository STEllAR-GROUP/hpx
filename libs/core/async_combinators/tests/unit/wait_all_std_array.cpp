//  Copyright (c) 2017-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/future.hpp>
#include <hpx/local/init.hpp>
#include <hpx/local/thread.hpp>
#include <hpx/modules/testing.hpp>

#include <array>
#include <chrono>
#include <stdexcept>

int make_int_slowly()
{
    hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
    return 42;
}

hpx::future<int> make_future()
{
    hpx::lcos::local::packaged_task<int()> task(make_int_slowly);
    return task.get_future();
}

void test_wait_all()
{
    {
        std::array<hpx::future<int>, 2> future_array;
        future_array[0] = make_future();
        future_array[1] = make_future();

        hpx::wait_all_nothrow(future_array);

        for (auto& f : future_array)
        {
            HPX_TEST(f.is_ready());
        }
    }
    {
        std::array<hpx::future<int>, 2> future_array;
        future_array[0] = make_future();
        future_array[1] =
            hpx::make_exceptional_future<int>(std::runtime_error(""));

        bool caught_exception = false;
        try
        {
            hpx::wait_all_nothrow(future_array);

            for (auto& f : future_array)
            {
                HPX_TEST(f.is_ready());
            }
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
            hpx::wait_all(future_array);
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

void test_wait_all_n()
{
    {
        std::array<hpx::future<int>, 2> future_array;
        future_array[0] = make_future();
        future_array[1] = make_future();

        hpx::wait_all_n_nothrow(future_array.begin(), 2);

        for (auto& f : future_array)
        {
            HPX_TEST(f.is_ready());
        }
    }
    {
        std::array<hpx::future<int>, 2> future_array;
        future_array[0] = make_future();
        future_array[1] =
            hpx::make_exceptional_future<int>(std::runtime_error(""));

        bool caught_exception = false;
        try
        {
            hpx::wait_all_n_nothrow(future_array.begin(), 2);

            for (auto& f : future_array)
            {
                HPX_TEST(f.is_ready());
            }
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
            hpx::wait_all_n(future_array.begin(), 2);
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
    test_wait_all();
    test_wait_all_n();
    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::local::init(hpx_main, argc, argv), 0);
    return hpx::util::report_errors();
}
