//  Copyright (c) 2017-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <chrono>
#include <stdexcept>
#include <vector>

hpx::future<int> make_future()
{
    return hpx::make_ready_future_after(std::chrono::milliseconds(100), 42);
}

void test_wait_all()
{
    {
        auto f1 = make_future();

        HPX_TEST(!hpx::wait_all_nothrow(f1));

        HPX_TEST(f1.is_ready());
    }
    {
        std::vector<hpx::future<int>> future_array;
        future_array.push_back(make_future());
        future_array.push_back(make_future());

        HPX_TEST(!hpx::wait_all_nothrow(future_array));

        for (auto& f : future_array)
        {
            HPX_TEST(f.is_ready());
        }
    }
    {
        auto f1 = make_future();
        auto f2 = make_future();

        HPX_TEST(!hpx::wait_all_nothrow(f1, f2));

        HPX_TEST(f1.is_ready());
        HPX_TEST(f2.is_ready());
    }
    {
        std::vector<hpx::future<int>> future_array;
        future_array.push_back(make_future());
        future_array.push_back(
            hpx::make_exceptional_future<int>(std::runtime_error("")));

        bool caught_exception = false;
        try
        {
            HPX_TEST(hpx::wait_all_nothrow(future_array));

            for (auto& f : future_array)
            {
                HPX_TEST(f.is_ready());
            }
        }
        catch (std::runtime_error const&)
        {
            HPX_TEST(false);
            caught_exception = true;
        }
        catch (...)
        {
            HPX_TEST(false);
        }
        HPX_TEST(!caught_exception);
    }
    {
        std::vector<hpx::future<int>> future_array;
        future_array.push_back(make_future());
        future_array.push_back(
            hpx::make_exceptional_future<int>(std::runtime_error("")));

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
    {
        auto f1 = make_future();
        auto f2 = hpx::make_exceptional_future<int>(std::runtime_error(""));

        bool caught_exception = false;
        try
        {
            hpx::wait_all(f1, f2);
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
    {
        auto f1 = hpx::make_exceptional_future<int>(std::runtime_error(""));

        bool caught_exception = false;
        try
        {
            hpx::wait_all(f1);
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
    {
        auto f1 = make_future();
        auto f2 = hpx::make_exceptional_future<int>(std::runtime_error(""));

        bool caught_exception = false;
        try
        {
            hpx::wait_any(f1, f2);
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
        std::vector<hpx::future<int>> future_array;
        future_array.push_back(make_future());
        future_array.push_back(make_future());

        HPX_TEST(!hpx::wait_all_n_nothrow(
            future_array.begin(), future_array.size()));

        for (auto& f : future_array)
        {
            HPX_TEST(f.is_ready());
        }
    }
    {
        std::vector<hpx::future<int>> future_array;
        future_array.push_back(make_future());
        future_array.push_back(
            hpx::make_exceptional_future<int>(std::runtime_error("")));

        bool caught_exception = false;
        try
        {
            HPX_TEST(hpx::wait_all_n_nothrow(
                future_array.begin(), future_array.size()));

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
        std::vector<hpx::future<int>> future_array;
        future_array.push_back(make_future());
        future_array.push_back(
            hpx::make_exceptional_future<int>(std::runtime_error("")));

        bool caught_exception = false;
        try
        {
            hpx::wait_all_n(future_array.begin(), future_array.size());
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
