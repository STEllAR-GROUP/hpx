//  Copyright (c) 2016-2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
void test_split_invalid_future()
{
    // Test split_nth_continuation
    {
        hpx::future<hpx::tuple<int, int>> f_tuple;
        HPX_TEST(!f_tuple.valid());

        auto result_tuple = hpx::split_future(HPX_MOVE(f_tuple));

        try
        {
            hpx::get<0>(result_tuple).get();
            HPX_TEST(false);
        }
        catch (hpx::exception const& e)
        {
            HPX_TEST_EQ(e.get_error(), hpx::error::no_state);
        }
        catch (...)
        {
            HPX_TEST(false);
        }

        try
        {
            hpx::get<1>(result_tuple).get();
            HPX_TEST(false);
        }
        catch (hpx::exception const& e)
        {
            HPX_TEST_EQ(e.get_error(), hpx::error::no_state);
        }
        catch (...)
        {
            HPX_TEST(false);
        }
    }

#if defined(HPX_DATASTRUCTURES_HAVE_ADAPT_STD_TUPLE)
    {
        hpx::future<std::tuple<int, int>> f_tuple;
        HPX_TEST(!f_tuple.valid());

        auto result_tuple = hpx::split_future(HPX_MOVE(f_tuple));

        try
        {
            std::get<0>(result_tuple).get();
            HPX_TEST(false);
        }
        catch (hpx::exception const& e)
        {
            HPX_TEST_EQ(e.get_error(), hpx::error::no_state);
        }
        catch (...)
        {
            HPX_TEST(false);
        }

        try
        {
            std::get<1>(result_tuple).get();
            HPX_TEST(false);
        }
        catch (hpx::exception const& e)
        {
            HPX_TEST_EQ(e.get_error(), hpx::error::no_state);
        }
        catch (...)
        {
            HPX_TEST(false);
        }
    }
#endif

    // Test split_continuation
    {
        hpx::future<std::vector<int>> f_vector;
        HPX_TEST(!f_vector.valid());

        auto result_vector = hpx::split_future(HPX_MOVE(f_vector), 2);
        HPX_TEST_EQ(result_vector.size(), std::size_t(2));

        for (auto& f : result_vector)
        {
            try
            {
                f.get();
                HPX_TEST(false);
            }
            catch (hpx::exception const& e)
            {
                HPX_TEST_EQ(e.get_error(), hpx::error::no_state);
            }
            catch (...)
            {
                HPX_TEST(false);
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_split_invalid_future();

    hpx::local::finalize();
    return hpx::util::report_errors();
}

int main(int argc, char* argv[])
{
    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.cfg = cfg;

    return hpx::local::init(hpx_main, argc, argv, init_args);
}
