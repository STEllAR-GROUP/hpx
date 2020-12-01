//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Checking that #582 was fixed

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>

int hpx_main()
{
    HPX_THROW_EXCEPTION(hpx::invalid_status, "hpx_main", "testing");
    return hpx::finalize();
}

int main(int argc, char** argv)
{
    bool caught_exception = false;
    try
    {
        hpx::init(argc, argv);
    }
    catch (hpx::exception const& e)
    {
        HPX_TEST(e.get_error() == hpx::invalid_status);
        caught_exception = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }
    HPX_TEST(caught_exception);

    return hpx::util::report_errors();
}
