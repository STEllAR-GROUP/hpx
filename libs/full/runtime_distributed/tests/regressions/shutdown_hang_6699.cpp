//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/modules/testing.hpp>

#include <stdexcept>
#include <string>

constexpr auto error_message = "throwing runtime_error";

void throw_exception()
{
    throw std::runtime_error(error_message);
}
HPX_PLAIN_ACTION(throw_exception);

int hpx_main()
{
    hpx::async(throw_exception_action(), hpx::find_here()).get();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    try
    {
        return hpx::init(argc, argv);
    }
    catch (std::exception const& e)
    {
        HPX_TEST_EQ(std::string(e.what()), error_message);
    }
    return hpx::util::report_errors();
}
