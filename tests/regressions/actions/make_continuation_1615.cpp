//  Copyright (c) 2015 Matthias Vill
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Verify that #1615 was properly fixed (hpx::make_continuation requires input
// and output to be the same)

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstdint>
#include <string>

std::int32_t times2(std::int32_t i)
{
    return i * 2;
}
HPX_PLAIN_ACTION(times2);           // defines times2_action

std::string to_string(std::int32_t i)
{
    return std::to_string(i);
}
HPX_PLAIN_ACTION(to_string);    // defines to_string_action

int hpx_main(int argc, char* argv[])
{
    std::string result = hpx::async_continue(
        times2_action(), hpx::make_continuation(to_string_action()),
        hpx::find_here(), 42).get();

    HPX_TEST_EQ(result, std::string("84"));

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
