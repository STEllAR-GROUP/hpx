//  Copyright (c) 2016 Hadrian G. (a.k.a. Neolander)
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This compile-only test case verifies that #2035 remains fixed

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <array>

int hpx_main()
{
    std::array<hpx::shared_future<int>, 1> future_array{
        {hpx::make_ready_future(0)}};

    hpx::wait_all(future_array.cbegin(), future_array.cend());

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
