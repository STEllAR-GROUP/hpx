//  Copyright (c) 2022 Deepak Suresh
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// fixes #5735: hpx::for_loop executes without checking start and end bounds

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/init.hpp>

#include <iostream>

int hpx_main()
{
    int start = 7;
    int end = 3;

    hpx::experimental::for_loop(hpx::execution::par, start, end,
        [&](int) { std::cout << "loop running \n"; });

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::local::init(hpx_main, argc, argv);
}
