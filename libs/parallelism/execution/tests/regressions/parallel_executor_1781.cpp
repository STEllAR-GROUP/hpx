//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[])
{
    std::vector<int> v(100);

    {
        hpx::parallel::execution::static_chunk_size block(1);
        hpx::parallel::execution::parallel_executor exec;
        hpx::ranges::for_each(
            hpx::parallel::execution::par.on(exec).with(block), v,
            [](int i) {});
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}
