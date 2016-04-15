//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/parallel_algorithm.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
struct my_executor : hpx::parallel::parallel_executor {};

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[])
{
    using hpx::parallel::for_each;
    using hpx::parallel::par;

    my_executor exec;

    std::vector<int> v(100);

    for_each(par.on(exec), v.begin(), v.end(), [](int x){ });

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}
