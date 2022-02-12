//  Copyright (c) 2022 Deepak Suresh
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <hpx/local/algorithm.hpp>
#include <hpx/local/execution.hpp>
#include <hpx/local/init.hpp>
#include <hpx/modules/testing.hpp>

#include "hpx/hpx.hpp"
#include "hpx/hpx_init.hpp"

int hpx_main(int argc, char *argv[])
{
    int start = 7;
    int end = 3;

    hpx::for_loop(hpx::execution::seq, start, end, [&](int){
        std::cout << "loop running \n";
    });    

    return hpx::finalize();
}

int main(int argc, char* argv[])
{   
    hpx::init_params init_args;

    return hpx::init(argc, argv, init_args);
}
