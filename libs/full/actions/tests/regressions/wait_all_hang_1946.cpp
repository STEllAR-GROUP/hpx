//  Copyright 2016 (c) Jan-Tobias Sohns
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #1946:
// Hang in wait_all() in distributed run

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/iostream.hpp>
#include <hpx/serialization/serialize.hpp>

#include <cstddef>
#include <iostream>
#include <list>
#include <math.h>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

void out(std::vector<unsigned int> const&)
{
    hpx::cout << "out called " << hpx::find_here() << std::endl;
}
HPX_PLAIN_ACTION(out, out_action);

int main(int argc, char* argv[])
{
    // Initialize and run HPX.
    return hpx::init(argc, argv);
}

int hpx_main()
{
    // find locality info
    std::vector<hpx::naming::id_type> locs = hpx::find_all_localities();

    // create data
    std::vector<unsigned int> vec;
    for (unsigned long j = 0; j < 300000; j++)
    {
        vec.push_back(1);
    }
    // send out data
    for (unsigned int j = 0; j < 8; j++)
    {
        std::vector<hpx::future<void>> fut1;
        for (std::size_t i = 0; i < locs.size(); i++)
        {
            typedef out_action out_act;
            fut1.push_back(hpx::async<out_act>(locs.at(i), vec));
            hpx::cout << "Scheduled out to " << i + 1 << std::endl;
        }
        wait_all(fut1);
        hpx::cout << j + 1 << ". round finished " << std::endl;
    }
    hpx::cout << "program finished!!!" << std::endl;
    return hpx::finalize();
}
#endif
