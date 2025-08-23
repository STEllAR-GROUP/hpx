//  Copyright (c) 2007-2017 Hartmut Kaiser, Richard D Guidry Jr.
//  Copyright (c) 2011 Vinay C Amatya
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Parts of this nqueen_client.cpp has been taken from the accumulator example
//  by Hartmut Kaiser.

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/util.hpp>

#include <cstddef>
#include <iostream>
#include <list>
#include <string>
#include <vector>

#include "nqueen.hpp"

int hpx_main(hpx::program_options::variables_map&)
{
    const std::size_t default_size = 8;

    std::size_t soln_count_total = 0;

    hpx::id_type locality_ = hpx::find_here();

    std::cout << "Enter size of board. Default size is 8." << std::endl;
    std::cout << "Command Options: size[value] | default | print | quit"
              << std::endl;
    std::string cmd;
    std::cin >> cmd;

    while (std::cin.good())
    {
        if (cmd == "size")
        {
            soln_count_total = 0;
            std::string arg;
            std::cin >> arg;
            std::size_t sz = hpx::util::from_string<std::size_t>(arg);

            std::size_t i = 0;
            std::list<nqueen::board> b;
            nqueen::board bi = hpx::new_<nqueen::board>(locality_);
            while (i != sz)
            {
                b.push_back(bi);
                ++i;
            }

            i = 0;
            for (std::list<nqueen::board>::iterator iter = b.begin();
                iter != b.end(); ++iter)
            {
                iter->init_board(sz);
                soln_count_total +=
                    iter->solve_board(iter->access_board(), sz, 0, i);
                ++i;
            }
            std::cout << "soln_count:" << soln_count_total << std::endl;
            b.clear();
        }
        else if (cmd == "default")
        {
            soln_count_total = 0;
            nqueen::board a = hpx::new_<nqueen::board>(locality_);
            std::size_t i = 0;
            std::vector<nqueen::board> b;
            while (i != default_size)
            {
                b.push_back(a);
                ++i;
            }
            i = 0;
            for (std::vector<nqueen::board>::iterator iter = b.begin();
                iter != b.end(); ++iter)
            {
                iter->init_board(default_size);
                soln_count_total +=
                    iter->solve_board(iter->access_board(), default_size, 0, i);
                ++i;
            }
            std::cout << "soln_count:" << soln_count_total << std::endl;
            b.clear();
        }
        else if (cmd == "print")
        {
            std::cout << "soln_count : " << soln_count_total << std::endl;
        }
        else if (cmd == "quit")
        {
            //std::cout << "soln_count : " << soln_count_total << std::endl;
            break;
        }
        else
        {
            std::cout << "Invalid Command." << std::endl;
            std::cout << "Options: size[value] | default | print "
                      << "| quit" << std::endl;
        }
        std::cin >> cmd;
    }

    hpx::finalize();

    return 0;
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}
#endif
