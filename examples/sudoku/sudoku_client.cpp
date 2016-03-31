//  Copyright (c) 2016 Satyaki Upadhyay
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/util.hpp>

#include <boost/cstdint.hpp>
#include "sudoku.hpp"

int hpx_main(boost::program_options::variables_map&)
{
    const std::size_t default_size = 9*9;
    std::size_t soln_count_total = 0, i = 0;
    hpx::naming::id_type locality_ = hpx::find_here();

    std::list<sudoku::board> b;
    sudoku::board new_board;
    while(i != 9){
        b.push_back(new_board);
        ++i;
    }

    new_board.init_board(default_size);
    // hpx::cout << "Initial state of the board:" << hpx::endl;
    // std::vector<boost::uint8_t> board_config = new_board.access_board();
    // for(boost::uint8_t r=0;r<9;r++){
    //     for(boost::uint8_t c=0;c<9;c++)
    //         if(board_config[r*9+c] == 0)
    //             hpx::cout << "_" << " " << hpx::flush;
    //         else
    //             hpx::cout << board_config[r*9+c] << " " << hpx::flush;
    //     hpx::cout << hpx::endl;
    // }
    // hpx::cout << hpx::endl;

    i = 1;
    for(std::list<sudoku::board>::iterator iter = b.begin();
                iter != b.end(); ++iter){
        iter->create(locality_);
        iter->init_board(default_size);
        soln_count_total+= iter->solve_board(iter->access_board(),
                                                default_size, 0);
        if(soln_count_total > 0)    break;
        ++i;
    }
    b.clear();

    if(soln_count_total == 0)
        hpx::cout << "The given sudoku puzzle has no solution" << hpx::endl;

    hpx::finalize();
    return 0;
}


int main(int argc, char* argv[])
{
    boost::program_options::options_description
        desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    return hpx::init(desc_commandline, argc, argv);
}
