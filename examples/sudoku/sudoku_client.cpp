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
    hpx::naming::id_type locality_ = hpx::find_here();

    std::list<sudoku::board> b;
    sudoku::board new_board = hpx::new_<sudoku::board>(locality_);

    new_board.init_board(default_size);
    hpx::cout << "Initial state of the board:" << hpx::endl;
    std::vector<boost::uint8_t> board_config = new_board.access_board();
    for(boost::uint8_t r=0;r<9;r++){
        for(boost::uint8_t c=0;c<9;c++)
            if(board_config[r*9+c] == 0)
                hpx::cout << "_" << " " << hpx::flush;
            else
                hpx::cout << unsigned(board_config[r*9+c]) << " " << hpx::flush;
        hpx::cout << hpx::endl;
    }
    hpx::cout << hpx::endl;

    std::vector<boost::uint8_t> final_board = new_board.solve_board(default_size, 0);

    bool no_solution = false;
    for(std::size_t r=0;r<9;r++){
        for(std::size_t c=0;c<9;c++)
            if(final_board[9*r+c] == 0){
                no_solution = true;
                break;
            }
    }

    if(no_solution){
        hpx::cout << "The given sudoku puzzle has no solution" << hpx::endl;
    }
    else{
        hpx::cout << "Completed puzzle:" << hpx::endl;
        for(std::size_t r=0;r<9;r++){
            for(std::size_t c=0;c<9;c++)
                hpx::cout << final_board[r*9+c] << " " << hpx::flush;
            hpx::cout << hpx::endl;
        }
    }

    hpx::finalize();
    return 0;
}


int main(int argc, char* argv[])
{
    boost::program_options::options_description
        desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    return hpx::init(desc_commandline, argc, argv);
}
