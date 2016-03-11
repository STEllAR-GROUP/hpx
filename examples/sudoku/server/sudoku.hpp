//  Copyright (c) 2016 Satyaki Upadhyay
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SUDOKU_EXAMPLE_SERVER)
#define HPX_SUDOKU_EXAMPLE_SERVER

#include <iostream>

#include <hpx/include/async.hpp>
#include <hpx/include/util.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/include/components.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/serialization/serialize.hpp>

namespace sudoku
{
    typedef std::vector<std::size_t> list_type;

namespace server
{
    class board
        : public hpx::components::component_base<board>
    {
    private:
        list_type list_;
        std::size_t level_;
        std::size_t size_;
        std::size_t count_;

        // here board is a component

        friend class hpx::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version){
            ar & size_;
            ar & list_;
            ar & level_;
            ar & count_;
        }

    public:

        board():list_(0), level_(0), size_(0)
        {}

        board(list_type const& list, std::size_t size, std::size_t level)
            : list_(list), level_(level), size_(size)
        {}

        ~board(){}



        void init_board(std::size_t size)
        {
            //initialize the starting state of the puzzle
            //0 denotes empty positions
            //non-zero values denote already-filled cells
            std::size_t i = 0;
            while(i < size){
                list_.push_back(0);
                i++;
            }

            list_.at(1) = 2;
            list_.at(5) = 4;
            list_.at(6) = 3;
            list_.at(9) = 9;
            list_.at(13) = 2;
            list_.at(17) = 8;
            list_.at(21) = 6;
            list_.at(23) = 9;
            list_.at(25) = 5;
            list_.at(35) = 1;
            list_.at(37) = 7;
            list_.at(38) = 2;
            list_.at(39) = 5;
            list_.at(41) = 3;
            list_.at(42) = 6;
            list_.at(43) = 8;
            list_.at(45) = 6;
            list_.at(55) = 8;
            list_.at(57) = 2;
            list_.at(59) = 5;
            list_.at(63) = 1;
            list_.at(67) = 9;
            list_.at(71) = 3;
            list_.at(74) = 9;
            list_.at(75) = 8;
            list_.at(79) = 6;
        }

        bool check_board(list_type const& list, std::size_t level)
        {
            std::size_t x = level/9, y = level%9;

            //no equal value in same row
            for(std::size_t col=0;col<9;col++)
                if((x*9+col) != level && list_.at(x*9+col) == list_.at(level))
                    return false;

            //no equal value in same column
            for(std::size_t row=0;row<9;row++)
                if((row*9+y) != level && list_.at(row*9+y) == list_.at(level))
                    return false;

            //no equal value in same mini-grid
            std::size_t minigrid_row = x/3, minigrid_col = y/3;
            for(std::size_t r=3*minigrid_row;r<3*(minigrid_row+1);r++)
            for(std::size_t c=3*minigrid_col;c<3*(minigrid_col+1);c++)
                if((r*9+c) != level && list_.at(r*9+c) == list_.at(level))
                    return false;
            return true;
        }

        list_type access_board()
        {
            return list_;
        }

        void update_board(std::size_t pos, std::size_t val)
        {
            list_.at(pos) = val;
        }

        void clear_board()
        {
            board::list_.clear();
        }

        std::size_t solve_board(list_type const& list, std::size_t size,
            std::size_t level)
        {
            if(level == size){
                std::cout << "Completed puzzle:" << std::endl;
                for(std::size_t r=0;r<9;r++){
                    for(std::size_t c=0;c<9;c++)
                        std::cout << list_[r*9+c] << " " << std::flush;
                    std::cout << std::endl;
                }
                return 1;
            }

            board b(list, size, level);
            if(level == 0){
                if(!b.check_board( b.access_board(), level))
                    return 0;
                return solve_board( b.access_board(),
                                    size, 1);
            }

            for(std::size_t i = 0; i < size; ++i){
                b.update_board(level,i);
                if(b.check_board( b.access_board(), level)){
                   if(solve_board( b.access_board(),
                                        size, level+1) > 0)
                        return 1;
                }
            }

            return 0;
        }

        HPX_DEFINE_COMPONENT_ACTION(board, init_board, init_action);
        HPX_DEFINE_COMPONENT_ACTION(board, access_board, access_action);
        HPX_DEFINE_COMPONENT_ACTION(board, update_board, update_action);
        HPX_DEFINE_COMPONENT_ACTION(board, check_board, check_action);
        HPX_DEFINE_COMPONENT_ACTION(board, solve_board, solve_action);
        HPX_DEFINE_COMPONENT_ACTION(board, clear_board, clear_action);
    };

}}

// Declaration of serialization support for the board actions

HPX_REGISTER_ACTION_DECLARATION(
    sudoku::server::board::init_action,
    board_init_action);

HPX_REGISTER_ACTION_DECLARATION(
    sudoku::server::board::check_action,
    board_check_action);

HPX_REGISTER_ACTION_DECLARATION(
    sudoku::server::board::access_action,
    board_access_action);

HPX_REGISTER_ACTION_DECLARATION(
    sudoku::server::board::update_action,
    board_update_action);

HPX_REGISTER_ACTION_DECLARATION(
    sudoku::server::board::solve_action,
    board_solve_action);

HPX_REGISTER_ACTION_DECLARATION(
    sudoku::server::board::clear_action,
    board_clear_action);

HPX_REGISTER_ACTION_DECLARATION(
    hpx::lcos::base_lco_with_value<sudoku::list_type>::set_value_action,
    set_value_action_vector_std_size_t);

HPX_REGISTER_ACTION_DECLARATION(
    hpx::lcos::base_lco_with_value<sudoku::list_type>::get_value_action,
    get_value_action_vector_std_size_t);


#endif // HPX_SUDOKU_EXAMPLE_SERVER

