//  Copyright (c) 2016 Satyaki Upadhyay
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SUDOKU_EXAMPLE)
#define HPX_SUDOKU_EXAMPLE

#include <hpx/runtime.hpp>
#include <hpx/include/client.hpp>

#include <examples/sudoku/stubs/sudoku.hpp>

namespace sudoku
{
    class board
        : public hpx::components::client_base<board, stubs::board>
    {
        typedef hpx::components::client_base<board, stubs::board> base_type;

    public:
        board()
        {}
        board(hpx::future<hpx::naming::id_type> && gid)
            : base_type(std::move(gid))
        {}

        void init_board(std::size_t size ){
            return this->base_type::init_board(get_id(), size);
        }
        //-------------------------------------------------------

        board_type access_board(){
            return this->base_type::access_board(get_id());
        }

        hpx::lcos::future<board_type> access_board_async(){
            return this->base_type::access_board_async(get_id());
        }
        //------------------------------------------------------

        void update_board(std::size_t level, std::size_t pos){
            return this->base_type::update_board(get_id(), level, pos);
        }
        //-----------------------------------------------------

        bool check_board(board_type const& board_config, std::size_t level){
            return this->base_type::check_board(get_id(), board_config, level);
        }

        hpx::lcos::future<bool> check_board_async(board_type const& board_config,
            std::size_t level)
        {
            return this->base_type::check_board_async(get_id(), board_config, level);
        }
        //---------------------------------------------------------

        std::size_t solve_board(board_type const& board_config, std::size_t size,
            std::size_t level)
        {
            return this->base_type::solve_board(get_id(), board_config, size, level);
        }

        hpx::lcos::future<std::size_t>
        solve_board_async(board_type const& board_config, std::size_t size,
            std::size_t level)
        {
            return this->base_type::solve_board_async
                (get_id(), board_config, size, level);
        }
        //---------------------------------------------------------

        void clear_board(){
            return this->base_type::clear_board(get_id());
        }
    };

}

#endif // HPX_SUDOKU_EXAMPLE

