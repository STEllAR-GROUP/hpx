//  Copyright (c) 2016 Satyaki Upadhyay
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SUDOKU_EXAMPLE_STUBS)
#define HPX_SUDOKU_EXAMPLE_STUBS

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/include/async.hpp>

#include <examples/sudoku/server/sudoku.hpp>


namespace sudoku { namespace stubs
{

    struct board : hpx::components::stub_base<server::board>
    {

        static void init_board(hpx::naming::id_type const& gid, std::size_t size)
        {
            hpx::apply<server::board::init_action>(gid, size);
        }
        //--------------------------------------------------------------

        static hpx::lcos::future<board_type>
        access_board_async(hpx::naming::id_type const& gid)
        {
            typedef server::board::access_action action_type;
            return hpx::async<action_type>(gid);
        }

        static board_type access_board(hpx::naming::id_type const& gid)
        {
            return access_board_async(gid).get();
        }

        //-------------------------------------------------------------

        static void update_board(hpx::naming::id_type const& gid,
            std::size_t level, std::size_t pos)
        {
            hpx::apply<server::board::update_action>(gid, level, pos);
        }

        //------------------------------------------------------------

        static hpx::lcos::future<bool>
        check_board_async(hpx::naming::id_type const& gid,
            board_type const& board_config, std::size_t level)
        {
            typedef server::board::check_action action_type;
            return hpx::async<action_type>(gid, board_config, level);
        }

        static bool check_board(hpx::naming::id_type const& gid,
            board_type const& board_config, std::size_t level)
        {
            return check_board_async(gid, board_config, level).get();
        }
        //-----------------------------------------------------------

        static std::size_t solve_board(hpx::naming::id_type const& gid,
            board_type const& board_config, std::size_t size, std::size_t level)
        {
            return solve_board_async(gid, board_config, size, level).get();
        }

        static hpx::lcos::future<std::size_t> solve_board_async(
            hpx::naming::id_type const& gid, board_type const& board_config,
            std::size_t size, std::size_t level)
        {
            typedef server::board::solve_action action_type;
            return hpx::async<action_type>
                (gid, board_config, size, level);
        }
        //----------------------------------------------------------

        static void clear_board(hpx::naming::id_type const& gid)
        {
            hpx::apply<server::board::clear_action>(gid);
        }

    };
}}

#endif // HPX_SUDOKU_EXAMPLE_STUBS

