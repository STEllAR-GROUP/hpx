//  Copyright (c) 2011 Vinay C Amatya
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_E4B0BA36_0E1C_48F5_928B_CDC78F1D2C40)
#define HPX_E4B0BA36_0E1C_48F5_928B_CDC78F1D2C40

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/include/async.hpp>

#include "../server/nqueen.hpp"

namespace nqueen { namespace stubs
{

    struct board : hpx::components::stub_base<server::board>
    {

        static void init_board(hpx::naming::id_type const& gid, std::size_t size)
        {
            hpx::apply<server::board::init_action>(gid, size);
        }
        //--------------------------------------------------------------

        static hpx::lcos::future<list_type>
        access_board_async(hpx::naming::id_type const& gid)
        {
            typedef server::board::access_action action_type;
            return hpx::async<action_type>(gid);
        }

        static list_type access_board(hpx::naming::id_type const& gid)
        {
            return access_board_async(gid).get();
        }

        //-------------------------------------------------------------

        static void update_board(hpx::naming::id_type const& gid, std::size_t level,
            std::size_t pos)
        {
            hpx::apply<server::board::update_action>(gid, level, pos);
        }

        //------------------------------------------------------------

        static hpx::lcos::future<bool>
        check_board_async(hpx::naming::id_type const& gid, list_type const& list,
            std::size_t level)
        {
            typedef server::board::check_action action_type;
            return hpx::async<action_type>(gid, list, level);
        }

        static bool check_board(hpx::naming::id_type const& gid,
            list_type const& list, std::size_t level)
        {
            return check_board_async(gid, list, level).get();
        }
        //-----------------------------------------------------------

        static std::size_t solve_board(hpx::naming::id_type const& gid,
            list_type const& list, std::size_t size, std::size_t level,
            std::size_t col)
        {
            return solve_board_async(gid, list, size, level, col).get();
        }

        static hpx::lcos::future<std::size_t> solve_board_async(
            hpx::naming::id_type const& gid, list_type const& list,
            std::size_t size, std::size_t level, std::size_t col)
        {
            typedef server::board::solve_action action_type;
            return hpx::async<action_type>
                (gid, list, size, level, col);
        }
        //----------------------------------------------------------

        static void clear_board(hpx::naming::id_type const& gid)
        {
            hpx::apply<server::board::clear_action>(gid);
        }

    };
}}

#endif // HPX_E4B0BA36_0E1C_48F5_928B_CDC78F1D2C40

