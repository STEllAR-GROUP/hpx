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
#include <hpx/lcos/eager_future.hpp>

#include <examples/nqueen/server/nqueen.hpp>

namespace hpx { namespace components { namespace stubs
{

    struct board : stub_base<server::board>
    {

        static void init_board(naming::id_type gid, std::size_t size )
        {
            applier::apply<server::board::init_action>( gid, size );
        }
        //--------------------------------------------------------------

        static lcos::future_value<list_t> access_board_async(naming::id_type gid)
        {
            typedef server::board::access_action action_type;
            return lcos::eager_future<action_type>(gid);
        }

        static list_t access_board(naming::id_type gid)
        {
            return access_board_async(gid).get();
        }

        //-------------------------------------------------------------

        static void update_board(naming::id_type gid, std::size_t level, 
            std::size_t pos )
        {
            applier::apply<server::board::update_action>(gid, level, pos);
        }
        
        //------------------------------------------------------------

        static lcos::future_value<bool> check_board_async(naming::id_type gid, 
            list_t list, std::size_t level)
        {
            typedef server::board::check_action action_type;
            return lcos::eager_future<action_type>(gid, list, level);
        }

        static bool check_board(naming::id_type gid, list_t list, 
            std::size_t level )
        {
            return check_board_async(gid, list, level).get();
        }
        //-----------------------------------------------------------

        static std::size_t solve_board(naming::id_type gid, list_t list, 
            std::size_t size, std::size_t level, std::size_t col)
        {
            return solve_board_async(gid, list, size, level, col).get();
        }

        static lcos::future_value<std::size_t> solve_board_async(
            naming::id_type gid, list_t list, std::size_t size, 
            std::size_t level, std::size_t col)
        {
            typedef server::board::solve_action action_type;
            return lcos::eager_future<action_type>(gid, list, size, level, col);
        }
        //----------------------------------------------------------

        static void clear_board(naming::id_type gid)
        {
            applier::apply<server::board::clear_action>(gid);
        }

    };
}}}

#endif // HPX_E4B0BA36_0E1C_48F5_928B_CDC78F1D2C40

