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

#include <examples/nqueen/nqueen/server/nqueen.hpp>

namespace hpx { namespace components { namespace stubs
{

    struct Board : stub_base<server::Board>
    {

        static void initBoard(naming::id_type gid,std::size_t size, std::size_t level)
        {
            applier::apply<server::Board::init_action>(gid, size, level);
        }

        static void printBoard(naming::id_type gid)
        {
            applier::apply<server::Board::print_action>(gid);
        }

        static lcos::future_value<list_t> accessBoard_async(naming::id_type gid)
        {
            typedef server::Board::access_action action_type;
            return lcos::eager_future<action_type>(gid);
        }

        static list_t accessBoard(naming::id_type gid)
        {
            return accessBoard_async(gid).get();
        }

        static lcos::future_value<std::size_t> getSize_async(naming::id_type gid)
        {
            typedef server::Board::size_action action_type;
            return lcos::eager_future<action_type>(gid);
        }

        static std::size_t getSize(naming::id_type gid)
        {
            return getSize_async(gid).get();
        }

        static lcos::future_value<std::size_t> getLevel_async(naming::id_type gid)
        {
            typedef server::Board::level_action action_type;
            return lcos::eager_future<action_type>(gid);
        }

        static std::size_t getLevel(naming::id_type gid)
        {
            return getLevel_async(gid).get();
        }

        static void updateBoard(naming::id_type gid, std::size_t level, std::size_t pos )
        {
            applier::apply<server::Board::update_action>(gid, level, pos);
        }

        static lcos::future_value<bool> checkBoard_async(naming::id_type gid, list_t list, std::size_t level)
        {
            typedef server::Board::check_action action_type;
            return lcos::eager_future<action_type>(gid, list, level);
        }

        static bool checkBoard(naming::id_type gid, list_t list, std::size_t level )
        {
            return checkBoard_async(gid, list, level).get();
        }

        static void solveNqueen(naming::id_type gid, list_t list, std::size_t size, std::size_t level)
        {
            applier::apply<server::Board::solve_action>(gid, list, size, level);
        }

        static void clearBoard(naming::id_type gid)
        {
            applier::apply<server::Board::clear_action>(gid);
        }

        static void testBoard(naming::id_type gid, list_t list, std::size_t size, std::size_t level)
        {
            applier::apply<server::Board::test_action>(gid, list, size, level);
        }

    };
}}}

#endif // HPX_E4B0BA36_0E1C_48F5_928B_CDC78F1D2C40

