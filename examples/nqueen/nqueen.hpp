//  Copyright (c) 2011 Vinay C Amatya
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/include/runtime.hpp>
#include <hpx/include/client.hpp>

#include "server/nqueen.hpp"

#include <cstddef>
#include <utility>

namespace nqueen
{
    class board : public hpx::components::client_base<board, server::board>
    {
        typedef hpx::components::client_base<board, server::board> base_type;

    public:
        board() = default;

        board(hpx::future<hpx::id_type>&& gid)
          : base_type(std::move(gid))
        {
        }

        explicit board(hpx::id_type&& id)
          : base_type(std::move(id))
        {
        }

        void init_board(std::size_t size)
        {
            return init_board_async(size).get();
        }

        hpx::lcos::future<void> init_board_async(std::size_t size)
        {
            return hpx::async<server::board::init_action>(this->get_id(), size);
        }

        //-------------------------------------------------------
        list_type access_board()
        {
            return access_board_async().get();
        }

        hpx::lcos::future<list_type> access_board_async()
        {
            typedef server::board::access_action action_type;
            return hpx::async<action_type>(this->get_id());
        }

        //------------------------------------------------------
        void update_board(std::size_t level, std::size_t pos)
        {
            hpx::apply<server::board::update_action>(
                this->get_id(), level, pos);
        }

        //-----------------------------------------------------
        bool check_board(list_type const& list, std::size_t level)
        {
            return check_board_async(list, level).get();
        }

        hpx::lcos::future<bool> check_board_async(list_type const& list,
            std::size_t level)
        {
            typedef server::board::check_action action_type;
            return hpx::async<action_type>(this->get_id(), list, level);
        }

        //---------------------------------------------------------
        std::size_t solve_board(list_type const& list, std::size_t size,
            std::size_t level, std::size_t col)
        {
            return solve_board_async(list, size, level, col).get();
        }

        hpx::lcos::future<std::size_t> solve_board_async(list_type const& list,
            std::size_t size, std::size_t level, std::size_t col)
        {
            typedef server::board::solve_action action_type;
            return hpx::async<action_type>(
                this->get_id(), list, size, level, col);
        }

        //---------------------------------------------------------
        void clear_board()
        {
            hpx::apply<server::board::clear_action>(this->get_id());
        }
    };
}


#endif
