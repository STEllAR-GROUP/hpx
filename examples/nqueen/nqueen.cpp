//  Copyright (c) 2011 Vinay C Amatya
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>

#include "server/nqueen.hpp"

HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::component<nqueen::server::board> board_type;

HPX_REGISTER_COMPONENT(board_type, board);

// Serialization support for the board actions
HPX_REGISTER_ACTION(board_type::wrapped_type::init_action, board_init_action);

HPX_REGISTER_ACTION(board_type::wrapped_type::check_action, board_check_action);

HPX_REGISTER_ACTION(board_type::wrapped_type::access_action,
    board_access_action);

HPX_REGISTER_ACTION(board_type::wrapped_type::update_action,
    board_update_action);

HPX_REGISTER_ACTION(board_type::wrapped_type::solve_action, board_solve_action);

HPX_REGISTER_ACTION(board_type::wrapped_type::clear_action, board_clear_action);
#endif
