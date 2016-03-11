//  Copyright (c) 2016 Satyaki Upadhyay
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include "server/sudoku.hpp"

HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::component<
        sudoku::server::board
        > board_type;


HPX_REGISTER_COMPONENT(board_type, board);

// Serialization support for the board actions

HPX_REGISTER_ACTION(
    board_type::wrapped_type::init_action,
    board_init_action);

HPX_REGISTER_ACTION(
    board_type::wrapped_type::check_action,
    board_check_action);

HPX_REGISTER_ACTION(
    board_type::wrapped_type::access_action,
    board_access_action);

HPX_REGISTER_ACTION(
    board_type::wrapped_type::update_action,
    board_update_action);

HPX_REGISTER_ACTION(
    board_type::wrapped_type::solve_action,
    board_solve_action);

HPX_REGISTER_ACTION(
    board_type::wrapped_type::clear_action,
    board_clear_action);

HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<sudoku::list_type>::set_value_action,
    set_value_action_vector_std_size_t);

HPX_REGISTER_ACTION(
    hpx::lcos::base_lco_with_value<sudoku::list_type>::get_value_action,
    get_value_action_vector_std_size_t);

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<sudoku::list_type>,
    hpx::components::component_base_lco_with_value);

