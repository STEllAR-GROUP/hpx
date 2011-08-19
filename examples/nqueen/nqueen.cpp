//  Copyright (c) 2011 Vinay C Amatya
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/nqueen.hpp"

HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::managed_component<
        hpx::components::server::board
        > board_type;


HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(board_type, board);

// Serialization support for the board actions

HPX_REGISTER_ACTION_EX(
    board_type::wrapped_type::init_action,
    board_init_action);

HPX_REGISTER_ACTION_EX(
    board_type::wrapped_type::check_action,
    board_check_action);

HPX_REGISTER_ACTION_EX(
    board_type::wrapped_type::access_action,
    board_access_action);

HPX_REGISTER_ACTION_EX(
    board_type::wrapped_type::update_action,
    board_update_action);

HPX_REGISTER_ACTION_EX(
    board_type::wrapped_type::solve_action,
    board_solve_action);

HPX_REGISTER_ACTION_EX(
    board_type::wrapped_type::clear_action,
    board_clear_action);

HPX_DEFINE_GET_COMPONENT_TYPE(board_type::wrapped_type);

//------------------------------------------------------------------
HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<bool>::set_result_action,
    set_result_action_bool);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<list_t>::set_result_action,
    set_result_action_list_t);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<std::size_t>::set_result_action,
    set_result_action_uint); 

//-------------------------------------------------------------------

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<bool>::get_value_action,
    get_value_action_bool);
HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<list_t>::get_value_action,
    get_value_action_list_t);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<std::size_t>::get_value_action,
    get_value_action_uint);

//--------------------------------------------------------------------

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<bool>,
    hpx::components::component_base_lco_with_value);

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<list_t>,
    hpx::components::component_base_lco_with_value);

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<std::size_t>,
    hpx::components::component_base_lco_with_value);

