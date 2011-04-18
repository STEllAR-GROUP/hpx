/*
 * nqueen.cpp
 *      Author: vamatya
 */
#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/nqueen.hpp"

HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::managed_component<
        hpx::components::server::Board
        > board_type;


HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(board_type, Board);

// Serialization support for the board actions

HPX_REGISTER_ACTION_EX(
    board_type::wrapped_type::init_action,
    board_init_action);
HPX_REGISTER_ACTION_EX(
    board_type::wrapped_type::print_action,
    board_print_action);

HPX_REGISTER_ACTION_EX(
    board_type::wrapped_type::check_action,
    board_check_action);
HPX_REGISTER_ACTION_EX(
    board_type::wrapped_type::access_action,
    board_access_action);

HPX_REGISTER_ACTION_EX(
    board_type::wrapped_type::size_action,
    board_size_action);

HPX_REGISTER_ACTION_EX(
    board_type::wrapped_type::level_action,
    board_level_action);

HPX_REGISTER_ACTION_EX(
    board_type::wrapped_type::update_action,
    board_update_action);
HPX_REGISTER_ACTION_EX(
    board_type::wrapped_type::solve_action,
    board_solve_action);
HPX_REGISTER_ACTION_EX(
    board_type::wrapped_type::clear_action,
    board_clear_action);
HPX_REGISTER_ACTION_EX(
    board_type::wrapped_type::test_action,
    board_test_action);


HPX_DEFINE_GET_COMPONENT_TYPE(board_type::wrapped_type);

//////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<bool>::set_result_action,
    set_result_action_bool);
HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<list_t>::set_result_action,
    set_result_action_list_t);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<unsigned int>::set_result_action,
    set_result_action_uint);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<int>::set_result_action,
    set_result_action_int);

////////////////////////////////////////////////////////////////////////

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<bool>::get_value_action,
    get_value_action_bool);
HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<list_t>::get_value_action,
    get_value_action_list_t);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<unsigned int>::get_value_action,
    get_value_action_uint);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<int>::get_value_action,
    get_value_action_int);

///////////////////////////////////////////////////////////////////////

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<bool>,
    hpx::components::component_base_lco_with_value);
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<list_t>,
    hpx::components::component_base_lco_with_value);

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<unsigned int>,
    hpx::components::component_base_lco_with_value);

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<int>,
    hpx::components::component_base_lco_with_value);

////////////////////////////////////////////////////////////
/*HPX_REGISTER_ACTION_EX(
    hpx::components::server::Board::init_action,
    Board_init_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::Board::access_action,
    Board_access_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::Board::update_action,
    Board_update_action);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::Board::check_action,
    Board_check_action);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::components::server::Board);*/

