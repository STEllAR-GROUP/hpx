//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/components.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include "server/allgather.hpp"
#include "server/allgather_and_gate.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::component<
    ag::server::allgather
> allgather_type;

HPX_REGISTER_COMPONENT(allgather_type, ag_allgather);

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION(
    allgather_type::wrapped_type::init_action,
    allgather_init_action);

HPX_REGISTER_ACTION(
    allgather_type::wrapped_type::compute_action,
    allgather_compute_action);

HPX_REGISTER_ACTION(
    allgather_type::wrapped_type::print_action,
    allgather_print_action);

HPX_REGISTER_ACTION(
    allgather_type::wrapped_type::get_item_action,
    allgather_get_item_action);

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::component<
    ag::server::allgather_and_gate
> allgather_and_gate_type;

HPX_REGISTER_COMPONENT(allgather_and_gate_type, ag_allgather_and_gate);

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION(
    allgather_and_gate_type::wrapped_type::compute_action,
    allgather_and_gate_compute_action);

HPX_REGISTER_ACTION(
    allgather_and_gate_type::wrapped_type::print_action,
    allgather_and_gate_print_action);

HPX_REGISTER_ACTION(
    allgather_and_gate_type::wrapped_type::set_data_action,
    allgather_and_gate_set_data_action);

