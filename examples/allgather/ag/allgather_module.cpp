//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/allgather.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    ag::server::allgather
> allgather_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(allgather_type, ag_allgather);

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION_EX(
    allgather_type::wrapped_type::init_action,
    allgather_init_action);

HPX_REGISTER_ACTION_EX(
    allgather_type::wrapped_type::compute_action,
    allgather_compute_action);

HPX_REGISTER_ACTION_EX(
    allgather_type::wrapped_type::print_action,
    allgather_print_action);

HPX_REGISTER_ACTION_EX(
    allgather_type::wrapped_type::get_item_action,
    allgather_get_item_action);
