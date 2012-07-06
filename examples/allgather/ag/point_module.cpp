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

#include "server/point.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    ag::server::point
> ag_point_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(ag_point_type, ag_point);

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION_EX(
    ag_point_type::wrapped_type::init_action,
    ag_point_init_action);

HPX_REGISTER_ACTION_EX(
    ag_point_type::wrapped_type::compute_action,
    ag_point_compute_action);

HPX_REGISTER_ACTION_EX(
    ag_point_type::wrapped_type::print_action,
    ag_point_print_action);

HPX_REGISTER_ACTION_EX(
    ag_point_type::wrapped_type::get_item_action,
    ag_point_get_item_action);
