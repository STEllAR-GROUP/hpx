//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include "server/props.hpp"

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    hpx::components::server::props
> props_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(props_type, props);

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the props actions
HPX_REGISTER_ACTION_EX(
    props_type::wrapped_type::init_action,
    props_init_action);
HPX_REGISTER_ACTION_EX(
    props_type::wrapped_type::color_action,
    props_color_action);
HPX_REGISTER_ACTION_EX(
    props_type::wrapped_type::incr_action,
    props_incr_action);
HPX_DEFINE_GET_COMPONENT_TYPE(props_type::wrapped_type);

