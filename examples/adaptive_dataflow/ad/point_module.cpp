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
    ad::server::point
> ad_point_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(ad_point_type, ad_point);

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION(
    ad_point_type::wrapped_type::init_action,
    ad_point_init_action);

HPX_REGISTER_ACTION(
    ad_point_type::wrapped_type::compute_action,
    ad_point_compute_action);

HPX_REGISTER_ACTION(
    ad_point_type::wrapped_type::get_item_action,
    ad_point_get_item_action);

HPX_REGISTER_ACTION(
    ad_point_type::wrapped_type::remove_item_action,
    ad_point_remove_item_action);
