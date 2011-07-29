////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "bhnode.hpp"

///////////////////////////////////////////////////////////////
HPX_REGISTER_COMPONENT_MODULE();

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<hpx::components::server::bhnode>,
    bhnode);

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<region_path>, component_base_lco_with_value);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<region_path>::set_result_action,
    base_lco_with_value_set_result_region_path);

//Register the actions
HPX_REGISTER_ACTION_EX(
    hpx::components::server::bhnode::constNodeAction,hplConstNodeAction);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::bhnode::cnstNodeAction2,hplCnstNodeAction2);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::bhnode::setBoundsAction,hplSetBoundsAction);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::bhnode::insrtNodeAction,hplInsrtNodeAction);
HPX_REGISTER_ACTION_EX(
    hpx::components::server::bhnode::updatChldAction,hplUpdatChldAction);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::components::server::bhnode);
