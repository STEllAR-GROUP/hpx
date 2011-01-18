//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/bh.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    hpx::components::server::bh
> bh_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(bh_type, bh);

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the bh actions
HPX_REGISTER_ACTION_EX(
    bh_type::wrapped_type::init_action,
    bh_init_action);
HPX_REGISTER_ACTION_EX(
    bh_type::wrapped_type::add_action,
    bh_add_action);
HPX_REGISTER_ACTION_EX(
    bh_type::wrapped_type::query_action,
    bh_query_action);
HPX_REGISTER_ACTION_EX(
    bh_type::wrapped_type::print_action,
    bh_print_action);
HPX_DEFINE_GET_COMPONENT_TYPE(bh_type::wrapped_type);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<unsigned long>::set_result_action,
    set_result_action_long);
HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<unsigned long>::get_value_action,
    get_value_action_long);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::lcos::base_lco_with_value<unsigned long>);
