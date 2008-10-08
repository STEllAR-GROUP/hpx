//  Copyright (c) 2007-2008 Hartmut Kaiser
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

#include <hpx/components/simple_accumulator/server/simple_accumulator.hpp>

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::server::simple_accumulator, "simple_accumulator");

///////////////////////////////////////////////////////////////////////////////
// make sure all needed action::get_action_name() functions get defined
HPX_DEFINE_ACTION_NAME(hpx::lcos::base_lco_with_value<double>::set_result_action);
HPX_DEFINE_ACTION_NAME(hpx::lcos::base_lco_with_value<hpx::naming::id_type>::set_result_action);

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the simple_accumulator actions
HPX_REGISTER_ACTION(hpx::components::server::simple_accumulator::init_action);
HPX_REGISTER_ACTION(hpx::components::server::simple_accumulator::add_action);
HPX_REGISTER_ACTION(hpx::components::server::simple_accumulator::query_action);
HPX_REGISTER_ACTION(hpx::components::server::simple_accumulator::print_action);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server 
{
    component_type simple_accumulator::value = component_invalid;

    component_type simple_accumulator::get_component_type()
    {
        return value;
    }

    void simple_accumulator::set_component_type(component_type type)
    {
        value = type;
    }

}}}

