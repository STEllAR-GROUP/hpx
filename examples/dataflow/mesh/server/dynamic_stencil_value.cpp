//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2009 Matt Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "dynamic_stencil_value.hpp"
#include "dynamic_stencil_value_impl.hpp"

///////////////////////////////////////////////////////////////////////////////
/// Define types of stencil components exposed by this module
typedef hpx::components::managed_component<
    hpx::components::amr::server::dynamic_stencil_value
> dataflow_dynamic_stencil_value_double_type;

///////////////////////////////////////////////////////////////////////////////
/// The following construct registers a minimal factory needed for the creation
/// of new stencil instances. The name used as the second macro parameter
/// must match the component name used in the ini configuration file used
/// for this component. For instance the configuration file had_amr.ini may look
/// like:
///
/// [hpx.components.dynamic_stencil_double]      # this must match the string below
/// name = had_amr                               # this must match the name of the shared library
/// path = $[hpx.location]/lib    # this is the default location where to find the shared library
///
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(dataflow_dynamic_stencil_value_double_type,
    dataflow_dynamic_stencil_double3d);

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION(
    dataflow_dynamic_stencil_value_double_type::wrapped_type::call_action,
    dataflow_dynamic_stencil_value_double_call_action);
HPX_REGISTER_ACTION(
    dataflow_dynamic_stencil_value_double_type::wrapped_type::get_output_ports_action,
    dataflow_dynamic_stencil_value_double_get_output_ports_action);
HPX_REGISTER_ACTION(
    dataflow_dynamic_stencil_value_double_type::wrapped_type::connect_input_ports_action,
    dataflow_dynamic_stencil_value_double_connect_input_ports_action);
HPX_REGISTER_ACTION(
    dataflow_dynamic_stencil_value_double_type::wrapped_type::set_functional_component_action,
    dataflow_dynamic_stencil_value_double_set_functional_component_action);
HPX_REGISTER_ACTION(
    dataflow_dynamic_stencil_value_double_type::wrapped_type::start_action,
    dataflow_dynamic_stencil_value_double_start_action);
