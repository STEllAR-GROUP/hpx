//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/components/amr/stencil.hpp>
#include <hpx/components/amr/server/stencil_value.ipp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////
/// The following construct registers a minimal factory needed for the creation
/// of new stencil instances. The name used as the second macro parameter
/// must match the component name used in the ini configuration file used
/// for this component. For instance the configuration file amr.ini may look 
/// like:
/// 
/// [hpx.components.stencil]      # this must match the string below
/// name = amr                    # this must match the name of the shared library
/// path = $[hpx.location]/lib    # this is the default location where to find the shared library
///
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::amr::stencil, "stencil");

///////////////////////////////////////////////////////////////////////////////
// For any component derived from manage_component_base we must use the 
// following in exactly one source file
HPX_REGISTER_MANAGED_COMPONENT(hpx::components::amr::stencil);

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION(hpx::components::amr::detail::stencil::call_action);
HPX_REGISTER_ACTION(hpx::components::amr::detail::stencil::get_output_ports_action);
HPX_REGISTER_ACTION(hpx::components::amr::detail::stencil::connect_input_ports_action);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////////
    // Implement actual functionality of this stencil

    // Compute the result value for the current time step
    double stencil::eval(double x, double y, double z)
    {
        ++timestep_;
        return (x + y + z) / 3;
    }

    // Return, whether the current time step is the final one
    bool stencil::is_last_timestep() const
    {
        return timestep_ == 2;
    }

}}}}

