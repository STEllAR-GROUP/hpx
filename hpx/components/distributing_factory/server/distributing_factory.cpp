//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/runtime/components/server/manage_component.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "distributing_factory.hpp"

///////////////////////////////////////////////////////////////////////////////
// make sure all needed action::get_action_name() functions get defined
HPX_DEFINE_ACTION_NAME(hpx::lcos::base_lco_with_value<hpx::naming::id_type>::set_result_action);

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the accumulator actions
HPX_REGISTER_ACTION(hpx::components::server::detail::distributing_factory::create_action);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    component_type distributing_factory::value = component_invalid;

    component_type distributing_factory::get_component_type()
    {
        return value;
    }

    void distributing_factory::set_component_type(component_type type)
    {
        value = type;
    }

    ///////////////////////////////////////////////////////////////////////////
    // create a new instance of a component
    threads::thread_state distributing_factory::create(
        threads::thread_self& self, applier::applier& appl,
        naming::id_type* gid, components::component_type type,
        std::size_t count)
    {
    // set result if requested
        if (0 != gid)
            *gid = naming::invalid_id;
        return threads::terminated;
    }

}}}}

