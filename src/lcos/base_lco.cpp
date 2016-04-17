//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/serialize_exception.hpp>

#include <hpx/plugins/parcel/coalescing_message_handler_registration.hpp>

#include <boost/exception_ptr.hpp>

namespace hpx { namespace lcos
{
    void base_lco::set_exception(boost::exception_ptr const& e)
    {
        // just rethrow the exception
        boost::rethrow_exception(e);
    }

    void base_lco::connect(naming::id_type const &)
    {
    }

    void base_lco::disconnect(naming::id_type const &)
    {
    }

    components::component_type base_lco::get_component_type()
    {
        return components::get_component_type<base_lco>();
    }
    void base_lco::set_component_type(components::component_type type)
    {
        components::set_component_type<base_lco>(type);
    }

    base_lco::~base_lco() {}
    void base_lco::finalize() {}

    void base_lco::set_event_nonvirt()
    {
        set_event();
    }

    void base_lco::set_exception_nonvirt (boost::exception_ptr const& e)
    {
        set_exception(e);
    }

    void base_lco::connect_nonvirt(naming::id_type const & id)
    {
        connect(id);
    }

    void base_lco::disconnect_nonvirt(naming::id_type const & id)
    {
        disconnect(id);
    }
}}

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the base LCO actions
HPX_REGISTER_ACTION_ID(hpx::lcos::base_lco::set_event_action,
    base_set_event_action, hpx::actions::base_set_event_action_id)
HPX_REGISTER_ACTION_ID(hpx::lcos::base_lco::set_exception_action,
    base_set_exception_action, hpx::actions::base_set_exception_action_id)
HPX_REGISTER_ACTION_ID(hpx::lcos::base_lco::connect_action,
    base_connect_action, hpx::actions::base_connect_action_id)
HPX_REGISTER_ACTION_ID(hpx::lcos::base_lco::disconnect_action,
    base_disconnect_action, hpx::actions::base_disconnect_action_id)

///////////////////////////////////////////////////////////////////////////////
HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW(
    hpx::lcos::base_lco::set_event_action, "lco_set_value_action",
    std::size_t(-1), std::size_t(-1))
HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW(
    hpx::lcos::base_lco::set_exception_action, "lco_set_value_action",
    std::size_t(-1), std::size_t(-1))

///////////////////////////////////////////////////////////////////////////////
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco, hpx::components::component_base_lco)

