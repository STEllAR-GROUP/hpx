//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/actions/transfer_action.hpp>
#include <hpx/actions/transfer_continuation_action.hpp>
#include <hpx/actions_base/basic_action.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime/components/server/component_heap.hpp>

#include <cstddef>
#include <exception>

namespace hpx { namespace lcos
{
    void base_lco::set_exception(std::exception_ptr const& e)
    {
        // just rethrow the exception
        std::rethrow_exception(e);
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

    void base_lco::set_exception_nonvirt (std::exception_ptr const& e)
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
HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW_DEFINITION(
    hpx::lcos::base_lco::set_event_action, "lco_set_value_action",
    std::size_t(-1), std::size_t(-1))
HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW_DEFINITION(
    hpx::lcos::base_lco::set_exception_action, "lco_set_value_action",
    std::size_t(-1), std::size_t(-1))

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

HPX_REGISTER_COMPONENT_HEAP(hpx::components::managed_component<hpx::lcos::base_lco>)

///////////////////////////////////////////////////////////////////////////////
HPX_DEFINE_COMPONENT_NAME(hpx::lcos::base_lco, hpx_lcos_base_lco);
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco, hpx::components::component_base_lco)

