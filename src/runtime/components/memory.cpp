//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/components/server/memory.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/util/assert.hpp>

///////////////////////////////////////////////////////////////////////////////
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(hpx::components::server::memory,
    hpx::components::component_memory)

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the runtime_support actions
HPX_REGISTER_ACTION_ID(
    hpx::components::server::allocate_action, allocate_action,
    hpx::actions::allocate_action_id);

HPX_REGISTER_ACTION_ID(
    hpx::components::server::memory::store8_action, store8_action,
    hpx::actions::store8_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::memory::store16_action, store16_action,
    hpx::actions::store16_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::memory::store32_action, store32_action,
    hpx::actions::store32_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::memory::store64_action, store64_action,
    hpx::actions::store64_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::memory::store128_action, store128_action,
    hpx::actions::store128_action_id)

HPX_REGISTER_ACTION_ID(
    hpx::components::server::memory::load8_action, load8_action,
    hpx::actions::load8_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::memory::load16_action, load16_action,
    hpx::actions::load16_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::memory::load32_action, load32_action,
    hpx::actions::load32_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::memory::load64_action, load64_action,
    hpx::actions::load64_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::memory::load128_action, load128_action,
    hpx::actions::load128_action_id)

HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    hpx::components::server::memory::uint128_t, hpx_components_memory_uint128_t,
    hpx::actions::base_lco_with_value_uint128_get,
    hpx::actions::base_lco_with_value_uint128_set)

namespace hpx { namespace components { namespace server
{
    naming::gid_type allocate(std::size_t size)
    {
        naming::gid_type gid(hpx::applier::get_applier().get_raw_locality());
        gid.set_lsb(new boost::uint8_t[size]);
        naming::detail::set_credit_for_gid(gid,
            boost::int64_t(HPX_GLOBALCREDIT_INITIAL));
        return gid;
    }
}}}
