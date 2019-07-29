//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/lcos/latch.hpp>

#include <hpx/assertion.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/server/latch.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/actions/detail/action_factory.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/components/new.hpp>
#include <hpx/util/serialize_exception.hpp>

#include <cstddef>
#include <exception>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    latch::latch(std::ptrdiff_t count)
      : base_type(hpx::new_<lcos::server::latch>(hpx::find_here(), count))
    {}

    hpx::future<void> latch::count_down_and_wait_async()
    {
        lcos::server::latch::set_event_action act;
        return hpx::async(act, get_id());
    }

    hpx::future<void> latch::count_down_async(std::ptrdiff_t n)
    {
        lcos::server::latch::set_value_action act;
        return hpx::async(act, get_id(), std::move(n));
    }

    hpx::future<bool> latch::is_ready_async() const
    {
        lcos::server::latch::get_value_action act;
        return hpx::async(act, get_id());
    }

    hpx::future<void> latch::wait_async() const
    {
        lcos::server::latch::wait_action act;
        return hpx::async(act, get_id());
    }

    hpx::future<void> latch::set_exception_async(std::exception_ptr const& e)
    {
        lcos::server::latch::set_exception_action act;
        return hpx::async(act, get_id(), e);
    }
}}

///////////////////////////////////////////////////////////////////////////////
// latch
typedef hpx::components::managed_component<hpx::lcos::server::latch> latch_type;

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::server::latch, hpx::components::component_latch)
HPX_REGISTER_DERIVED_COMPONENT_FACTORY(latch_type, hpx_lcos_server_latch,
    "hpx::lcos::base_lco_with_value", hpx::components::factory_enabled)

HPX_REGISTER_ACTION_ID(
    hpx::lcos::server::latch::create_component_action,
    hpx_lcos_server_latch_create_component_action,
    hpx::actions::hpx_lcos_server_latch_create_component_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::lcos::server::latch::wait_action,
    hpx_lcos_server_latch_wait_action,
    hpx::actions::hpx_lcos_server_latch_wait_action_id)

HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    bool, std::ptrdiff_t, bool_std_ptrdiff,
    hpx::actions::base_lco_with_value_std_bool_ptrdiff_get,
    hpx::actions::base_lco_with_value_std_bool_ptrdiff_set)
