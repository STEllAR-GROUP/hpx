//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/actions/continuation.hpp>
#include <hpx/actions/transfer_action.hpp>
#include <hpx/actions/transfer_continuation_action.hpp>
#include <hpx/actions_base/detail/action_factory.hpp>
#include <hpx/assert.hpp>
#include <hpx/collectives/latch.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/components/new.hpp>
#include <hpx/runtime_local/detail/serialize_exception.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <exception>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos {
    latch::latch(std::ptrdiff_t count)
      : base_type(hpx::new_<lcos::server::latch>(hpx::find_here(), count))
    {
    }

    hpx::future<void> latch::count_down_and_wait_async()
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        lcos::server::latch::set_event_action act;
        return hpx::async(act, get_id());
#else
        HPX_ASSERT(false);
        return hpx::make_ready_future();
#endif
    }

    hpx::future<void> latch::count_down_async(std::ptrdiff_t n)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        lcos::server::latch::set_value_action act;
        return hpx::async(act, get_id(), std::move(n));
#else
        HPX_ASSERT(false);
        HPX_UNUSED(n);
        return hpx::make_ready_future();
#endif
    }

    hpx::future<bool> latch::is_ready_async() const
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        lcos::server::latch::get_value_action act;
        return hpx::async(act, get_id());
#else
        HPX_ASSERT(false);
        return hpx::make_ready_future(true);
#endif
    }

    hpx::future<void> latch::wait_async() const
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        lcos::server::latch::wait_action act;
        return hpx::async(act, get_id());
#else
        HPX_ASSERT(false);
        return hpx::make_ready_future();
#endif
    }

    hpx::future<void> latch::set_exception_async(std::exception_ptr const& e)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        lcos::server::latch::set_exception_action act;
        return hpx::async(act, get_id(), e);
#else
        HPX_ASSERT(false);
        HPX_UNUSED(e);
        return hpx::make_ready_future();
#endif
    }
}}    // namespace hpx::lcos

///////////////////////////////////////////////////////////////////////////////
// latch
typedef hpx::components::managed_component<hpx::lcos::server::latch> latch_type;

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::server::latch, hpx::components::component_latch)
HPX_REGISTER_DERIVED_COMPONENT_FACTORY(latch_type, hpx_lcos_server_latch,
    "hpx::lcos::base_lco_with_value", hpx::components::factory_enabled)

HPX_REGISTER_ACTION_ID(hpx::lcos::server::latch::create_component_action,
    hpx_lcos_server_latch_create_component_action,
    hpx::actions::hpx_lcos_server_latch_create_component_action_id)
HPX_REGISTER_ACTION_ID(hpx::lcos::server::latch::wait_action,
    hpx_lcos_server_latch_wait_action,
    hpx::actions::hpx_lcos_server_latch_wait_action_id)

HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(bool, std::ptrdiff_t, bool_std_ptrdiff,
    hpx::actions::base_lco_with_value_std_bool_ptrdiff_get,
    hpx::actions::base_lco_with_value_std_bool_ptrdiff_set)
