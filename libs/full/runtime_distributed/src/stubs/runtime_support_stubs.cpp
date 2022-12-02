//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/actions_base/traits/action_was_object_migrated.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_colocated/post_colocated.hpp>
#include <hpx/async_distributed/post.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/components_base/traits/component_supports_migration.hpp>
#include <hpx/ini/ini.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/runtime_distributed/applier.hpp>
#include <hpx/runtime_distributed/stubs/runtime_support.hpp>
#include <hpx/runtime_local/runtime_local.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace hpx { namespace components { namespace stubs {

    hpx::future<int> runtime_support::load_components_async(
        hpx::id_type const& gid)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        typedef server::runtime_support::load_components_action action_type;
        return hpx::async<action_type>(gid);
#else
        HPX_ASSERT(false);
        HPX_UNUSED(gid);
        return hpx::make_ready_future(0);
#endif
    }

    int runtime_support::load_components(hpx::id_type const& gid)
    {
        return load_components_async(gid).get();
    }

    hpx::future<void> runtime_support::call_startup_functions_async(
        hpx::id_type const& gid, bool pre_startup)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        typedef server::runtime_support::call_startup_functions_action
            action_type;
        return hpx::async<action_type>(gid, pre_startup);
#else
        HPX_ASSERT(false);
        HPX_UNUSED(gid);
        HPX_UNUSED(pre_startup);
        return ::hpx::make_ready_future();
#endif
    }

    void runtime_support::call_startup_functions(
        hpx::id_type const& gid, bool pre_startup)
    {
        call_startup_functions_async(gid, pre_startup).get();
    }

    /// \brief Shutdown the given runtime system
    hpx::future<void> runtime_support::shutdown_async(
        hpx::id_type const& targetgid, double timeout)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        // Create a promise directly and execute the required action.
        // This action has implemented special response handling as the
        // back-parcel is sent explicitly (and synchronously).
        typedef server::runtime_support::shutdown_action action_type;

        hpx::distributed::promise<void> value;
        auto f = value.get_future();

        // We need to make it unmanaged to avoid late refcnt requests
        id_type gid(
            value.get_id().get_gid(), id_type::management_type::unmanaged);
        hpx::post<action_type>(targetgid, timeout, gid);

        return f;
#else
        HPX_ASSERT(false);
        HPX_UNUSED(targetgid);
        HPX_UNUSED(timeout);
        return ::hpx::make_ready_future();
#endif
    }

    void runtime_support::shutdown(
        hpx::id_type const& targetgid, double timeout)
    {
        // The following get yields control while the action above
        // is executed and the result is returned to the future
        shutdown_async(targetgid, timeout).get();
    }

    /// \brief Shutdown the runtime systems of all localities
    void runtime_support::shutdown_all(
        hpx::id_type const& targetgid, double timeout)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        hpx::post<server::runtime_support::shutdown_all_action>(
            targetgid, timeout);
#else
        HPX_ASSERT(false);
        HPX_UNUSED(targetgid);
        HPX_UNUSED(timeout);
#endif
    }

    void runtime_support::shutdown_all(double timeout)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        hpx::post<server::runtime_support::shutdown_all_action>(
            hpx::id_type(
                hpx::applier::get_applier().get_runtime_support_raw_gid(),
                hpx::id_type::management_type::unmanaged),
            timeout);
#else
        HPX_ASSERT(false);
        HPX_UNUSED(timeout);
#endif
    }

    ///////////////////////////////////////////////////////////////////////
    /// \brief Retrieve configuration information
    /// \brief Terminate the given runtime system
    hpx::future<void> runtime_support::terminate_async(
        hpx::id_type const& targetgid)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        // Create a future directly and execute the required action.
        // This action has implemented special response handling as the
        // back-parcel is sent explicitly (and synchronously).
        typedef server::runtime_support::terminate_action action_type;

        hpx::distributed::promise<void> value;
        auto f = value.get_future();

        hpx::post<action_type>(targetgid, value.get_id());
        return f;
#else
        HPX_ASSERT(false);
        HPX_UNUSED(targetgid);
        return ::hpx::make_ready_future();
#endif
    }

    void runtime_support::terminate(hpx::id_type const& targetgid)
    {
        // The following get yields control while the action above
        // is executed and the result is returned to the future
        terminate_async(targetgid).get();
    }

    /// \brief Terminate the runtime systems of all localities
    void runtime_support::terminate_all(hpx::id_type const& targetgid)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        hpx::post<server::runtime_support::terminate_all_action>(targetgid);
#else
        HPX_ASSERT(false);
        HPX_UNUSED(targetgid);
#endif
    }

    void runtime_support::terminate_all()
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        hpx::post<server::runtime_support::terminate_all_action>(hpx::id_type(
            hpx::applier::get_applier().get_runtime_support_raw_gid(),
            hpx::id_type::management_type::unmanaged));
#else
        HPX_ASSERT(false);
#endif
    }

    ///////////////////////////////////////////////////////////////////////
    void runtime_support::garbage_collect_non_blocking(
        hpx::id_type const& targetgid)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        typedef server::runtime_support::garbage_collect_action action_type;
        hpx::post<action_type>(targetgid);
#else
        HPX_ASSERT(false);
        HPX_UNUSED(targetgid);
#endif
    }

    hpx::future<void> runtime_support::garbage_collect_async(
        hpx::id_type const& targetgid)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        typedef server::runtime_support::garbage_collect_action action_type;
        return hpx::async<action_type>(targetgid);
#else
        HPX_ASSERT(false);
        HPX_UNUSED(targetgid);
        return ::hpx::make_ready_future();
#endif
    }

    void runtime_support::garbage_collect(hpx::id_type const& targetgid)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        typedef server::runtime_support::garbage_collect_action action_type;
        hpx::async<action_type>(targetgid).get();
#else
        HPX_ASSERT(false);
        HPX_UNUSED(targetgid);
#endif
    }

    ///////////////////////////////////////////////////////////////////////
    hpx::future<hpx::id_type> runtime_support::create_performance_counter_async(
        hpx::id_type targetgid, performance_counters::counter_info const& info)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        if (!naming::is_locality(targetgid))
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "stubs::runtime_support::create_performance_counter_async",
                "The id passed as the first argument is not representing"
                " a locality");
            return make_ready_future(hpx::invalid_id);
        }

        typedef server::runtime_support::create_performance_counter_action
            action_type;
        return hpx::async<action_type>(targetgid, info);
#else
        HPX_ASSERT(false);
        HPX_UNUSED(targetgid);
        HPX_UNUSED(info);
        return ::hpx::make_ready_future(hpx::invalid_id);
#endif
    }

    hpx::id_type runtime_support::create_performance_counter(
        hpx::id_type targetgid, performance_counters::counter_info const& info,
        error_code& ec)
    {
        return create_performance_counter_async(targetgid, info).get(ec);
    }

    ///////////////////////////////////////////////////////////////////////
    /// \brief Retrieve configuration information
    hpx::future<util::section> runtime_support::get_config_async(
        hpx::id_type const& targetgid)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        // Create a future, execute the required action,
        // we simply return the initialized future, the caller needs
        // to call get() on the return value to obtain the result
        typedef server::runtime_support::get_config_action action_type;
        return hpx::async<action_type>(targetgid);
#else
        HPX_ASSERT(false);
        HPX_UNUSED(targetgid);
        return ::hpx::make_ready_future(util::section{});
#endif
    }

    void runtime_support::get_config(
        hpx::id_type const& targetgid, util::section& ini)
    {
        // The following get yields control while the action above
        // is executed and the result is returned to the future
        ini = get_config_async(targetgid).get();
    }

    ///////////////////////////////////////////////////////////////////////
    void runtime_support::remove_from_connection_cache_async(
        hpx::id_type const& target, naming::gid_type const& gid,
        parcelset::endpoints_type const& endpoints)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        typedef server::runtime_support::remove_from_connection_cache_action
            action_type;
        hpx::post<action_type>(target, gid, endpoints);
#else
        HPX_ASSERT(false);
        HPX_UNUSED(target);
        HPX_UNUSED(gid);
        HPX_UNUSED(endpoints);
#endif
    }
}}}    // namespace hpx::components::stubs
