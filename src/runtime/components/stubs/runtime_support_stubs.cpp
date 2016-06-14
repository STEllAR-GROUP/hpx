//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/apply.hpp>
#include <hpx/async.hpp>
#include <hpx/runtime/actions/manage_object_action.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/applier/detail/apply_colocated.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/get_colocation_id.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/runtime.hpp>
#include <hpx/traits/component_supports_migration.hpp>
#include <hpx/traits/action_was_object_migrated.hpp>

#include <utility>
#include <vector>

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////
    lcos::future<std::vector<naming::id_type> >
    runtime_support::bulk_create_components_async(
        naming::id_type const& gid, components::component_type type,
        std::size_t count)
    {
        if (!naming::is_locality(gid))
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "stubs::runtime_support::bulk_create_components_async",
                "The id passed as the first argument is not representing"
                    " a locality");
            return make_ready_future(std::vector<naming::id_type>());
        }

        // Create a future, execute the required action,
        // we simply return the initialized future, the caller needs
        // to call get() on the return value to obtain the result
        typedef server::runtime_support::bulk_create_components_action action_type;
        return hpx::async<action_type>(gid, type, count);
    }

    std::vector<naming::id_type> runtime_support::bulk_create_components(
        naming::id_type const& gid, components::component_type type,
        std::size_t count)
    {
        // The following get yields control while the action above
        // is executed and the result is returned to the future
        return bulk_create_components_async(gid, type, count).get();
    }

    ///////////////////////////////////////////////////////////////////////
    /// Create a new memory block using the runtime_support with the
    /// given \a targetgid. This is a non-blocking call. The caller needs
    /// to call \a future#get on the result of this function
    /// to obtain the global id of the newly created object.
    template <typename T, typename Config>
    lcos::future<naming::id_type>
    runtime_support::create_memory_block_async(
        naming::id_type const& id, std::size_t count,
        hpx::actions::manage_object_action<T, Config> const& act)
    {
        if (!naming::is_locality(id))
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "stubs::runtime_support::create_memory_block_async",
                "The id passed as the first argument is not representing"
                    " a locality");
            return make_ready_future(naming::invalid_id);
        }

        // Create a future, execute the required action,
        // we simply return the initialized future, the caller needs
        // to call get() on the return value to obtain the result
        typedef server::runtime_support::create_memory_block_action
            action_type;
        return hpx::async<action_type>(id, count, act);
    }

    template lcos::future<naming::id_type>
    HPX_EXPORT runtime_support::create_memory_block_async<boost::uint8_t, void>(
        naming::id_type const& id, std::size_t count,
        hpx::actions::manage_object_action<boost::uint8_t, void> const& act);

    lcos::future<int>
    runtime_support::load_components_async(naming::id_type const& gid)
    {
        typedef server::runtime_support::load_components_action action_type;
        return hpx::async<action_type>(gid);
    }

    int runtime_support::load_components(naming::id_type const& gid)
    {
        return load_components_async(gid).get();
    }

    lcos::future<void>
    runtime_support::call_startup_functions_async(naming::id_type const& gid,
        bool pre_startup)
    {
        typedef server::runtime_support::call_startup_functions_action action_type;
        return hpx::async<action_type>(gid, pre_startup);
    }

    void runtime_support::call_startup_functions(naming::id_type const& gid,
        bool pre_startup)
    {
        call_startup_functions_async(gid, pre_startup).get();
    }

    lcos::future<void>
    runtime_support::call_shutdown_functions_async(naming::id_type const& gid,
        bool pre_shutdown)
    {
        typedef server::runtime_support::call_shutdown_functions_action action_type;
        return hpx::async<action_type>(gid, pre_shutdown);
    }

    void runtime_support::call_shutdown_functions(naming::id_type const& gid,
        bool pre_shutdown)
    {
        call_shutdown_functions_async(gid, pre_shutdown).get();
    }

    void runtime_support::free_component_sync(agas::gva const& g,
        naming::gid_type const& gid, boost::uint64_t count)
    {
        typedef server::runtime_support::free_component_action action_type;

        // Determine whether the gid of the component to delete is local or
        // remote
        std::pair<bool, components::pinned_ptr> r;

        naming::address addr;
        if (g.prefix == hpx::get_locality() ||
            agas::is_local_address_cached(gid, addr))
        {
            typedef action_type::component_type component_type;
            if (traits::component_supports_migration<component_type>::call())
            {
                r = traits::action_was_object_migrated<action_type>::call(
                    hpx::id_type(gid, hpx::id_type::unmanaged), addr.address_);
                if (!r.first)
                {
                    // apply locally
                    components::server::runtime_support* p =
                        reinterpret_cast<components::server::runtime_support*>(
                            hpx::get_runtime().get_runtime_support_lva());
                    p->free_component(g, gid, count);
                    return;
                }
            }
            else
            {
                // apply locally
                components::server::runtime_support* p =
                    reinterpret_cast<components::server::runtime_support*>(
                        hpx::get_runtime().get_runtime_support_lva());
                p->free_component(g, gid, count);
                return;
            }
        }

        // apply remotely (only if runtime is not stopping)
        naming::id_type id = get_colocation_id_sync(
            naming::id_type(gid, naming::id_type::unmanaged));

        lcos::packaged_action<action_type, void> p;
        lcos::future<void> f = p.get_future();
        p.apply(id, g, gid, count);
        f.get();
    }

    void runtime_support::free_component_locally(agas::gva const& g,
        naming::gid_type const& gid)
    {
        components::server::runtime_support* p =
            reinterpret_cast<components::server::runtime_support*>(
                  get_runtime().get_runtime_support_lva());
        p->free_component(g, gid, 1);
    }

    /// \brief Shutdown the given runtime system
    lcos::future<void>
    runtime_support::shutdown_async(naming::id_type const& targetgid,
        double timeout)
    {
        // Create a promise directly and execute the required action.
        // This action has implemented special response handling as the
        // back-parcel is sent explicitly (and synchronously).
        typedef server::runtime_support::shutdown_action action_type;

        lcos::promise<void> value;
        auto f = value.get_future();

        // We need to make it unmanaged to avoid late refcnt requests
        id_type gid(value.get_id().get_gid(), id_type::unmanaged);
        hpx::apply<action_type>(targetgid, timeout, gid);

        return f;
    }

    void runtime_support::shutdown(naming::id_type const& targetgid,
        double timeout)
    {
        // The following get yields control while the action above
        // is executed and the result is returned to the future
        shutdown_async(targetgid, timeout).get();
    }

    /// \brief Shutdown the runtime systems of all localities
    void runtime_support::shutdown_all(naming::id_type const& targetgid,
        double timeout)
    {
        hpx::apply<server::runtime_support::shutdown_all_action>(
            targetgid, timeout);
    }

    void runtime_support::shutdown_all(double timeout)
    {
        hpx::apply<server::runtime_support::shutdown_all_action>(
            hpx::naming::id_type(
                hpx::applier::get_applier().get_runtime_support_raw_gid(),
                hpx::naming::id_type::unmanaged), timeout);
    }

    ///////////////////////////////////////////////////////////////////////
    /// \brief Retrieve configuration information
    /// \brief Terminate the given runtime system
    lcos::future<void>
    runtime_support::terminate_async(naming::id_type const& targetgid)
    {
        // Create a future directly and execute the required action.
        // This action has implemented special response handling as the
        // back-parcel is sent explicitly (and synchronously).
        typedef server::runtime_support::terminate_action action_type;

        lcos::promise<void> value;
        auto f = value.get_future();

        hpx::apply<action_type>(targetgid, value.get_id());
        return f;
    }

    void runtime_support::terminate(naming::id_type const& targetgid)
    {
        // The following get yields control while the action above
        // is executed and the result is returned to the future
        terminate_async(targetgid).get();
    }

    /// \brief Terminate the runtime systems of all localities
    void runtime_support::terminate_all(naming::id_type const& targetgid)
    {
        hpx::apply<server::runtime_support::terminate_all_action>(
            targetgid);
    }

    void runtime_support::terminate_all()
    {
        hpx::apply<server::runtime_support::terminate_all_action>(
            hpx::naming::id_type(
                hpx::applier::get_applier().get_runtime_support_raw_gid(),
                hpx::naming::id_type::unmanaged));
    }

    ///////////////////////////////////////////////////////////////////////
    void runtime_support::update_agas_cache_entry(
        naming::id_type const& targetgid, naming::gid_type const& gid,
        naming::address const& g, boost::uint64_t count,
        boost::uint64_t offset)
    {
        typedef server::runtime_support::update_agas_cache_entry_action
            action_type;
        hpx::apply<action_type>(targetgid, gid, g, count, offset);
    }

    void runtime_support::update_agas_cache_entry_colocated(
        naming::id_type const& targetgid, naming::gid_type const& gid,
        naming::address const& g, boost::uint64_t count,
        boost::uint64_t offset)
    {
        typedef server::runtime_support::update_agas_cache_entry_action
            action_type;
        hpx::detail::apply_colocated<action_type>(
            targetgid, gid, g, count, offset);
    }

    ///////////////////////////////////////////////////////////////////////
    void runtime_support::garbage_collect_non_blocking(
        naming::id_type const& targetgid)
    {
        typedef server::runtime_support::garbage_collect_action
            action_type;
        hpx::apply<action_type>(targetgid);
    }

    lcos::future<void> runtime_support::garbage_collect_async(
        naming::id_type const& targetgid)
    {
        typedef server::runtime_support::garbage_collect_action
            action_type;
        return hpx::async<action_type>(targetgid);
    }

    void runtime_support::garbage_collect(naming::id_type const& targetgid)
    {
        typedef server::runtime_support::garbage_collect_action
            action_type;
        hpx::async<action_type>(targetgid).get();
    }

    ///////////////////////////////////////////////////////////////////////
    lcos::future<naming::id_type>
    runtime_support::create_performance_counter_async(naming::id_type targetgid,
        performance_counters::counter_info const& info)
    {
        if (!naming::is_locality(targetgid))
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "stubs::runtime_support::create_performance_counter_async",
                "The id passed as the first argument is not representing"
                    " a locality");
            return make_ready_future(naming::invalid_id);
        }

        typedef server::runtime_support::create_performance_counter_action
            action_type;
        return hpx::async<action_type>(targetgid, info);
    }

    naming::id_type
    runtime_support::create_performance_counter(naming::id_type targetgid,
        performance_counters::counter_info const& info,
        error_code& ec)
    {
        return create_performance_counter_async(targetgid, info).get(ec);
    }

    ///////////////////////////////////////////////////////////////////////
    /// \brief Retrieve configuration information
    lcos::future<util::section> runtime_support::get_config_async(
        naming::id_type const& targetgid)
    {
        // Create a future, execute the required action,
        // we simply return the initialized future, the caller needs
        // to call get() on the return value to obtain the result
        typedef server::runtime_support::get_config_action action_type;
        return hpx::async<action_type>(targetgid);
    }

    void runtime_support::get_config(naming::id_type const& targetgid,
        util::section& ini)
    {
        // The following get yields control while the action above
        // is executed and the result is returned to the future
        ini = get_config_async(targetgid).get();
    }

    ///////////////////////////////////////////////////////////////////////
    /// \brief Retrieve instance count for given component type
    lcos::future<boost::int32_t> runtime_support::get_instance_count_async(
        naming::id_type const& targetgid, components::component_type type)
    {
        // Create a future, execute the required action,
        // we simply return the initialized future, the caller needs
        // to call get() on the return value to obtain the result
        typedef server::runtime_support::get_instance_count_action
            action_type;
        return hpx::async<action_type>(targetgid, type);
    }

    boost::int32_t runtime_support::get_instance_count(
        naming::id_type const& targetgid, components::component_type type)
    {
        // The following get yields control while the action above
        // is executed and the result is returned to the future
        return get_instance_count_async(targetgid, type).get();
    }

    ///////////////////////////////////////////////////////////////////////
    void runtime_support::remove_from_connection_cache_async(
        naming::id_type const& target, naming::gid_type const& gid,
        parcelset::endpoints_type const& endpoints)
    {
        typedef server::runtime_support::remove_from_connection_cache_action
            action_type;
        hpx::apply<action_type>(target, gid, endpoints);
    }
}}}

HPX_REGISTER_APPLY_COLOCATED(
    hpx::components::server::runtime_support::update_agas_cache_entry_action
  , hpx_apply_colocated_update_agas_cache_entry_action
)

