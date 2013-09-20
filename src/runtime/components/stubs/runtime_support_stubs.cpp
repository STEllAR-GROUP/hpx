//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/actions/manage_object_action.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/include/async.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/move.hpp>

#include <boost/serialization/vector.hpp>

namespace hpx { namespace components { namespace stubs
{
    /// \brief  The function \a get_factory_properties is used to
    ///         determine, whether instances of the derived component can
    ///         be created in blocks (i.e. more than one instance at once).
    ///         This function is used by the \a distributing_factory to
    ///         determine a correct allocation strategy
    lcos::future<int> runtime_support::get_factory_properties_async(
        naming::id_type const& targetgid, components::component_type type)
    {
        // Create a future, execute the required action,
        // we simply return the initialized future, the caller needs
        // to call get() on the return value to obtain the result
        typedef
            server::runtime_support::factory_properties_action
        action_type;
        return hpx::async<action_type>(targetgid, type);
    }

    int runtime_support::get_factory_properties(naming::id_type const& targetgid,
        components::component_type type)
    {
        // The following get yields control while the action above
        // is executed and the result is returned to the future
        return get_factory_properties_async(targetgid, type).get();
    }

    ///////////////////////////////////////////////////////////////////////
    lcos::future<std::vector<naming::id_type> >
    runtime_support::bulk_create_components_async(
        naming::id_type const& gid, components::component_type type,
        std::size_t count)
    {
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
        // Create a future, execute the required action,
        // we simply return the initialized future, the caller needs
        // to call get() on the return value to obtain the result
        typedef server::runtime_support::create_memory_block_action
            action_type;
        return hpx::async<action_type>(id, count, act);
    }

    lcos::future<bool>
    runtime_support::load_components_async(naming::id_type const& gid)
    {
        typedef server::runtime_support::load_components_action action_type;
        return hpx::async<action_type>(gid);
    }

    bool runtime_support::load_components(naming::id_type const& gid)
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

    void runtime_support::free_component_sync(components::component_type type,
        naming::gid_type const& gid, boost::uint64_t count)
    {
        free_component_sync(type, gid, naming::gid_type(0, count));
    }

    void runtime_support::free_component_sync(components::component_type type,
        naming::gid_type const& gid, naming::gid_type const& count)
    {
        typedef server::runtime_support::free_component_action action_type;

        // Determine whether the gid of the component to delete is local or
        // remote
        //naming::resolver_client& agas = appl.get_agas_client();
        if (/*agas.is_bootstrap() || */agas::is_local_address(gid)) {
            // apply locally
            applier::detail::apply_helper<action_type>::call(
                applier::get_applier().get_runtime_support_raw_gid().get_lsb(),
                threads::thread_priority_default,
                util::forward_as_tuple(type, gid, count));
        }
        else {
            // apply remotely
            naming::id_type id = get_colocation_id(
                naming::id_type(gid, naming::id_type::unmanaged));

            lcos::packaged_action<action_type, void> p;
            p.apply(launch::async, id, type, gid, count);
            p.get_future().get();
        }
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
        hpx::apply<action_type>(targetgid, timeout, value.get_gid());
        return value.get_future();
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
        hpx::apply<action_type>(targetgid, value.get_gid());
        return value.get_future();
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
    lcos::future<long> runtime_support::get_instance_count_async(
        naming::id_type const& targetgid, components::component_type type)
    {
        // Create a future, execute the required action,
        // we simply return the initialized future, the caller needs
        // to call get() on the return value to obtain the result
        typedef server::runtime_support::get_instance_count_action
            action_type;
        return hpx::async<action_type>(targetgid, type);
    }

    long runtime_support::get_instance_count(naming::id_type const& targetgid,
        components::component_type type)
    {
        // The following get yields control while the action above
        // is executed and the result is returned to the future
        return get_instance_count_async(targetgid, type).get();
    }

    ///////////////////////////////////////////////////////////////////////
    void runtime_support::call_shutdown_functions_async(
        naming::id_type const& gid, naming::locality const& l)
    {
        typedef server::runtime_support::remove_from_connection_cache_action
            action_type;
        hpx::apply<action_type>(gid, l);
    }
}}}
