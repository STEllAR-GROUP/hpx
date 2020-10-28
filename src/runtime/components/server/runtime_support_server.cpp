//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/filesystem.hpp>
#include <hpx/runtime_distributed.hpp>
#include <hpx/runtime_local/runtime_local.hpp>
#ifdef HPX_HAVE_LIB_MPI_BASE
#include <hpx/modules/mpi_base.hpp>
#endif
#include <hpx/actions/continuation.hpp>
#include <hpx/actions_base/plain_action.hpp>
#include <hpx/async_combinators/wait_all.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/modules/threadmanager.hpp>
#include <hpx/modules/timing.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/prefix/find_prefix.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/components/component_commandline_base.hpp>
#include <hpx/runtime/components/component_startup_shutdown_base.hpp>
#include <hpx/runtime/components/server/component_database.hpp>
#include <hpx/runtime/components/server/create_component.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/runtime_fwd.hpp>
#include <hpx/runtime_configuration/ini.hpp>
#include <hpx/runtime_configuration/static_factory_data.hpp>
#include <hpx/runtime_distributed/find_localities.hpp>
#include <hpx/runtime_local/runtime_local.hpp>
#include <hpx/runtime_local/shutdown_function.hpp>
#include <hpx/runtime_local/startup_function.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/vector.hpp>
#include <hpx/string_util/case_conv.hpp>
#include <hpx/thread_support/unlock_guard.hpp>
#include <hpx/type_support/unused.hpp>
#include <hpx/util/from_string.hpp>

#include <hpx/lcos_local/packaged_task.hpp>
#include <hpx/modules/collectives.hpp>

#include <hpx/assert.hpp>
#include <hpx/command_line_handling/command_line_handling.hpp>
#include <hpx/command_line_handling/parse_command_line.hpp>
#include <hpx/execution_base/this_thread.hpp>

#include <hpx/plugins/binary_filter_factory_base.hpp>
#include <hpx/plugins/message_handler_factory_base.hpp>

#include <boost/tokenizer.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the runtime_support actions
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::load_components_action,
    load_components_action, hpx::actions::load_components_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::call_startup_functions_action,
    call_startup_functions_action,
    hpx::actions::call_startup_functions_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::call_shutdown_functions_action,
    call_shutdown_functions_action,
    hpx::actions::call_shutdown_functions_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::shutdown_action, shutdown_action,
    hpx::actions::shutdown_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::shutdown_all_action,
    shutdown_all_action, hpx::actions::shutdown_all_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::terminate_action,
    terminate_action, hpx::actions::terminate_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::terminate_all_action,
    terminate_all_action, hpx::actions::terminate_all_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::get_config_action,
    get_config_action, hpx::actions::get_config_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::garbage_collect_action,
    garbage_collect_action, hpx::actions::garbage_collect_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::create_performance_counter_action,
    create_performance_counter_action,
    hpx::actions::create_performance_counter_action_id)
HPX_REGISTER_ACTION_ID(hpx::components::server::runtime_support::
                           remove_from_connection_cache_action,
    remove_from_connection_cache_action,
    hpx::actions::remove_from_connection_cache_action_id)
#if defined(HPX_HAVE_NETWORKING)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::dijkstra_termination_action,
    dijkstra_termination_action, hpx::actions::dijkstra_termination_action_id)
#endif

///////////////////////////////////////////////////////////////////////////////
HPX_DEFINE_COMPONENT_NAME(
    hpx::components::server::runtime_support, hpx_runtime_support);
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(hpx::components::server::runtime_support,
    hpx::components::component_runtime_support)

namespace hpx {
    // helper function to stop evaluating counters during shutdown
    void stop_evaluating_counters(bool terminate = false);
}    // namespace hpx

namespace hpx { namespace components {
    bool initial_static_loading = true;

    ///////////////////////////////////////////////////////////////////////////
    // There is no need to protect these global from thread concurrent access
    // as they are access during early startup only.
    std::vector<static_factory_load_data_type>& get_static_module_data()
    {
        static std::vector<static_factory_load_data_type>
            global_module_init_data;
        return global_module_init_data;
    }

    void init_registry_module(static_factory_load_data_type const& data)
    {
        if (initial_static_loading)
            get_static_module_data().push_back(data);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::map<std::string, util::plugin::get_plugins_list_type>&
    get_static_factory_data()
    {
        static std::map<std::string, util::plugin::get_plugins_list_type>
            global_factory_init_data;
        return global_factory_init_data;
    }

    void init_registry_factory(static_factory_load_data_type const& data)
    {
        if (initial_static_loading)
            get_static_factory_data().insert(
                std::make_pair(data.name, data.get_factory));
    }

    bool get_static_factory(
        std::string const& instance, util::plugin::get_plugins_list_type& f)
    {
        typedef std::map<std::string, util::plugin::get_plugins_list_type>
            map_type;

        map_type const& m = get_static_factory_data();
        map_type::const_iterator it = m.find(instance);
        if (it == m.end())
            return false;

        f = it->second;
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    std::map<std::string, util::plugin::get_plugins_list_type>&
    get_static_commandline_data()
    {
        static std::map<std::string, util::plugin::get_plugins_list_type>
            global_commandline_init_data;
        return global_commandline_init_data;
    }

    void init_registry_commandline(static_factory_load_data_type const& data)
    {
        if (initial_static_loading)
            get_static_commandline_data().insert(
                std::make_pair(data.name, data.get_factory));
    }

    bool get_static_commandline(
        std::string const& instance, util::plugin::get_plugins_list_type& f)
    {
        typedef std::map<std::string, util::plugin::get_plugins_list_type>
            map_type;

        map_type const& m = get_static_commandline_data();
        map_type::const_iterator it = m.find(instance);
        if (it == m.end())
            return false;

        f = it->second;
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    std::map<std::string, util::plugin::get_plugins_list_type>&
    get_static_startup_shutdown_data()
    {
        static std::map<std::string, util::plugin::get_plugins_list_type>
            global_startup_shutdown_init_data;
        return global_startup_shutdown_init_data;
    }

    void init_registry_startup_shutdown(
        static_factory_load_data_type const& data)
    {
        if (initial_static_loading)
            get_static_startup_shutdown_data().insert(
                std::make_pair(data.name, data.get_factory));
    }

    bool get_static_startup_shutdown(
        std::string const& instance, util::plugin::get_plugins_list_type& f)
    {
        typedef std::map<std::string, util::plugin::get_plugins_list_type>
            map_type;

        map_type const& m = get_static_startup_shutdown_data();
        map_type::const_iterator it = m.find(instance);
        if (it == m.end())
            return false;

        f = it->second;
        return true;
    }
}}    // namespace hpx::components

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server {
    ///////////////////////////////////////////////////////////////////////////
    runtime_support::runtime_support(hpx::util::runtime_configuration& cfg)
      : stop_called_(false)
      , stop_done_(false)
      , terminated_(false)
      , main_thread_id_(std::this_thread::get_id())
      , shutdown_all_invoked_(false)
#if defined(HPX_HAVE_NETWORKING)
      , dijkstra_mtx_()
      , dijkstra_cond_()
      , dijkstra_color_(false)
#endif
      , p_mtx_()
      , plugins_()
      , modules_(cfg.modules())
      , static_modules_()
    {
    }

    // function to be called during shutdown
    // Action: shut down this runtime system instance
    void runtime_support::shutdown(
        double timeout, naming::id_type const& respond_to)
    {
        // initiate system shutdown
        stop(timeout, respond_to, false);
    }

    // function to be called to terminate this locality immediately
    void runtime_support::terminate(naming::id_type const& respond_to)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        // push pending logs
        components::cleanup_logging();

        if (respond_to)
        {
            // respond synchronously
            using void_lco_type = lcos::base_lco_with_value<void>;
            using action_type = void_lco_type::set_event_action;

            naming::address addr;
            if (agas::is_local_address_cached(respond_to, addr))
            {
                // execute locally, action is executed immediately as it is
                // a direct_action
                hpx::applier::detail::apply_l<action_type>(
                    respond_to, std::move(addr));
            }
#if defined(HPX_HAVE_NETWORKING)
            else
            {
                // apply remotely, parcel is sent synchronously
                hpx::applier::detail::apply_r_sync<action_type>(
                    std::move(addr), respond_to);
            }
#endif
        }
#else
        HPX_ASSERT(false);
        HPX_UNUSED(respond_to);
#endif
        std::abort();
    }
}}}    // namespace hpx::components::server

///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_COMPUTE_DEVICE_CODE)
typedef hpx::components::server::runtime_support::call_shutdown_functions_action
    call_shutdown_functions_action;
typedef hpx::lcos::detail::make_broadcast_action<
    call_shutdown_functions_action>::type
    call_shutdown_functions_broadcast_action;
#endif

HPX_ACTION_USES_MEDIUM_STACK(call_shutdown_functions_broadcast_action)

HPX_REGISTER_BROADCAST_ACTION_DECLARATION(
    call_shutdown_functions_action, call_shutdown_functions_action)
HPX_REGISTER_BROADCAST_ACTION_ID(call_shutdown_functions_action,
    call_shutdown_functions_action,
    hpx::actions::broadcast_call_shutdown_functions_action_id)

namespace hpx { namespace components { namespace server {
    ///////////////////////////////////////////////////////////////////////////
    // initiate system shutdown for all localities
    void invoke_shutdown_functions(
        std::vector<naming::id_type> const& localities, bool pre_shutdown)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        call_shutdown_functions_action act;
        lcos::broadcast(act, localities, pre_shutdown).get();
#else
        HPX_ASSERT(false);
        HPX_UNUSED(localities);
        HPX_UNUSED(pre_shutdown);
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_NETWORKING)
    void runtime_support::dijkstra_make_black()
    {
        // Rule 1: A machine sending a message makes itself black.
        std::lock_guard<dijkstra_mtx_type> l(dijkstra_mtx_);
        dijkstra_color_ = true;
    }

    void runtime_support::send_dijkstra_termination_token(
        std::uint32_t target_locality_id, std::uint32_t initiating_locality_id,
        std::uint32_t num_localities, bool dijkstra_token)
    {
        // First wait for this locality to become passive. We do this by
        // periodically checking the number of still running threads.
        //
        // Rule 0: When active, machine nr.i + 1 keeps the token; when passive,
        // it hands over the token to machine nr.i.
        applier::applier& appl = hpx::applier::get_applier();
        threads::threadmanager& tm = appl.get_thread_manager();

        util::yield_while(
            [&tm]() -> bool {
                tm.cleanup_terminated(true);
                return tm.get_thread_count() >
                    std::int64_t(1) + tm.get_background_thread_count();
            },
            "runtime_support::dijkstra_termination", false);

        // Now this locality has become passive, thus we can send the token
        // to the next locality.
        //
        // Rule 2: When machine nr.i + 1 propagates the probe, it hands over a
        // black token to machine nr.i if it is black itself, whereas while
        // being white it leaves the color of the token unchanged.
        {
            std::lock_guard<dijkstra_mtx_type> l(dijkstra_mtx_);
            if (dijkstra_color_)
                dijkstra_token = true;

            // Rule 5: Upon transmission of the token to machine nr.i, machine
            // nr.i + 1 becomes white.
            dijkstra_color_ = false;
        }

#if !defined(HPX_COMPUTE_DEVICE_CODE)
        naming::id_type id(naming::get_id_from_locality_id(target_locality_id));
        apply<dijkstra_termination_action>(
            id, initiating_locality_id, num_localities, dijkstra_token);
#else
        HPX_ASSERT(false);
        HPX_UNUSED(target_locality_id);
        HPX_UNUSED(initiating_locality_id);
        HPX_UNUSED(num_localities);
        HPX_UNUSED(dijkstra_token);
#endif
    }

    // invoked during termination detection
    void runtime_support::dijkstra_termination(
        std::uint32_t initiating_locality_id, std::uint32_t num_localities,
        bool dijkstra_token)
    {
        applier::applier& appl = hpx::applier::get_applier();
        naming::resolver_client& agas_client = naming::get_agas_client();

        agas_client.start_shutdown();

        parcelset::parcelhandler& ph = appl.get_parcel_handler();
        ph.flush_parcels();

        std::uint32_t locality_id = get_locality_id();

        if (initiating_locality_id == locality_id)
        {
            // we received the token after a full circle
            if (dijkstra_token)
            {
                std::lock_guard<dijkstra_mtx_type> l(dijkstra_mtx_);
                dijkstra_color_ = true;    // unsuccessful termination
            }

            dijkstra_cond_.notify_one();
            return;
        }

        if (0 == locality_id)
            locality_id = num_localities;

        send_dijkstra_termination_token(locality_id - 1, initiating_locality_id,
            num_localities, dijkstra_token);
    }
#endif

    // Kick off termination detection, this is modeled after Dijkstra's paper:
    // http://www.cs.mcgill.ca/~lli22/575/termination3.pdf.
    std::size_t runtime_support::dijkstra_termination_detection(
        std::vector<naming::id_type> const& locality_ids)
    {
#if defined(HPX_HAVE_NETWORKING)
        std::uint32_t num_localities =
            static_cast<std::uint32_t>(locality_ids.size());
        if (num_localities == 1)
#endif

        {
            // While no real distributed termination detection has to be
            // performed, we should still wait for the thread-queues to drain.
            applier::applier& appl = hpx::applier::get_applier();
            threads::threadmanager& tm = appl.get_thread_manager();

            util::yield_while(
                [&tm]() -> bool {
                    tm.cleanup_terminated(true);
                    return tm.get_thread_count() >
                        std::int64_t(1) + tm.get_background_thread_count();
                },
                "runtime_support::dijkstra_termination", false);

            HPX_UNUSED(locality_ids);
            return 0;
        }

#if defined(HPX_HAVE_NETWORKING)
        std::uint32_t initiating_locality_id = get_locality_id();

        // send token to previous node
        std::uint32_t target_id = initiating_locality_id;
        if (0 == target_id)
            target_id = num_localities;

        std::size_t count = 0;    // keep track of number of trials

        {
            // Note: we protect the entire loop here since the stopping condition
            // depends on the shared variable "dijkstra_color_"
            // Proper unlocking for possible remote actions needs to be taken care of
            typedef std::unique_lock<dijkstra_mtx_type> dijkstra_scoped_lock;
            dijkstra_scoped_lock l(dijkstra_mtx_);
            do
            {
                // Rule 4: Machine nr.0 initiates a probe by making itself white
                // and sending a white token to machine nr.N - 1.
                dijkstra_color_ = false;    // start off with white

                {
                    util::unlock_guard<dijkstra_scoped_lock> ul(l);
                    send_dijkstra_termination_token(target_id - 1,
                        initiating_locality_id, num_localities,
                        dijkstra_color_);
                }

                // wait for token to come back to us
                dijkstra_cond_.wait(l);

                // Rule 3: After the completion of an unsuccessful probe, machine
                // nr.0 initiates a next probe.

                ++count;

            } while (dijkstra_color_);
        }

        return count;
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime_support::shutdown_all(double timeout)
    {
        if (find_here() != hpx::find_root_locality())
        {
            HPX_THROW_EXCEPTION(invalid_status, "runtime_support::shutdown_all",
                "shutdown_all should be invoked on the root locality only");
            return;
        }

        // make sure shutdown_all is invoked only once
        bool flag = false;
        if (!shutdown_all_invoked_.compare_exchange_strong(flag, true))
        {
            return;
        }

        LRT_(info) << "runtime_support::shutdown_all: "
                      "initializing application shutdown";

        applier::applier& appl = hpx::applier::get_applier();
        naming::resolver_client& agas_client = naming::get_agas_client();

        agas_client.start_shutdown();

        stop_evaluating_counters(true);

        // wake up suspended pus
        threads::threadmanager& tm = appl.get_thread_manager();
        tm.resume();

        std::vector<naming::id_type> locality_ids = find_all_localities();
        std::size_t count = dijkstra_termination_detection(locality_ids);

        LRT_(info) << "runtime_support::shutdown_all: "    //-V128
                      "passed first termination detection (count: "
                   << count << ").";

        // execute registered shutdown functions on all localities
        invoke_shutdown_functions(locality_ids, true);
        invoke_shutdown_functions(locality_ids, false);

        LRT_(info) << "runtime_support::shutdown_all: "
                      "invoked shutdown functions";

        // Do a second round of termination detection to synchronize with all
        // work that was triggered by the invocation of the shutdown
        // functions.
        count = dijkstra_termination_detection(locality_ids);

        LRT_(info) << "runtime_support::shutdown_all: "    //-V128
                      "passed second termination detection (count: "
                   << count << ").";

        // Shut down all localities except the local one, we can't use
        // broadcast here as we have to handle the back parcel in a special
        // way.
        std::reverse(locality_ids.begin(), locality_ids.end());
        std::uint32_t locality_id = get_locality_id();
        std::vector<lcos::future<void>> lazy_actions;

        for (naming::id_type const& id : locality_ids)
        {
            if (locality_id != naming::get_locality_id_from_id(id))
            {
                using components::stubs::runtime_support;
                lazy_actions.push_back(
                    runtime_support::shutdown_async(id, timeout));
            }
        }

        // wait for all localities to be stopped
        wait_all(lazy_actions);

        LRT_(info) << "runtime_support::shutdown_all: "
                      "all localities have been shut down";

        // Now make sure this local locality gets shut down as well.
        // There is no need to respond...
        stop(timeout, naming::invalid_id, false);
    }

    ///////////////////////////////////////////////////////////////////////////
    // initiate system shutdown for all localities
    void runtime_support::terminate_all()
    {
        std::vector<naming::gid_type> locality_ids;
        naming::get_agas_client().get_localities(locality_ids);
        std::reverse(locality_ids.begin(), locality_ids.end());

        // Terminate all localities except the local one, we can't use
        // broadcast here as we have to handle the back parcel in a special
        // way.
        {
            std::uint32_t locality_id = get_locality_id();
            std::vector<lcos::future<void>> lazy_actions;

            for (naming::gid_type gid : locality_ids)
            {
                if (locality_id != naming::get_locality_id_from_gid(gid))
                {
                    using components::stubs::runtime_support;
                    naming::id_type id(gid, naming::id_type::unmanaged);
                    lazy_actions.push_back(
                        runtime_support::terminate_async(id));
                }
            }

            // wait for all localities to be stopped
            wait_all(lazy_actions);
        }

        // now make sure this local locality gets terminated as well.
        terminate(naming::invalid_id);    //good night
    }

    ///////////////////////////////////////////////////////////////////////////
    // Retrieve configuration information
    util::section runtime_support::get_config()
    {
        return *(get_runtime().get_config().get_section("application"));
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Force a garbage collection operation in the AGAS layer.
    void runtime_support::garbage_collect()
    {
        naming::get_agas_client().garbage_collect_non_blocking();
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create the given performance counter instance.
    naming::gid_type runtime_support::create_performance_counter(
        performance_counters::counter_info const& info)
    {
        return performance_counters::detail::create_counter_local(info);
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime_support::delete_function_lists()
    {
        pre_startup_functions_.clear();
        startup_functions_.clear();
        pre_shutdown_functions_.clear();
        shutdown_functions_.clear();
    }

    void runtime_support::tidy()
    {
        // Only after releasing the components we are allowed to release
        // the modules. This is done in reverse order of loading.
        plugins_.clear();    // unload all plugins
        modules_.clear();    // unload all modules
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Remove the given locality from our connection cache
    void runtime_support::remove_from_connection_cache(
        naming::gid_type const& gid, parcelset::endpoints_type const& eps)
    {
        runtime_distributed* rt = get_runtime_distributed_ptr();
        if (rt == nullptr)
            return;

#if defined(HPX_HAVE_NETWORKING)
        // instruct our connection cache to drop all connections it is holding
        rt->get_parcel_handler().remove_from_connection_cache(gid, eps);
#else
        HPX_UNUSED(gid);
        HPX_UNUSED(eps);
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime_support::run()
    {
        std::unique_lock<std::mutex> l(mtx_);
        stop_called_ = false;
        stop_done_ = false;
        terminated_ = false;
        shutdown_all_invoked_.store(false);
    }

    void runtime_support::wait()
    {
        std::unique_lock<std::mutex> l(mtx_);
        while (!stop_done_)
        {
            LRT_(info) << "runtime_support: about to enter wait state";
            wait_condition_.wait(l);
            LRT_(info) << "runtime_support: exiting wait state";
        }
    }

    void runtime_support::stop(double timeout,
        naming::id_type const& respond_to, bool remove_from_remote_caches)
    {
        std::unique_lock<std::mutex> l(mtx_);
        if (!stop_called_)
        {
            // push pending logs
            components::cleanup_logging();

            HPX_ASSERT(!terminated_);

            applier::applier& appl = hpx::applier::get_applier();
            threads::threadmanager& tm = appl.get_thread_manager();
            naming::resolver_client& agas_client = naming::get_agas_client();

            hpx::chrono::high_resolution_timer t;
            double start_time = t.elapsed();
            bool timed_out = false;
            error_code ec(lightweight);

            stop_called_ = true;

            {
                util::unlock_guard<std::mutex> ul(mtx_);

                util::yield_while(
                    [&tm, timeout, &t, start_time, &timed_out]() -> bool {
                        tm.cleanup_terminated(true);

                        if (timeout >= 0.0 &&
                            timeout < (t.elapsed() - start_time))
                        {
                            timed_out = true;
                            return false;
                        }

                        return tm.get_thread_count() >
                            std::int64_t(1) + tm.get_background_thread_count();
                    },
                    "runtime_support::stop", false);

                // If it took longer than expected, kill all suspended threads as
                // well.
                if (timed_out)
                {
                    // now we have to wait for all threads to be aborted
                    start_time = t.elapsed();

                    util::yield_while(
                        [&tm, timeout, &t, start_time]() -> bool {
                            tm.abort_all_suspended_threads();
                            tm.cleanup_terminated(true);

                            if (timeout >= 0.0 &&
                                timeout < (t.elapsed() - start_time))
                            {
                                return false;
                            }

                            return tm.get_thread_count() > std::int64_t(1) +
                                tm.get_background_thread_count();
                        },
                        "runtime_support::dijkstra_termination", false);
                }

                // Drop the locality from the partition table.
                naming::gid_type here = agas_client.get_local_locality();

                // unregister fixed components
                agas_client.unbind_local(
                    appl.get_runtime_support_raw_gid(), ec);

                if (remove_from_remote_caches)
                    remove_here_from_connection_cache();

                agas_client.unregister_locality(here, ec);

                if (remove_from_remote_caches)
                    remove_here_from_console_connection_cache();

                if (respond_to)
                {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#if defined(HPX_HAVE_NETWORKING)
                    // respond synchronously
                    using void_lco_type = lcos::base_lco_with_value<void>;
                    using action_type = void_lco_type::set_event_action;
#endif
#else
                    HPX_ASSERT(false);
#endif

                    naming::address addr;
                    if (agas::is_local_address_cached(respond_to, addr))
                    {
                        // this should never happen
                        HPX_ASSERT(false);
                    }
#if defined(HPX_HAVE_NETWORKING)
                    else
                    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
                        // apply remotely, parcel is sent synchronously
                        hpx::applier::detail::apply_r_sync<action_type>(
                            std::move(addr), respond_to);
#else
                        HPX_ASSERT(false);
#endif
                    }
#endif
                }
            }

            stop_done_ = true;
            wait_condition_.notify_all();

            // The main thread notifies stop_condition_, so don't wait if we're
            // on the main thread.
            if (std::this_thread::get_id() != main_thread_id_)
            {
                stop_condition_.wait(l);    // wait for termination
            }
        }
    }

    void runtime_support::notify_waiting_main()
    {
        std::unique_lock<std::mutex> l(mtx_);
        if (!stop_called_)
        {
            stop_called_ = true;
            stop_done_ = true;
            wait_condition_.notify_all();

            // The main thread notifies stop_condition_, so don't wait if we're
            // on the main thread.
            if (std::this_thread::get_id() != main_thread_id_)
            {
                stop_condition_.wait(l);    // wait for termination
            }
        }
    }

    // this will be called after the thread manager has exited
    void runtime_support::stopped()
    {
        std::lock_guard<std::mutex> l(mtx_);
        if (!terminated_)
        {
            terminated_ = true;
            stop_condition_.notify_all();    // finished cleanup/termination
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    inline void decode(std::string& str, char const* s, char const* r)
    {
        std::string::size_type pos = 0;
        while ((pos = str.find(s, pos)) != std::string::npos)
        {
            str.replace(pos, 2, r);
        }
    }

    inline std::string decode_string(std::string str)
    {
        decode(str, "\\n", "\n");
        return str;
    }

    namespace detail {
        void handle_print_bind(
            hpx::program_options::variables_map const& /* vm_ */,
            std::size_t num_threads)
        {
            threads::topology& top = threads::create_topology();
            auto const& rp = hpx::resource::get_partitioner();
            auto const& tm = get_runtime().get_thread_manager();
            {
                std::ostringstream
                    strm;    // make sure all output is kept together

                strm << std::string(79, '*') << '\n';
                strm << "locality: " << hpx::get_locality_id() << '\n';
                for (std::size_t i = 0; i != num_threads; ++i)
                {
                    // print the mask for the current PU
                    threads::mask_cref_type pu_mask = rp.get_pu_mask(i);

                    if (!threads::any(pu_mask))
                    {
                        strm << std::setw(4) << i    //-V112
                             << ": thread binding disabled" << std::endl;
                    }
                    else
                    {
                        std::string pool_name = tm.get_pool(i).get_pool_name();
                        top.print_affinity_mask(strm, i, pu_mask, pool_name);
                    }

                    // Make sure the mask does not contradict the CPU bindings
                    // returned by the system (see #973: Would like option to
                    // report HWLOC bindings).
                    error_code ec(lightweight);
                    std::thread& blob = tm.get_os_thread_handle(i);
                    threads::mask_type boundcpu =
                        top.get_cpubind_mask(blob, ec);

                    /* threads::mask_type boundcpu = top.get_cpubind_mask(
                    rt.get_thread_manager().get_os_thread_handle(i), ec);*/

                    // The masks reported by HPX must be the same as the ones
                    // reported from HWLOC.
                    if (!ec && threads::any(boundcpu) &&
                        !threads::equal(boundcpu, pu_mask, num_threads))
                    {
                        std::string boundcpu_str = threads::to_string(boundcpu);
                        std::string pu_mask_str = threads::to_string(pu_mask);
                        HPX_THROW_EXCEPTION(invalid_status, "handle_print_bind",
                            hpx::util::format(
                                "unexpected mismatch between locality {1}: "
                                "binding "
                                "reported from HWLOC({2}) and HPX({3}) on "
                                "thread {4}",
                                hpx::get_locality_id(), boundcpu_str,
                                pu_mask_str, i));
                    }
                }

                std::cout << strm.str();
            }
        }

#if defined(HPX_HAVE_NETWORKING)
        void handle_list_parcelports()
        {
            {
                std::ostringstream
                    strm;    // make sure all output is kept together
                strm << std::string(79, '*') << '\n';
                strm << "locality: " << hpx::get_locality_id() << '\n';

                get_runtime_distributed().get_parcel_handler().list_parcelports(
                    strm);

                std::cout << strm.str();
            }
        }
#endif
    }    // namespace detail

    int runtime_support::load_components()
    {
        // load components now that AGAS is up
        util::runtime_configuration& ini = get_runtime().get_config();

        // first static components
        ini.load_components_static(components::get_static_module_data());

        // modules loaded dynamically should not register themselves statically
        components::initial_static_loading = false;

        // make sure every component module gets asked for startup/shutdown
        // functions only once
        std::set<std::string> startup_handled;

        // collect additional command-line options
        hpx::program_options::options_description options;

        // then dynamic ones
        naming::resolver_client& client =
            get_runtime_distributed().get_agas_client();
        int result = load_components(
            ini, client.get_local_locality(), client, options, startup_handled);

        if (!load_plugins(ini, options, startup_handled))
            result = -2;

        // do secondary command line processing, check validity of options only
        try
        {
            std::string unknown_cmd_line(
                ini.get_entry("hpx.unknown_cmd_line", ""));
            if (!unknown_cmd_line.empty())
            {
                std::string runtime_mode(ini.get_entry("hpx.runtime_mode", ""));
                hpx::program_options::variables_map vm;

                util::commandline_error_mode mode = util::rethrow_on_error;
                std::string allow_unknown(
                    ini.get_entry("hpx.commandline.allow_unknown", "0"));
                if (allow_unknown != "0")
                    mode = util::allow_unregistered;

                std::vector<std::string> still_unregistered_options;
                util::parse_commandline(ini, options, unknown_cmd_line, vm,
                    std::size_t(-1), mode,
                    get_runtime_mode_from_name(runtime_mode), nullptr,
                    &still_unregistered_options);

                std::string still_unknown_commandline;
                for (std::size_t i = 1; i < still_unregistered_options.size();
                     ++i)
                {
                    if (i != 1)
                    {
                        still_unknown_commandline += " ";
                    }
                    still_unknown_commandline +=
                        util::detail::enquote(still_unregistered_options[i]);
                }

                if (!still_unknown_commandline.empty())
                {
                    util::section* s = ini.get_section("hpx");
                    HPX_ASSERT(s != nullptr);
                    s->add_entry(
                        "unknown_cmd_line_option", still_unknown_commandline);
                }
            }

            std::string fullhelp(ini.get_entry("hpx.cmd_line_help", ""));
            if (!fullhelp.empty())
            {
                std::string help_option(
                    ini.get_entry("hpx.cmd_line_help_option", ""));
                if (0 == std::string("full").find(help_option))
                {
                    std::cout << decode_string(fullhelp);
                    std::cout << options << std::endl;
                }
                else
                {
                    throw hpx::detail::command_line_error(
                        "unknown help option: " + help_option);
                }
                return 1;
            }

            // secondary command line handling, looking for --exit and other
            // options
            std::string cmd_line(ini.get_entry("hpx.cmd_line", ""));
            if (!cmd_line.empty())
            {
                std::string runtime_mode(ini.get_entry("hpx.runtime_mode", ""));
                hpx::program_options::variables_map vm;

                util::parse_commandline(ini, options, cmd_line, vm,
                    std::size_t(-1),
                    util::allow_unregistered | util::report_missing_config_file,
                    get_runtime_mode_from_name(runtime_mode));

                if (vm.count("hpx:print-bind"))
                {
                    std::size_t num_threads =
                        hpx::util::from_string<std::size_t>(
                            ini.get_entry("hpx.os_threads", 1));
                    detail::handle_print_bind(vm, num_threads);
                }

#if defined(HPX_HAVE_NETWORKING)
                if (vm.count("hpx:list-parcel-ports"))
                    detail::handle_list_parcelports();
#endif
                if (vm.count("hpx:exit"))
                    return 1;
            }
        }
        catch (std::exception const& e)
        {
            std::cerr << "runtime_support::load_components: "
                      << "command line processing: " << e.what() << std::endl;
            return -1;
        }

        return result;
    }

    void runtime_support::call_startup_functions(bool pre_startup)
    {
        if (pre_startup)
        {
            get_runtime().set_state(state_pre_startup);
            for (startup_function_type& f : pre_startup_functions_)
            {
                f();
            }
        }
        else
        {
            get_runtime().set_state(state_startup);
            for (startup_function_type& f : startup_functions_)
            {
                f();
            }
        }
    }

    void runtime_support::call_shutdown_functions(bool pre_shutdown)
    {
        runtime& rt = get_runtime();
        if (pre_shutdown)
        {
            rt.set_state(state_pre_shutdown);
            for (shutdown_function_type& f : pre_shutdown_functions_)
            {
                try
                {
                    f();
                }
                catch (...)
                {
                    rt.report_error(std::current_exception());
                }
            }
        }
        else
        {
            rt.set_state(state_shutdown);
            for (shutdown_function_type& f : shutdown_functions_)
            {
                try
                {
                    f();
                }
                catch (...)
                {
                    rt.report_error(std::current_exception());
                }
            }
            lcos::barrier::get_global_barrier().detach();
        }
    }

    // working around non-copy-ability of packaged_task
    struct indirect_packaged_task
    {
        typedef lcos::local::packaged_task<void()> packaged_task_type;

        indirect_packaged_task()
          : pt(std::make_shared<packaged_task_type>([]() {}))
        {
        }

        hpx::future<void> get_future()
        {
            return pt->get_future();
        }

        template <typename... Ts>
        void operator()(Ts&&... /* vs */)
        {
            // This needs to be run on a HPX thread
            hpx::apply(std::move(*pt));
            pt.reset();
        }

        std::shared_ptr<packaged_task_type> pt;
    };

    void runtime_support::remove_here_from_connection_cache()
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#if defined(HPX_HAVE_NETWORKING)
        runtime_distributed* rtd = get_runtime_distributed_ptr();
        if (rtd == nullptr)
            return;

        std::vector<naming::id_type> locality_ids = find_remote_localities();

        typedef server::runtime_support::remove_from_connection_cache_action
            action_type;

        std::vector<future<void>> callbacks;
        callbacks.reserve(locality_ids.size());

        action_type act;
        for (naming::id_type const& id : locality_ids)
        {
            // console is handled separately
            if (naming::get_locality_id_from_id(id) == 0)
                continue;

            indirect_packaged_task ipt;
            callbacks.push_back(ipt.get_future());
            apply_cb(
                act, id, std::move(ipt), hpx::get_locality(), rtd->endpoints());
        }

        wait_all(callbacks);
#endif
#else
        HPX_ASSERT(false);
#endif
    }

    void runtime_support::remove_here_from_console_connection_cache()
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#if defined(HPX_HAVE_NETWORKING)
        runtime_distributed* rtd = get_runtime_distributed_ptr();
        if (rtd == nullptr)
            return;

        typedef server::runtime_support::remove_from_connection_cache_action
            action_type;

        action_type act;
        indirect_packaged_task ipt;
        future<void> callback = ipt.get_future();

        // handle console separately
        id_type id = naming::get_id_from_locality_id(0);
        apply_cb(
            act, id, std::move(ipt), hpx::get_locality(), rtd->endpoints());

        callback.wait();
#endif
#else
        HPX_ASSERT(false);
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_NETWORKING)
    void runtime_support::register_message_handler(
        char const* message_handler_type, char const* action, error_code& ec)
    {
        // locate the factory for the requested plugin type
        typedef std::unique_lock<plugin_map_mutex_type> plugin_map_scoped_lock;
        plugin_map_scoped_lock l(p_mtx_);

        plugin_map_type::const_iterator it =
            plugins_.find(message_handler_type);
        if (it == plugins_.end() || !(*it).second.first)
        {
            l.unlock();
            if (ec.category() != hpx::get_lightweight_hpx_category())
            {
                // we don't know anything about this component
                std::ostringstream strm;
                strm << "attempt to create message handler plugin instance of "
                        "invalid/unknown type: "
                     << message_handler_type;
                HPX_THROWS_IF(ec, hpx::bad_plugin_type,
                    "runtime_support::create_message_handler", strm.str());
            }
            else
            {
                // lightweight error handling
                HPX_THROWS_IF(ec, hpx::bad_plugin_type,
                    "runtime_support::create_message_handler",
                    "attempt to create message handler plugin instance of "
                    "invalid/unknown type");
            }
            return;
        }

        l.unlock();

        // create new component instance
        std::shared_ptr<plugins::message_handler_factory_base> factory(
            std::static_pointer_cast<plugins::message_handler_factory_base>(
                (*it).second.first));

        factory->register_action(action, ec);

        if (ec)
        {
            std::ostringstream strm;
            strm << "couldn't register action '" << action
                 << "' for message handler plugin of type: "
                 << message_handler_type;
            HPX_THROWS_IF(ec, hpx::bad_plugin_type,
                "runtime_support::register_message_handler", strm.str());
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();

        // log result if requested
        LRT_(info) << "successfully registered message handler plugin of type: "
                   << message_handler_type;
    }

    parcelset::policies::message_handler*
    runtime_support::create_message_handler(char const* message_handler_type,
        char const* action, parcelset::parcelport* pp, std::size_t num_messages,
        std::size_t interval, error_code& ec)
    {
        // locate the factory for the requested plugin type
        typedef std::unique_lock<plugin_map_mutex_type> plugin_map_scoped_lock;
        plugin_map_scoped_lock l(p_mtx_);

        plugin_map_type::const_iterator it =
            plugins_.find(message_handler_type);
        if (it == plugins_.end() || !(*it).second.first)
        {
            l.unlock();
            if (ec.category() != hpx::get_lightweight_hpx_category())
            {
                // we don't know anything about this component
                std::ostringstream strm;
                strm << "attempt to create message handler plugin instance of "
                        "invalid/unknown type: "
                     << message_handler_type;
                HPX_THROWS_IF(ec, hpx::bad_plugin_type,
                    "runtime_support::create_message_handler", strm.str());
            }
            else
            {
                // lightweight error handling
                HPX_THROWS_IF(ec, hpx::bad_plugin_type,
                    "runtime_support::create_message_handler",
                    "attempt to create message handler plugin instance of "
                    "invalid/unknown type");
            }
            return nullptr;
        }

        l.unlock();

        // create new component instance
        std::shared_ptr<plugins::message_handler_factory_base> factory(
            std::static_pointer_cast<plugins::message_handler_factory_base>(
                (*it).second.first));

        parcelset::policies::message_handler* mh =
            factory->create(action, pp, num_messages, interval);
        if (nullptr == mh)
        {
            std::ostringstream strm;
            strm << "couldn't create message handler plugin of type: "
                 << message_handler_type;
            HPX_THROWS_IF(ec, hpx::bad_plugin_type,
                "runtime_support::create_message_handler", strm.str());
            return nullptr;
        }

        if (&ec != &throws)
            ec = make_success_code();

        // log result if requested
        LRT_(info) << "successfully created message handler plugin of type: "
                   << message_handler_type;
        return mh;
    }

    serialization::binary_filter* runtime_support::create_binary_filter(
        char const* binary_filter_type, bool compress,
        serialization::binary_filter* next_filter, error_code& ec)
    {
        // locate the factory for the requested plugin type
        typedef std::unique_lock<plugin_map_mutex_type> plugin_map_scoped_lock;
        plugin_map_scoped_lock l(p_mtx_);

        plugin_map_type::const_iterator it = plugins_.find(binary_filter_type);
        if (it == plugins_.end() || !(*it).second.first)
        {
            l.unlock();
            // we don't know anything about this component
            std::ostringstream strm;
            strm << "attempt to create binary filter plugin instance of "
                    "invalid/unknown type: "
                 << binary_filter_type;
            HPX_THROWS_IF(ec, hpx::bad_plugin_type,
                "runtime_support::create_binary_filter", strm.str());
            return nullptr;
        }

        l.unlock();

        // create new component instance
        std::shared_ptr<plugins::binary_filter_factory_base> factory(
            std::static_pointer_cast<plugins::binary_filter_factory_base>(
                (*it).second.first));

        serialization::binary_filter* bf =
            factory->create(compress, next_filter);
        if (nullptr == bf)
        {
            std::ostringstream strm;
            strm << "couldn't create binary filter plugin of type: "
                 << binary_filter_type;
            HPX_THROWS_IF(ec, hpx::bad_plugin_type,
                "runtime_support::create_binary_filter", strm.str());
            return nullptr;
        }

        if (&ec != &throws)
            ec = make_success_code();

        // log result if requested
        LRT_(info)
            << "successfully created binary filter handler plugin of type: "
            << binary_filter_type;
        return bf;
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    bool runtime_support::load_component_static(util::section& ini,
        std::string const& instance, std::string const& component,
        filesystem::path const& lib, naming::gid_type const& /* prefix */,
        naming::resolver_client& /* agas_client */, bool /* isdefault */,
        bool /* isenabled */,
        hpx::program_options::options_description& options,
        std::set<std::string>& startup_handled)
    {
        try
        {
            // initialize the factory instance using the preferences from the
            // ini files
            util::section const* glob_ini = nullptr;
            if (ini.has_section("settings"))
                glob_ini = ini.get_section("settings");

            util::section const* component_ini = nullptr;
            std::string component_section("hpx.components." + instance);
            if (ini.has_section(component_section))
                component_ini = ini.get_section(component_section);

            if (nullptr == component_ini ||
                "0" == component_ini->get_entry("no_factory", "0"))
            {
                util::plugin::get_plugins_list_type f;
                if (!components::get_static_factory(instance, f))
                {
                    LRT_(warning) << "static loading failed: " << lib.string()
                                  << ": " << instance << ": couldn't find "
                                  << "factory in global static factory map";
                    return false;
                }

                LRT_(info) << "static loading succeeded: " << lib.string()
                           << ": " << instance;
            }

            // make sure startup/shutdown registration is called once for each
            // module, same for plugins
            if (startup_handled.find(component) == startup_handled.end())
            {
                error_code ec(lightweight);
                startup_handled.insert(component);
                load_commandline_options_static(component, options, ec);
                if (ec)
                    ec = error_code(lightweight);
                load_startup_shutdown_functions_static(component, ec);
            }
        }
        catch (hpx::exception const&)
        {
            throw;
        }
        catch (std::logic_error const& e)
        {
            LRT_(warning) << "static loading failed: " << lib.string() << ": "
                          << instance << ": " << e.what();
            return false;
        }
        catch (std::exception const& e)
        {
            LRT_(warning) << "static loading failed: " << lib.string() << ": "
                          << instance << ": " << e.what();
            return false;
        }
        return true;    // component got loaded
    }

    ///////////////////////////////////////////////////////////////////////////
    // Load all components from the ini files found in the configuration
    int runtime_support::load_components(util::section& ini,
        naming::gid_type const& prefix, naming::resolver_client& agas_client,
        hpx::program_options::options_description& options,
        std::set<std::string>& startup_handled)
    {
        // load all components as described in the configuration information
        if (!ini.has_section("hpx.components"))
        {
            LRT_(info) << "No components found/loaded, HPX will be mostly "
                          "non-functional (no section [hpx.components] found).";
            return 0;    // no components to load
        }

        // each shared library containing components may have an ini section
        //
        // # mandatory section describing the component module
        // [hpx.components.instance_name]
        //  name = ...           # the name of this component module
        //  path = ...           # the path where to find this component module
        //  enabled = false      # optional (default is assumed to be true)
        //  static = false       # optional (default is assumed to be false)
        //
        // # optional section defining additional properties for this module
        // [hpx.components.instance_name.settings]
        //  key = value
        //
        util::section* sec = ini.get_section("hpx.components");
        if (nullptr == sec)
        {
            LRT_(error) << "nullptr section found";
            return 0;    // something bad happened
        }

        util::section::section_map const& s = (*sec).get_sections();
        typedef util::section::section_map::const_iterator iterator;
        iterator end = s.end();
        for (iterator i = s.begin(); i != end; ++i)
        {
            namespace fs = filesystem;

            // the section name is the instance name of the component
            util::section const& sect = i->second;
            std::string instance(sect.get_name());
            std::string component;

            if (sect.has_entry("name"))
                component = sect.get_entry("name");
            else
                component = instance;

            bool isenabled = true;
            if (sect.has_entry("enabled"))
            {
                std::string tmp = sect.get_entry("enabled");
                hpx::string_util::to_lower(tmp);
                if (tmp == "no" || tmp == "false" || tmp == "0")
                {
                    LRT_(info) << "component factory disabled: " << instance;
                    isenabled = false;    // this component has been disabled
                }
            }

            // test whether this component section was generated
            bool isdefault = false;
            if (sect.has_entry("isdefault"))
            {
                std::string tmp = sect.get_entry("isdefault");
                hpx::string_util::to_lower(tmp);
                if (tmp == "true")
                    isdefault = true;
            }

            fs::path lib;
            try
            {
                std::string component_path;
                if (sect.has_entry("path"))
                    component_path = sect.get_entry("path");
                else
                    component_path = HPX_DEFAULT_COMPONENT_PATH;

                typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
                boost::char_separator<char> sep(HPX_INI_PATH_DELIMITER);
                tokenizer tokens(component_path, sep);
                std::error_code fsec;
                for (tokenizer::iterator it = tokens.begin();
                     it != tokens.end(); ++it)
                {
                    lib = fs::path(*it);
                    fs::path lib_path =
                        lib / std::string(HPX_MAKE_DLL_STRING(component));
                    if (fs::exists(lib_path, fsec))
                    {
                        break;
                    }
                    lib.clear();
                }

                if (sect.get_entry("static", "0") == "1")
                {
                    load_component_static(ini, instance, component, lib, prefix,
                        agas_client, isdefault, isenabled, options,
                        startup_handled);
                }
                else
                {
#if defined(HPX_HAVE_STATIC_LINKING)
                    HPX_THROW_EXCEPTION(service_unavailable,
                        "runtime_support::load_components",
                        "static linking configuration does not support dynamic "
                        "loading of component '" +
                            instance + "'");
#else
                    load_component_dynamic(ini, instance, component, lib,
                        prefix, agas_client, isdefault, isenabled, options,
                        startup_handled);
#endif
                }
            }
            catch (hpx::exception const& e)
            {
                LRT_(warning)
                    << "caught exception while loading " << instance << ", "
                    << e.get_error_code().get_message() << ": " << e.what();
                if (e.get_error_code().value() == hpx::commandline_option_error)
                {
                    std::cerr << "runtime_support::load_components: "
                              << "invalid command line option(s) to "
                              << instance << " component: " << e.what()
                              << std::endl;
                }
            }
        }    // for

        return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    bool runtime_support::load_startup_shutdown_functions_static(
        std::string const& module, error_code& ec)
    {
        try
        {
            // get the factory, may fail
            util::plugin::get_plugins_list_type f;
            if (!components::get_static_startup_shutdown(module, f))
            {
                LRT_(debug) << "static loading of startup/shutdown functions "
                               "failed: "
                            << module << ": couldn't find module in global "
                            << "static startup/shutdown functions data map";
                return false;
            }

            util::plugin::static_plugin_factory<component_startup_shutdown_base>
                pf(f);

            // create the startup_shutdown object
            std::shared_ptr<component_startup_shutdown_base> startup_shutdown(
                pf.create("startup_shutdown", ec));
            if (ec)
            {
                LRT_(debug) << "static loading of startup/shutdown functions "
                               "failed: "
                            << module << ": " << get_error_what(ec);
                return false;
            }

            startup_function_type startup;
            bool pre_startup = true;
            if (startup_shutdown->get_startup_function(startup, pre_startup))
            {
                if (!startup.empty())
                {
                    if (pre_startup)
                    {
                        pre_startup_functions_.push_back(std::move(startup));
                    }
                    else
                    {
                        startup_functions_.push_back(std::move(startup));
                    }
                }
            }

            shutdown_function_type shutdown;
            bool pre_shutdown = false;
            if (startup_shutdown->get_shutdown_function(shutdown, pre_shutdown))
            {
                if (!shutdown.empty())
                {
                    if (pre_shutdown)
                    {
                        pre_shutdown_functions_.push_back(std::move(shutdown));
                    }
                    else
                    {
                        shutdown_functions_.push_back(std::move(shutdown));
                    }
                }
            }
        }
        catch (hpx::exception const&)
        {
            throw;
        }
        catch (std::logic_error const& e)
        {
            LRT_(debug)
                << "static loading of startup/shutdown functions failed: "
                << module << ": " << e.what();
            return false;
        }
        catch (std::exception const& e)
        {
            LRT_(debug)
                << "static loading of startup/shutdown functions failed: "
                << module << ": " << e.what();
            return false;
        }
        return true;    // startup/shutdown functions got registered
    }

    ///////////////////////////////////////////////////////////////////////////
    bool runtime_support::load_commandline_options_static(
        std::string const& module,
        hpx::program_options::options_description& options, error_code& ec)
    {
        try
        {
            util::plugin::get_plugins_list_type f;
            if (!components::get_static_commandline(module, f))
            {
                LRT_(debug) << "static loading of command-line options failed: "
                            << module << ": couldn't find module in global "
                            << "static command line data map";
                return false;
            }

            // get the factory, may fail
            hpx::util::plugin::static_plugin_factory<component_commandline_base>
                pf(f);

            // create the startup_shutdown object
            std::shared_ptr<component_commandline_base> commandline_options(
                pf.create("commandline_options", ec));
            if (ec)
            {
                LRT_(debug) << "static loading of command-line options failed: "
                            << module << ": " << get_error_what(ec);
                return false;
            }

            options.add(commandline_options->add_commandline_options());
        }
        catch (hpx::exception const&)
        {
            throw;
        }
        catch (std::logic_error const& e)
        {
            LRT_(debug) << "static loading of command-line options failed: "
                        << module << ": " << e.what();
            return false;
        }
        catch (std::exception const& e)
        {
            LRT_(debug) << "static loading of command-line options failed: "
                        << module << ": " << e.what();
            return false;
        }
        return true;    // startup/shutdown functions got registered
    }

#if !defined(HPX_HAVE_STATIC_LINKING)
    bool runtime_support::load_component_dynamic(util::section& ini,
        std::string const& instance, std::string const& component,
        filesystem::path lib, naming::gid_type const& prefix,
        naming::resolver_client& agas_client, bool isdefault, bool isenabled,
        hpx::program_options::options_description& options,
        std::set<std::string>& startup_handled)
    {
        modules_map_type::iterator it =
            modules_.find(HPX_MANGLE_STRING(component));
        if (it != modules_.cend())
        {
            // use loaded module, instantiate the requested factory
            return load_component((*it).second, ini, instance, component, lib,
                prefix, agas_client, isdefault, isenabled, options,
                startup_handled);
        }

        // first, try using the path as the full path to the library
        error_code ec(lightweight);
        hpx::util::plugin::dll d(lib.string(), HPX_MANGLE_STRING(component));
        d.load_library(ec);
        if (ec)
        {
            // build path to component to load
            std::string libname(HPX_MAKE_DLL_STRING(component));
            lib /= filesystem::path(libname);
            d.load_library(ec);
            if (ec)
            {
                LRT_(warning) << "dynamic loading failed: " << lib.string()
                              << ": " << instance << ": " << get_error_what(ec);
                return false;    // next please :-P
            }
        }

        // now, instantiate the requested factory
        if (!load_component(d, ini, instance, component, lib, prefix,
                agas_client, isdefault, isenabled, options, startup_handled))
        {
            return false;    // next please :-P
        }

        modules_.insert(std::make_pair(HPX_MANGLE_STRING(component), d));
        return true;
    }

    bool runtime_support::load_startup_shutdown_functions(
        hpx::util::plugin::dll& d, error_code& ec)
    {
        try
        {
            // get the factory, may fail
            hpx::util::plugin::plugin_factory<component_startup_shutdown_base>
                pf(d, "startup_shutdown");

            // create the startup_shutdown object
            std::shared_ptr<component_startup_shutdown_base> startup_shutdown(
                pf.create("startup_shutdown", ec));
            if (ec)
            {
                LRT_(debug) << "loading of startup/shutdown functions failed: "
                            << d.get_name() << ": " << get_error_what(ec);
                return false;
            }

            startup_function_type startup;
            bool pre_startup = true;
            if (startup_shutdown->get_startup_function(startup, pre_startup))
            {
                if (pre_startup)
                    pre_startup_functions_.push_back(std::move(startup));
                else
                    startup_functions_.push_back(std::move(startup));
            }

            shutdown_function_type shutdown;
            bool pre_shutdown = false;
            if (startup_shutdown->get_shutdown_function(shutdown, pre_shutdown))
            {
                if (pre_shutdown)
                    pre_shutdown_functions_.push_back(std::move(shutdown));
                else
                    shutdown_functions_.push_back(std::move(shutdown));
            }
        }
        catch (hpx::exception const&)
        {
            throw;
        }
        catch (std::logic_error const& e)
        {
            LRT_(debug) << "loading of startup/shutdown functions failed: "
                        << d.get_name() << ": " << e.what();
            return false;
        }
        catch (std::exception const& e)
        {
            LRT_(debug) << "loading of startup/shutdown functions failed: "
                        << d.get_name() << ": " << e.what();
            return false;
        }
        return true;    // startup/shutdown functions got registered
    }

    bool runtime_support::load_commandline_options(hpx::util::plugin::dll& d,
        hpx::program_options::options_description& options, error_code& ec)
    {
        try
        {
            // get the factory, may fail
            hpx::util::plugin::plugin_factory<component_commandline_base> pf(
                d, "commandline_options");

            // create the startup_shutdown object
            std::shared_ptr<component_commandline_base> commandline_options(
                pf.create("commandline_options", ec));
            if (ec)
            {
                LRT_(debug) << "loading of command-line options failed: "
                            << d.get_name() << ": " << get_error_what(ec);
                return false;
            }

            options.add(commandline_options->add_commandline_options());
        }
        catch (hpx::exception const&)
        {
            throw;
        }
        catch (std::logic_error const& e)
        {
            LRT_(debug) << "loading of command-line options failed: "
                        << d.get_name() << ": " << e.what();
            return false;
        }
        catch (std::exception const& e)
        {
            LRT_(debug) << "loading of command-line options failed: "
                        << d.get_name() << ": " << e.what();
            return false;
        }
        return true;    // startup/shutdown functions got registered
    }

    ///////////////////////////////////////////////////////////////////////////
    bool runtime_support::load_component(hpx::util::plugin::dll& d,
        util::section& ini, std::string const& instance,
        std::string const& /* component */, filesystem::path const& lib,
        naming::gid_type const& /* prefix */,
        naming::resolver_client& /* agas_client */, bool /* isdefault */,
        bool /* isenabled */,
        hpx::program_options::options_description& options,
        std::set<std::string>& startup_handled)
    {
        try
        {
            // initialize the factory instance using the preferences from the
            // ini files
            util::section const* glob_ini = nullptr;
            if (ini.has_section("settings"))
                glob_ini = ini.get_section("settings");

            util::section const* component_ini = nullptr;
            std::string component_section("hpx.components." + instance);
            if (ini.has_section(component_section))
                component_ini = ini.get_section(component_section);

            if (nullptr == component_ini ||
                "0" == component_ini->get_entry("no_factory", "0"))
            {
                // get the factory
                hpx::util::plugin::plugin_factory<component_factory_base> pf(
                    d, "factory");

                LRT_(info) << "dynamic loading succeeded: " << lib.string()
                           << ": " << instance;
            }

            // make sure startup/shutdown registration is called once for each
            // module, same for plugins
            if (startup_handled.find(d.get_name()) == startup_handled.end())
            {
                error_code ec(lightweight);
                startup_handled.insert(d.get_name());
                load_commandline_options(d, options, ec);
                if (ec)
                    ec = error_code(lightweight);
                load_startup_shutdown_functions(d, ec);
            }
        }
        catch (hpx::exception const&)
        {
            throw;
        }
        catch (std::logic_error const& e)
        {
            LRT_(warning) << "dynamic loading failed: " << lib.string() << ": "
                          << instance << ": " << e.what();
            return false;
        }
        catch (std::exception const& e)
        {
            LRT_(warning) << "dynamic loading failed: " << lib.string() << ": "
                          << instance << ": " << e.what();
            return false;
        }
        return true;    // component got loaded
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // Load all components from the ini files found in the configuration
    bool runtime_support::load_plugins(util::section& ini,
        hpx::program_options::options_description& options,
        std::set<std::string>& startup_handled)
    {
        // load all components as described in the configuration information
        if (!ini.has_section("hpx.plugins"))
        {
            LRT_(info) << "No plugins found/loaded.";
            return true;    // no plugins to load
        }

        // each shared library containing components may have an ini section
        //
        // # mandatory section describing the component module
        // [hpx.plugins.instance_name]
        //  name = ...           # the name of this component module
        //  path = ...           # the path where to find this component module
        //  enabled = false      # optional (default is assumed to be true)
        //  static = false       # optional (default is assumed to be false)
        //
        // # optional section defining additional properties for this module
        // [hpx.plugins.instance_name.settings]
        //  key = value
        //
        util::section* sec = ini.get_section("hpx.plugins");
        if (nullptr == sec)
        {
            LRT_(error) << "nullptr section found";
            return false;    // something bad happened
        }

        util::section::section_map const& s = (*sec).get_sections();
        typedef util::section::section_map::const_iterator iterator;
        iterator end = s.end();
        for (iterator i = s.begin(); i != end; ++i)
        {
            namespace fs = filesystem;

            // the section name is the instance name of the component
            util::section const& sect = i->second;
            std::string instance(sect.get_name());
            std::string component;

            if (i->second.has_entry("name"))
                component = sect.get_entry("name");
            else
                component = instance;

            bool isenabled = true;
            if (sect.has_entry("enabled"))
            {
                std::string tmp = sect.get_entry("enabled");
                hpx::string_util::to_lower(tmp);
                if (tmp == "no" || tmp == "false" || tmp == "0")
                {
                    LRT_(info) << "plugin factory disabled: " << instance;
                    isenabled = false;    // this component has been disabled
                }
            }

            fs::path lib;
            try
            {
                std::string component_path;
                if (sect.has_entry("path"))
                    component_path = sect.get_entry("path");
                else
                    component_path = HPX_DEFAULT_COMPONENT_PATH;

                typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
                boost::char_separator<char> sep(HPX_INI_PATH_DELIMITER);
                tokenizer tokens(component_path, sep);
                std::error_code fsec;
                for (tokenizer::iterator it = tokens.begin();
                     it != tokens.end(); ++it)
                {
                    lib = fs::path(*it);
                    fs::path lib_path =
                        lib / std::string(HPX_MAKE_DLL_STRING(component));
                    if (fs::exists(lib_path, fsec))
                    {
                        break;
                    }
                    lib.clear();
                }

                if (sect.get_entry("static", "0") == "1")
                {
                    // FIXME: implement statically linked plugins
                    HPX_THROW_EXCEPTION(service_unavailable,
                        "runtime_support::load_plugins",
                        "static linking configuration does not support static "
                        "loading of plugin '" +
                            instance + "'");
                }
                else
                {
#if defined(HPX_HAVE_STATIC_LINKING)
                    HPX_THROW_EXCEPTION(service_unavailable,
                        "runtime_support::load_plugins",
                        "static linking configuration does not support dynamic "
                        "loading of plugin '" +
                            instance + "'");
#else
                    // first, try using the path as the full path to the library
                    load_plugin_dynamic(ini, instance, component, lib,
                        isenabled, options, startup_handled);
#endif
                }
            }
            catch (hpx::exception const& e)
            {
                LRT_(warning)
                    << "caught exception while loading " << instance << ", "
                    << e.get_error_code().get_message() << ": " << e.what();
                if (e.get_error_code().value() == hpx::commandline_option_error)
                {
                    std::cerr << "runtime_support::load_plugins: "
                              << "invalid command line option(s) to "
                              << instance << " component: " << e.what()
                              << std::endl;
                }
            }
        }    // for
        return true;
    }

#if !defined(HPX_HAVE_STATIC_LINKING)
    bool runtime_support::load_plugin(hpx::util::plugin::dll& d,
        util::section& ini, std::string const& instance,
        std::string const& /* plugin */, filesystem::path const& lib,
        bool isenabled, hpx::program_options::options_description& options,
        std::set<std::string>& startup_handled)
    {
        try
        {
            // initialize the factory instance using the preferences from the
            // ini files
            util::section const* glob_ini = nullptr;
            if (ini.has_section("settings"))
                glob_ini = ini.get_section("settings");

            util::section const* plugin_ini = nullptr;
            std::string plugin_section("hpx.plugins." + instance);
            if (ini.has_section(plugin_section))
                plugin_ini = ini.get_section(plugin_section);

            error_code ec(lightweight);
            if (nullptr == plugin_ini ||
                "0" == plugin_ini->get_entry("no_factory", "0"))
            {
                // get the factory
                hpx::util::plugin::plugin_factory<plugins::plugin_factory_base>
                    pf(d, "factory");

                // create the component factory object, if not disabled
                std::shared_ptr<plugins::plugin_factory_base> factory(
                    pf.create(instance, ec, glob_ini, plugin_ini, isenabled));
                if (!ec)
                {
                    // store component factory and module for later use
                    plugin_factory_type data(factory, d, isenabled);
                    std::pair<plugin_map_type::iterator, bool> p =
                        plugins_.insert(
                            plugin_map_type::value_type(instance, data));

                    if (!p.second)
                    {
                        LRT_(fatal) << "duplicate plugin type: " << instance;
                        return false;
                    }

                    LRT_(info) << "dynamic loading succeeded: " << lib.string()
                               << ": " << instance;
                }
                else
                {
                    LRT_(warning)
                        << "dynamic loading of plugin factory failed: "
                        << lib.string() << ": " << instance << ": "
                        << get_error_what(ec);
                }
            }

            // make sure startup/shutdown registration is called once for each
            // module, same for plugins
            if (startup_handled.find(d.get_name()) == startup_handled.end())
            {
                startup_handled.insert(d.get_name());
                load_commandline_options(d, options, ec);
                if (ec)
                    ec = error_code(lightweight);
                load_startup_shutdown_functions(d, ec);
            }
        }
        catch (hpx::exception const&)
        {
            throw;
        }
        catch (std::logic_error const& e)
        {
            LRT_(warning) << "dynamic loading failed: " << lib.string() << ": "
                          << instance << ": " << e.what();
            return false;
        }
        catch (std::exception const& e)
        {
            LRT_(warning) << "dynamic loading failed: " << lib.string() << ": "
                          << instance << ": " << e.what();
            return false;
        }
        return true;
    }

    bool runtime_support::load_plugin_dynamic(util::section& ini,
        std::string const& instance, std::string const& plugin,
        filesystem::path lib, bool isenabled,
        hpx::program_options::options_description& options,
        std::set<std::string>& startup_handled)
    {
        modules_map_type::iterator it =
            modules_.find(HPX_MANGLE_STRING(plugin));
        if (it != modules_.cend())
        {
            // use loaded module, instantiate the requested factory
            return load_plugin((*it).second, ini, instance, plugin, lib,
                isenabled, options, startup_handled);
        }

        // get the handle of the library
        error_code ec(lightweight);
        hpx::util::plugin::dll d(lib.string(), HPX_MANGLE_STRING(plugin));
        d.load_library(ec);
        if (ec)
        {
            // build path to component to load
            std::string libname(HPX_MAKE_DLL_STRING(plugin));
            lib /= filesystem::path(libname);
            d.load_library(ec);
            if (ec)
            {
                LRT_(warning) << "dynamic loading failed: " << lib.string()
                              << ": " << instance << ": " << get_error_what(ec);
                return false;    // next please :-P
            }
        }

        // now, instantiate the requested factory
        if (!load_plugin(d, ini, instance, plugin, lib, isenabled, options,
                startup_handled))
        {
            return false;    // next please :-P
        }

        modules_.insert(std::make_pair(HPX_MANGLE_STRING(plugin), d));
        return true;    // plugin got loaded
    }
#endif
}}}    // namespace hpx::components::server
