//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/runtime.hpp>
#include <hpx/exception.hpp>
#include <hpx/apply.hpp>
#include <hpx/config/defaults.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/filesystem_compatibility.hpp>
#include <hpx/util/unlock_guard.hpp>

#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/threads/coroutines/coroutine.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/naming/unmanaged.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/runtime/components/server/create_component.hpp>
#include <hpx/runtime/components/server/memory_block.hpp>
#include <hpx/runtime/components/server/memory.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/component_factory_base.hpp>
#include <hpx/runtime/components/base_lco_factory.hpp>
#include <hpx/runtime/components/component_registry_base.hpp>
#include <hpx/runtime/components/component_startup_shutdown_base.hpp>
#include <hpx/runtime/components/component_commandline_base.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/lcos/wait_all.hpp>

#include <hpx/lcos/broadcast.hpp>
#if defined(HPX_USE_FAST_DIJKSTRA_TERMINATION_DETECTION)
#include <hpx/lcos/reduce.hpp>
#endif
#include <hpx/lcos/local/packaged_task.hpp>

#include <hpx/util/assert.hpp>
#include <hpx/util/parse_command_line.hpp>
#include <hpx/util/command_line_handling.hpp>

#include <hpx/plugins/message_handler_factory_base.hpp>
#include <hpx/plugins/binary_filter_factory_base.hpp>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/convenience.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread/locks.hpp>
#include <boost/tokenizer.hpp>

#include <algorithm>
#include <set>
#include <sstream>

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the runtime_support actions
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::bulk_create_components_action,
    bulk_create_components_action,
    hpx::actions::bulk_create_components_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::create_memory_block_action,
    create_memory_block_action,
    hpx::actions::create_memory_block_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::load_components_action,
    load_components_action,
    hpx::actions::load_components_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::call_startup_functions_action,
    call_startup_functions_action,
    hpx::actions::call_startup_functions_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::call_shutdown_functions_action,
    call_shutdown_functions_action,
    hpx::actions::call_shutdown_functions_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::free_component_action,
    free_component_action,
    hpx::actions::free_component_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::shutdown_action,
    shutdown_action,
    hpx::actions::shutdown_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::shutdown_all_action,
    shutdown_all_action,
    hpx::actions::shutdown_all_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::terminate_action,
    terminate_action,
    hpx::actions::terminate_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::terminate_all_action,
    terminate_all_action,
    hpx::actions::terminate_all_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::get_config_action,
    get_config_action,
    hpx::actions::get_config_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::update_agas_cache_entry_action,
    update_agas_cache_entry_action,
    hpx::actions::update_agas_cache_entry_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::garbage_collect_action,
    garbage_collect_action,
    hpx::actions::garbage_collect_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::create_performance_counter_action,
    create_performance_counter_action,
    hpx::actions::create_performance_counter_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::get_instance_count_action,
    get_instance_count_action,
    hpx::actions::get_instance_count_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::remove_from_connection_cache_action,
    remove_from_connection_cache_action,
    hpx::actions::remove_from_connection_cache_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::components::server::runtime_support::dijkstra_termination_action,
    dijkstra_termination_action,
    hpx::actions::dijkstra_termination_action_id)

///////////////////////////////////////////////////////////////////////////////
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::components::server::runtime_support,
    hpx::components::component_runtime_support)

namespace hpx
{
    // helper function to stop evaluating counters during shutdown
    void stop_evaluating_counters();

    namespace parcelset
    {
        // default parcel-sent handler function
        void default_write_handler(boost::system::error_code const& ec,
            parcelset::parcel const& p);
    }
}

namespace hpx { namespace components
{
    bool initial_static_loading = true;

    ///////////////////////////////////////////////////////////////////////////
    // There is no need to protect these global from thread concurrent access
    // as they are access during early startup only.
    std::vector<static_factory_load_data_type>&
    get_static_module_data()
    {
        static std::vector<static_factory_load_data_type> global_module_init_data;
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

    bool get_static_factory(std::string const& instance,
        util::plugin::get_plugins_list_type& f)
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

    bool get_static_commandline(std::string const& instance,
        util::plugin::get_plugins_list_type& f)
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

    void init_registry_startup_shutdown(static_factory_load_data_type const& data)
    {
        if (initial_static_loading)
            get_static_startup_shutdown_data().insert(
                std::make_pair(data.name, data.get_factory));
    }

    bool get_static_startup_shutdown(std::string const& instance,
        util::plugin::get_plugins_list_type& f)
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
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    runtime_support::runtime_support(hpx::util::runtime_configuration & cfg)
      : stopped_(false), terminated_(false), dijkstra_color_(false),
        shutdown_all_invoked_(false),
        modules_(cfg.modules())
    {}

    /// \brief Action to create N new default constructed components
    std::vector<naming::gid_type> runtime_support::bulk_create_components(
        components::component_type type, std::size_t count)
    {
        // locate the factory for the requested component type
        boost::unique_lock<component_map_mutex_type> l(cm_mtx_);

        std::vector<naming::gid_type> ids;

        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end() || !(*it).second.first) {
            // we don't know anything about this component
            std::ostringstream strm;
            strm << "attempt to create component instance of invalid/unknown type: "
                 << components::get_component_type_name(type);

            l.unlock();
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::bulk_create_components",
                strm.str());
            return ids;
        }

        l.unlock();

    // create new component instance
        boost::shared_ptr<component_factory_base> factory((*it).second.first);

        ids.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
            ids.push_back(factory->create());

    // log result if requested
        LRT_(info) << "successfully created " << count << " components " //-V128
                    << " of type: "
                    << components::get_component_type_name(type);
        return ids;
    }

    ///////////////////////////////////////////////////////////////////////////
    boost::shared_ptr<util::one_size_heap_list_base>
        runtime_support::get_promise_heap(components::component_type type)
    {
        // locate the factory for the requested component type
        boost::unique_lock<component_map_mutex_type> l(cm_mtx_);

        component_map_type::iterator it = components_.find(type);
        if (it == components_.end())
        {
            // we don't know anything about this promise type yet
            boost::shared_ptr<components::base_lco_factory> factory(
                new components::base_lco_factory(type));

            component_factory_type data(factory);
            std::pair<component_map_type::iterator, bool> p =
                components_.insert(component_map_type::value_type(type, data));
            if (!p.second)
            {
                l.unlock();
                HPX_THROW_EXCEPTION(out_of_memory,
                    "runtime_support::get_promise_heap",
                    "could not create base_lco_factor for type " +
                        components::get_component_type_name(type));
            }

            it = p.first;
        }

        return boost::static_pointer_cast<components::base_lco_factory>(
            (*it).second.first)->get_heap();
    }

    ///////////////////////////////////////////////////////////////////////////
    // create a new instance of a memory block
    // FIXME: error code?
    naming::gid_type runtime_support::create_memory_block(
        std::size_t count, hpx::actions::manage_object_action_base const& act)
    {
        server::memory_block* c = server::memory_block::create(count, act);
        naming::gid_type gid = c->get_base_gid();
        if (gid) {
            LRT_(info) << "successfully created memory block of size " << count //-V128
                       << ": " << gid;
            return gid;
        }

        delete c;

        std::ostringstream strm;
        strm << "global id " << gid << " is already bound to a different "
                "component instance";
        HPX_THROW_EXCEPTION(hpx::duplicate_component_address,
            "runtime_support::create_memory_block",
            strm.str());

        return naming::invalid_gid;
    }

    ///////////////////////////////////////////////////////////////////////////
    // delete an existing instance of a component
    void runtime_support::free_component(
        agas::gva const& g, naming::gid_type const& gid, boost::uint64_t count)
    {
        // Special case: component_memory_block.
        if (g.type == components::component_memory_block) {
            for (std::size_t i = 0; i != count; ++i)
            {
                naming::gid_type target = gid + i;

                // make sure this component is located here
                if (get_locality() != g.prefix)
                {
                    // FIXME: should the component be re-bound ?
                    std::ostringstream strm;
                    strm << "global id " << target << " is not bound to any "
                            "local component instance";

                    // FIXME: If this throws then we leak the rest of count.
                    // What should we do instead?
                    HPX_THROW_EXCEPTION(hpx::unknown_component_address,
                        "runtime_support::free_component",
                        strm.str());
                    return;
                }

                // free the memory block
                components::server::memory_block::destroy(
                    reinterpret_cast<components::server::memory_block*>(
                        g.lva(target, gid)));

                LRT_(info) << "successfully destroyed memory block " << target;
            }

            return;
        }
        else if (naming::refers_to_virtual_memory(gid))
        {
            // simply delete the memory
            delete [] reinterpret_cast<boost::uint8_t*>(gid.get_lsb());
            return;
        }

        // locate the factory for the requested component type
        boost::shared_ptr<component_factory_base> factory;

        {
            boost::unique_lock<component_map_mutex_type> l(cm_mtx_);
            component_map_type::const_iterator it = components_.find(g.type);
            if (it == components_.end()) {
                // we don't know anything about this component
                std::ostringstream strm;

                naming::resolver_client& client = naming::get_agas_client();
                error_code ec(lightweight);
                strm << "attempt to destroy component "
                     << gid
                     << " of invalid/unknown type: "
                     << components::get_component_type_name(g.type)
                     << " ("
                     << client.get_component_type_name(g.type, ec)
                     << ")" << std::endl;

                strm << "list of registered components: \n";
                component_map_type::iterator end = components_.end();
                for (component_map_type::iterator cit = components_.begin();
                    cit!= end; ++cit)
                {
                    strm << "  "
                         << components::get_component_type_name((*cit).first)
                         << " ("
                         << client.get_component_type_name((*cit).first, ec)
                         << ")" << std::endl;
                }

                l.unlock();
                HPX_THROW_EXCEPTION(hpx::bad_component_type,
                    "runtime_support::free_component",
                    strm.str());
                return;
            }

            factory = (*it).second.first;
        }

        // we might end up with the same address, so cache the already deleted
        // ones here.
#if defined(HPX_DEBUG)
        std::vector<naming::address> freed_components;
        freed_components.reserve(std::size_t(count));
#endif

        for (std::size_t i = 0; i != count; ++i)
        {
            naming::gid_type target(gid + i);
            naming::address addr(g.prefix, g.type, g.lva(target, gid));

#if defined(HPX_DEBUG)
            bool found = false;
            for (naming::address const& a : freed_components)
            {
                if(a == addr)
                {
                    found = true;
                    break;
                }
            }
            HPX_ASSERT(!found);
#endif
            // FIXME: If this throws then we leak the rest of count.
            // What should we do instead?

            // destroy the component instance
            factory->destroy(target, addr);

            LRT_(info) << "successfully destroyed component " << (gid + i)
                << " of type: " << components::get_component_type_name(g.type);

#if defined(HPX_DEBUG)
            freed_components.push_back(std::move(addr));
#endif
        }
    }

    // function to be called during shutdown
    // Action: shut down this runtime system instance
    void runtime_support::shutdown(double timeout,
        naming::id_type const& respond_to)
    {
        // initiate system shutdown
        stop(timeout, respond_to, false);
    }

    // function to be called to terminate this locality immediately
    void runtime_support::terminate(naming::id_type const& respond_to)
    {
        // push pending logs
        components::cleanup_logging();

        if (respond_to) {
            // respond synchronously
            typedef lcos::base_lco_with_value<void> void_lco_type;
            typedef void_lco_type::set_event_action action_type;

            naming::address addr;
            if (agas::is_local_address_cached(respond_to, addr)) {
                // execute locally, action is executed immediately as it is
                // a direct_action
                hpx::applier::detail::apply_l<action_type>(respond_to,
                    std::move(addr));
            }
            else {
                // apply remotely, parcel is sent synchronously
                hpx::applier::detail::apply_r_sync<action_type>(std::move(addr),
                    respond_to);
            }
        }

        std::abort();
    }
}}}

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::server::runtime_support::call_shutdown_functions_action
    call_shutdown_functions_action;

HPX_REGISTER_BROADCAST_ACTION_DECLARATION(call_shutdown_functions_action,
        call_shutdown_functions_action)
HPX_REGISTER_BROADCAST_ACTION_ID(call_shutdown_functions_action,
        call_shutdown_functions_action,
        hpx::actions::broadcast_call_shutdown_functions_action_id)

#if defined(HPX_USE_FAST_DIJKSTRA_TERMINATION_DETECTION)

///////////////////////////////////////////////////////////////////////////////
typedef std::logical_or<bool> std_logical_or_type;

typedef hpx::components::server::runtime_support::dijkstra_termination_action
    dijkstra_termination_action;

HPX_REGISTER_REDUCE_ACTION_DECLARATION(dijkstra_termination_action, std_logical_or_type)
HPX_REGISTER_REDUCE_ACTION(dijkstra_termination_action, std_logical_or_type)

#endif

namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    // initiate system shutdown for all localities
    void invoke_shutdown_functions(
        std::vector<naming::id_type> const& localities, bool pre_shutdown)
    {
        call_shutdown_functions_action act;
        lcos::broadcast(act, localities, pre_shutdown).get();
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime_support::dijkstra_make_black()
    {
        // Rule 1: A machine sending a message makes itself black.
        boost::lock_guard<dijkstra_mtx_type> l(dijkstra_mtx_);
        dijkstra_color_ = true;
    }

#if defined(HPX_USE_FAST_DIJKSTRA_TERMINATION_DETECTION)
    // This new code does not work, currently, as the return actions generated
    // by the futures used by hpx::reduce make the sender black. This causes
    // an infinite loop while waiting for the Dijkstra termination detection
    // to return.

    // invoked during termination detection
    bool runtime_support::dijkstra_termination()
    {
        applier::applier& appl = hpx::applier::get_applier();
        naming::resolver_client& agas_client = appl.get_agas_client();

        agas_client.start_shutdown();

        // First wait for this locality to become passive. We do this by
        // periodically checking the number of still running threads.
        //
        // Rule 0: When active, machine nr.i + 1 keeps the token; when passive,
        // it hands over the token to machine nr.i.
        threads::threadmanager_base& tm = appl.get_thread_manager();

        while (tm.get_thread_count() > 1)
        {
            // FIXME: this sleep_for is causing very long shutdown times.
            // By commenting it, #1263 gets solved.
            //this_thread::sleep_for(boost::posix_time::millisec(100));
            this_thread::yield();
        }

        // Now this locality has become passive, thus we can send the token
        // to the next locality.
        //
        // Rule 2: When machine nr.i + 1 propagates the probe, it hands over a
        // black token to machine nr.i if it is black itself, whereas while
        // being white it leaves the color of the token unchanged.
        boost::lock_guard<dijkstra_mtx_type> l(dijkstra_mtx_);
        bool dijkstra_token = dijkstra_color_;

        // Rule 5: Upon transmission of the token to machine nr.i, machine
        // nr.i + 1 becomes white.
        dijkstra_color_ = false;

        // The reduce-function (logical_or) will make sure black will be
        // propagated.
        return dijkstra_token;
    }

    // kick off termination detection
    std::size_t runtime_support::dijkstra_termination_detection(
        std::vector<naming::id_type> const& locality_ids)
    {
        boost::uint32_t num_localities =
            static_cast<boost::uint32_t>(locality_ids.size());
        if (num_localities == 1)
            return 0;

        std::size_t count = 0;      // keep track of number of trials

        {
            // Note: we protect the entire loop here since the stopping condition
            // depends on the shared variable "dijkstra_color_"
            // Proper unlocking for possible remote actions needs to be taken care of
            typedef boost::unique_lock<dijkstra_mtx_type> dijkstra_scoped_lock;
            dijkstra_scoped_lock l(dijkstra_mtx_);
            do {
                // Rule 4: Machine nr.0 initiates a probe by making itself white
                // and sending a white token to machine nr.N - 1.
                dijkstra_color_ = false;        // start off with white

                dijkstra_termination_action act;
                bool termination_aborted = false;
                {
                    util::unlock_guard<dijkstra_scoped_lock> ul(l);
                    termination_aborted = lcos::reduce(act,
                        locality_ids, std_logical_or_type()).get()
                }

                if (termination_aborted)
                {
                    dijkstra_color_ = true;     // unsuccessful termination
                }

                // Rule 3: After the completion of an unsuccessful probe, machine
                // nr.0 initiates a next probe.

                ++count;

            } while (dijkstra_color_);
        }

        return count;
    }
#else
    void runtime_support::send_dijkstra_termination_token(
        boost::uint32_t target_locality_id,
        boost::uint32_t initiating_locality_id,
        boost::uint32_t num_localities, bool dijkstra_token)
    {
        // First wait for this locality to become passive. We do this by
        // periodically checking the number of still running threads.
        //
        // Rule 0: When active, machine nr.i + 1 keeps the token; when passive,
        // it hands over the token to machine nr.i.
        applier::applier& appl = hpx::applier::get_applier();
        threads::threadmanager_base& tm = appl.get_thread_manager();

        while (tm.get_thread_count() > 1)
        {
            // FIXME: this sleep_for is causing very long shutdown times.
            // By commenting it, #1263 gets solved.
            //this_thread::sleep_for(boost::posix_time::millisec(100));
            this_thread::yield();
        }

        // Now this locality has become passive, thus we can send the token
        // to the next locality.
        //
        // Rule 2: When machine nr.i + 1 propagates the probe, it hands over a
        // black token to machine nr.i if it is black itself, whereas while
        // being white it leaves the color of the token unchanged.
        {
            boost::lock_guard<dijkstra_mtx_type> l(dijkstra_mtx_);
            if (dijkstra_color_)
                dijkstra_token = dijkstra_color_;

            // Rule 5: Upon transmission of the token to machine nr.i, machine
            // nr.i + 1 becomes white.
            dijkstra_color_ = false;
        }

        naming::id_type id(naming::get_id_from_locality_id(target_locality_id));
        apply<dijkstra_termination_action>(id, initiating_locality_id,
            num_localities, dijkstra_token);
    }

    // invoked during termination detection
    void runtime_support::dijkstra_termination(
        boost::uint32_t initiating_locality_id, boost::uint32_t num_localities,
        bool dijkstra_token)
    {
        applier::applier& appl = hpx::applier::get_applier();
        naming::resolver_client& agas_client = appl.get_agas_client();

        agas_client.start_shutdown();

        boost::uint32_t locality_id = get_locality_id();

        if (initiating_locality_id == locality_id)
        {
            // we received the token after a full circle
            if (dijkstra_token)
            {
                boost::lock_guard<dijkstra_mtx_type> l(dijkstra_mtx_);
                dijkstra_color_ = true;     // unsuccessful termination
            }

            dijkstra_cond_.notify_one();
            return;
        }

        if (0 == locality_id)
            locality_id = num_localities;

        send_dijkstra_termination_token(locality_id - 1,
            initiating_locality_id, num_localities, dijkstra_token);
    }

    // kick off termination detection
    std::size_t runtime_support::dijkstra_termination_detection(
        std::vector<naming::id_type> const& locality_ids)
    {
        boost::uint32_t num_localities =
            static_cast<boost::uint32_t>(locality_ids.size());
        if (num_localities == 1)
            return 0;

        boost::uint32_t initiating_locality_id = get_locality_id();

        // send token to previous node
        boost::uint32_t target_id = initiating_locality_id;
        if (0 == target_id)
            target_id = static_cast<boost::uint32_t>(num_localities);

        std::size_t count = 0;      // keep track of number of trials

        {
            // Note: we protect the entire loop here since the stopping condition
            // depends on the shared variable "dijkstra_color_"
            // Proper unlocking for possible remote actions needs to be taken care of
            typedef boost::unique_lock<dijkstra_mtx_type> dijkstra_scoped_lock;
            dijkstra_scoped_lock l(dijkstra_mtx_);
            do {
                // Rule 4: Machine nr.0 initiates a probe by making itself white
                // and sending a white token to machine nr.N - 1.
                dijkstra_color_ = false;        // start off with white

                {
                    util::unlock_guard<dijkstra_scoped_lock> ul(l);
                    send_dijkstra_termination_token(target_id - 1,
                        initiating_locality_id, num_localities, false);
                }

                // wait for token to come back to us
                dijkstra_cond_.wait(l);

                // Rule 3: After the completion of an unsuccessful probe, machine
                // nr.0 initiates a next probe.

                ++count;

            } while (dijkstra_color_);
        }

        return count;
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    void runtime_support::shutdown_all(double timeout)
    {
        if (find_here() != hpx::find_root_locality())
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "runtime_support::shutdown_all",
                "shutdown_all shut be invoked on the troot locality only");
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
        naming::resolver_client& agas_client = appl.get_agas_client();

        agas_client.start_shutdown();

        stop_evaluating_counters();

        std::vector<naming::id_type> locality_ids = find_all_localities();
        std::size_t count = dijkstra_termination_detection(locality_ids);

        LRT_(info) << "runtime_support::shutdown_all: " //-V128
                      "passed first termination detection (count: "
                   << count << ").";

        // execute registered shutdown functions on all localities
        invoke_shutdown_functions(locality_ids, true);
        invoke_shutdown_functions(locality_ids, false);

        LRT_(info) << "runtime_support::shutdown_all: "
            "invoked shutdown functions";

        // Do a second round of termination detection to synchronize with all
        // work which was triggered by the invocation of the shutdown
        // functions.
        count = dijkstra_termination_detection(locality_ids);

        LRT_(info) << "runtime_support::shutdown_all: " //-V128
                      "passed second termination detection (count: "
                   << count << ").";

        // Shut down all localities except the the local one, we can't use
        // broadcast here as we have to handle the back parcel in a special
        // way.
        std::reverse(locality_ids.begin(), locality_ids.end());
        boost::uint32_t locality_id = get_locality_id();
        std::vector<lcos::future<void> > lazy_actions;

        for (naming::id_type const& id : locality_ids)
        {
            if (locality_id != naming::get_locality_id_from_id(id))
            {
                using components::stubs::runtime_support;
                lazy_actions.push_back(runtime_support::shutdown_async(id,
                    timeout));
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
        applier::applier& appl = hpx::applier::get_applier();
        appl.get_agas_client().get_localities(locality_ids);
        std::reverse(locality_ids.begin(), locality_ids.end());

        // Terminate all localities except the the local one, we can't use
        // broadcast here as we have to handle the back parcel in a special
        // way.
        {
            boost::uint32_t locality_id = get_locality_id();
            std::vector<lcos::future<void> > lazy_actions;

            for (naming::gid_type gid : locality_ids)
            {
                if (locality_id != naming::get_locality_id_from_gid(gid))
                {
                    using components::stubs::runtime_support;
                    naming::id_type id(gid, naming::id_type::unmanaged);
                    lazy_actions.push_back(runtime_support::terminate_async(id));
                }
            }

            // wait for all localities to be stopped
            wait_all(lazy_actions);
        }

        // now make sure this local locality gets terminated as well.
        terminate(naming::invalid_id);   //good night
    }

    ///////////////////////////////////////////////////////////////////////////
    // Retrieve configuration information
    util::section runtime_support::get_config()
    {
        return *(get_runtime().get_config().get_section("application"));
    }

    /// \brief Insert the given name mapping into the AGAS cache of this
    ///        locality.
    void runtime_support::update_agas_cache_entry(naming::gid_type const& gid,
        naming::address const& addr, boost::uint64_t count,
        boost::uint64_t offset)
    {
        naming::get_agas_client().update_cache_entry(gid, addr, count, offset);
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
    void runtime_support::tidy()
    {
        boost::lock_guard<component_map_mutex_type> l(cm_mtx_);

        // Only after releasing the components we are allowed to release
        // the modules. This is done in reverse order of loading.
        component_map_type::iterator end = components_.end();
        for (component_map_type::iterator it = components_.begin(); it != end; /**/)
        {
            component_map_type::iterator curr = it;
            ++it;
            if ((*curr).second.first)
            {
                // this is a workaround for sloppy memory management...
                // keep module in memory until application terminated
                if (!(*curr).second.first->may_unload())
                    (*curr).second.second.keep_alive();

                // delete factory in any case
                (*curr).second.first.reset();
            }

            // now delete the entry
            components_.erase(curr);
        }

        plugins_.clear();       // unload all plugins
        modules_.clear();       // unload all modules
    }

    ///////////////////////////////////////////////////////////////////////////
    boost::int32_t runtime_support::get_instance_count(components::component_type type)
    {
        boost::unique_lock<component_map_mutex_type> l(cm_mtx_);

        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end() || !(*it).second.first) {
            // we don't know anything about this component
            std::ostringstream strm;
            strm << "attempt to query instance count for components of "
                    "invalid/unknown type: "
                 << components::get_component_type_name(type);

            l.unlock();
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::get_instance_count",
                strm.str());
            return boost::int32_t(-1);
        }

        // ask for the factory's capabilities
        return (*it).second.first->instance_count();
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Remove the given locality from our connection cache
    void runtime_support::remove_from_connection_cache(
        naming::gid_type const& gid, parcelset::endpoints_type const& eps)
    {
        runtime* rt = get_runtime_ptr();
        if (rt == 0) return;

        // instruct our connection cache to drop all connections it is holding
        rt->get_parcel_handler().remove_from_connection_cache(gid, eps);
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime_support::run()
    {
        boost::unique_lock<mutex_type> l(mtx_);
        stopped_ = false;
        terminated_ = false;
        shutdown_all_invoked_.store(false);
    }

    void runtime_support::wait()
    {
        boost::unique_lock<mutex_type> l(mtx_);
        while (!stopped_) {
            LRT_(info) << "runtime_support: about to enter wait state";
            wait_condition_.wait(l);

            LRT_(info) << "runtime_support: exiting wait state";
        }
    }

    // let thread manager clean up HPX threads
    template <typename Lock>
    inline void
    cleanup_threads(threads::threadmanager_base& tm, Lock& l)
    {
        // re-acquire pointer to self as it might have changed
        threads::thread_self* self = threads::get_self_ptr();
        HPX_ASSERT(0 != self);    // needs to be executed by a HPX thread

        // give the scheduler some time to work on remaining tasks
        {
            util::unlock_guard<Lock> ul(l);
            self->yield(threads::pending);
        }

        // get rid of all terminated threads
        tm.cleanup_terminated(true);
    }

    void runtime_support::stop(double timeout,
        naming::id_type const& respond_to, bool remove_from_remote_caches)
    {
        boost::unique_lock<mutex_type> l(mtx_);
        if (!stopped_) {
            // push pending logs
            components::cleanup_logging();

            HPX_ASSERT(!terminated_);

            applier::applier& appl = hpx::applier::get_applier();
            threads::threadmanager_base& tm = appl.get_thread_manager();
            naming::resolver_client& agas_client = appl.get_agas_client();

            util::high_resolution_timer t;
            double start_time = t.elapsed();
            bool timed_out = false;
            error_code ec(lightweight);

            stopped_ = true;

            while (tm.get_thread_count() > 1)
            {
                // let thread-manager clean up threads
                cleanup_threads(tm, l);

                // obey timeout
                if (timeout >= 0.0 && timeout < (t.elapsed() - start_time))
                {
                    // we waited long enough
                    timed_out = true;
                    break;
                }
            }

            // If it took longer than expected, kill all suspended threads as
            // well.
            if (timed_out) {
                // now we have to wait for all threads to be aborted
                start_time = t.elapsed();
                while (tm.get_thread_count() > 1)
                {
                    // abort all suspended threads
                    tm.abort_all_suspended_threads();

                    // let thread-manager clean up threads
                    cleanup_threads(tm, l);

                    // obey timeout
                    if (timeout >= 0.0 && timeout < (t.elapsed() - start_time))
                    {
                        // we waited long enough
                        break;
                    }
                }
            }

            // Drop the locality from the partition table.
            naming::gid_type here = agas_client.get_local_locality();
            agas_client.unregister_locality(here, ec);

            // unregister fixed components
            agas_client.unbind_local(appl.get_runtime_support_raw_gid(), ec);
            agas_client.unbind_local(appl.get_memory_raw_gid(), ec);

            if (remove_from_remote_caches)
                remove_here_from_connection_cache();

            if (respond_to) {
                // respond synchronously
                typedef lcos::base_lco_with_value<void> void_lco_type;
                typedef void_lco_type::set_event_action action_type;

                naming::address addr;
                if (agas::is_local_address_cached(respond_to, addr)) {
                    // this should never happen
                    HPX_ASSERT(false);
                }
                else {
                    // apply remotely, parcel is sent synchronously
                    hpx::applier::detail::apply_r_sync<action_type>(
                        std::move(addr), respond_to);
                }
            }

            wait_condition_.notify_all();
            stop_condition_.wait(l);        // wait for termination
        }
    }

    void runtime_support::notify_waiting_main()
    {
        boost::unique_lock<mutex_type> l(mtx_);
        if (!stopped_) {
            stopped_ = true;
            wait_condition_.notify_all();
            stop_condition_.wait(l);        // wait for termination
        }
    }

    // this will be called after the thread manager has exited
    void runtime_support::stopped()
    {
        boost::lock_guard<mutex_type> l(mtx_);
        if (!terminated_) {
            terminated_ = true;
            stop_condition_.notify_all();   // finished cleanup/termination
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    inline void decode (std::string &str, char const *s, char const *r)
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
        boost::program_options::options_description options;

        // then dynamic ones
        naming::resolver_client& client = get_runtime().get_agas_client();
        int result = load_components(ini, client.get_local_locality(), client,
            options, startup_handled);

        if (!load_plugins(ini, options, startup_handled))
            result = -2;

        // do secondary command line processing, check validity of options only
        try {
            std::string unknown_cmd_line(ini.get_entry("hpx.unknown_cmd_line", ""));
            if (!unknown_cmd_line.empty()) {
                std::string runtime_mode(ini.get_entry("hpx.runtime_mode", ""));
                boost::program_options::variables_map vm;

                util::commandline_error_mode mode = util::rethrow_on_error;
                std::string allow_unknown(
                    ini.get_entry("hpx.commandline.allow_unknown", "0"));
                if (allow_unknown != "0") mode = util::allow_unregistered;

                std::vector<std::string> still_unregistered_options;
                util::parse_commandline(ini, options, unknown_cmd_line, vm,
                    std::size_t(-1), mode,
                    get_runtime_mode_from_name(runtime_mode), 0,
                    &still_unregistered_options);

                std::string still_unknown_commandline;
                for (std::string const& s: still_unregistered_options)
                    still_unknown_commandline += " " + util::detail::enquote(s);

                if (!still_unknown_commandline.empty())
                {
                    util::section* s = ini.get_section("hpx");
                    HPX_ASSERT(s != 0);
                    s->add_entry("unknown_cmd_line_option",
                        still_unknown_commandline);
                }
            }

            std::string fullhelp(ini.get_entry("hpx.cmd_line_help", ""));
            if (!fullhelp.empty()) {
                std::string help_option(
                    ini.get_entry("hpx.cmd_line_help_option", ""));
                if (0 == std::string("full").find(help_option)) {
                    std::cout << decode_string(fullhelp);
                    std::cout << options << std::endl;
                }
                else {
                    throw hpx::detail::command_line_error(
                        "unknown help option: " + help_option);
                }
                return 1;
            }

            // secondary command line handling, looking for --exit and other
            // options
            std::string cmd_line(ini.get_entry("hpx.cmd_line", ""));
            if (!cmd_line.empty()) {
                std::string runtime_mode(ini.get_entry("hpx.runtime_mode", ""));
                boost::program_options::variables_map vm;

                util::parse_commandline(ini, options, cmd_line, vm, std::size_t(-1),
                    util::allow_unregistered | util::report_missing_config_file,
                    get_runtime_mode_from_name(runtime_mode));

#if defined(HPX_HAVE_HWLOC)
                if (vm.count("hpx:print-bind")) {
                    std::size_t num_threads = boost::lexical_cast<std::size_t>(
                        ini.get_entry("hpx.os_threads", 1));
                    util::handle_print_bind(vm, num_threads);
                }
#endif
                if (vm.count("hpx:list-parcel-ports"))
                    util::handle_list_parcelports();

                if (vm.count("hpx:exit"))
                    return 1;
            }
        }
        catch (std::exception const& e) {
            std::cerr << "runtime_support::load_components: "
                      << "command line processing: " << e.what() << std::endl;
            return -1;
        }

        return result;
    }

    void runtime_support::call_startup_functions(bool pre_startup)
    {
        if (pre_startup) {
            get_runtime().set_state(state_pre_startup);
            for (util::function_nonser<void()> const& f : pre_startup_functions_)
            {
                f();
            }
        }
        else {
            get_runtime().set_state(state_startup);
            for (util::function_nonser<void()> const& f : startup_functions_)
            {
                f();
            }
        }
    }

    void runtime_support::call_shutdown_functions(bool pre_shutdown)
    {
        runtime& rt = get_runtime();
        if (pre_shutdown) {
            rt.set_state(state_pre_shutdown);
            for (util::function_nonser<void()> const& f : pre_shutdown_functions_)
            {
                try {
                    f();
                }
                catch (...) {
                    rt.report_error(boost::current_exception());
                }
            }
        }
        else {
            rt.set_state(state_shutdown);
            for (util::function_nonser<void()> const& f : shutdown_functions_)
            {
                try {
                    f();
                }
                catch (...) {
                    rt.report_error(boost::current_exception());
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    bool runtime_support::keep_factory_alive(component_type type)
    {
        boost::lock_guard<component_map_mutex_type> l(cm_mtx_);

        // Only after releasing the components we are allowed to release
        // the modules. This is done in reverse order of loading.
        component_map_type::iterator it = components_.find(type);
        if (it == components_.end() || !(*it).second.first)
            return false;

        (*it).second.second.keep_alive();
        return true;
    }

    // working around non-copy-ability of packaged_task
    struct indirect_packaged_task
    {
        typedef void write_handler_type(
            boost::system::error_code const&, parcelset::parcel const&);
        typedef lcos::local::packaged_task<write_handler_type> packaged_task_type;

        indirect_packaged_task()
          : pt(boost::make_shared<packaged_task_type>(
                &parcelset::default_write_handler))
        {}

        hpx::future<void> get_future()
        {
            return pt->get_future();
        }

        template <typename ...Ts>
        void operator()(Ts&& ... vs)
        {
            (*pt)(std::forward<Ts>(vs)...);
        }

        boost::shared_ptr<packaged_task_type> pt;
    };

    void runtime_support::remove_here_from_connection_cache()
    {
        runtime* rt = get_runtime_ptr();
        if (rt == 0)
            return;

        std::vector<naming::id_type> locality_ids = find_remote_localities();

        typedef server::runtime_support::remove_from_connection_cache_action
            action_type;

        std::vector<future<void> > callbacks;
        callbacks.reserve(locality_ids.size());

        action_type act;
        for (naming::id_type const& id : locality_ids)
        {
            indirect_packaged_task ipt;
            callbacks.push_back(ipt.get_future());
            apply_cb(act, id, std::move(ipt), hpx::get_locality(), rt->endpoints());
        }

        wait_all(callbacks);
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime_support::register_message_handler(
        char const* message_handler_type, char const* action, error_code& ec)
    {
        // locate the factory for the requested plugin type
        typedef boost::unique_lock<plugin_map_mutex_type> plugin_map_scoped_lock;
        plugin_map_scoped_lock l(p_mtx_);

        plugin_map_type::const_iterator it = plugins_.find(message_handler_type);
        if (it == plugins_.end() || !(*it).second.first) {
            if (ec.category() != hpx::get_lightweight_hpx_category())
            {
                // we don't know anything about this component
                std::ostringstream strm;
                strm << "attempt to create message handler plugin instance of "
                        "invalid/unknown type: " << message_handler_type;
                l.unlock();
                HPX_THROWS_IF(ec, hpx::bad_plugin_type,
                    "runtime_support::create_message_handler",
                    strm.str());
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
        boost::shared_ptr<plugins::message_handler_factory_base> factory(
            boost::static_pointer_cast<plugins::message_handler_factory_base>(
                (*it).second.first));

        factory->register_action(action, ec);

        if (ec)
        {
            std::ostringstream strm;
            strm << "couldn't register action '" << action
                 << "' for message handler plugin of type: "
                 << message_handler_type;
            HPX_THROWS_IF(ec, hpx::bad_plugin_type,
                "runtime_support::register_message_handler",
                strm.str());
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();

        // log result if requested
        LRT_(info) << "successfully registered message handler plugin of type: "
                    << message_handler_type;
    }

    parcelset::policies::message_handler*
    runtime_support::create_message_handler(
        char const* message_handler_type, char const* action,
        parcelset::parcelport* pp, std::size_t num_messages,
        std::size_t interval, error_code& ec)
    {
        // locate the factory for the requested plugin type
        typedef boost::unique_lock<plugin_map_mutex_type> plugin_map_scoped_lock;
        plugin_map_scoped_lock l(p_mtx_);

        plugin_map_type::const_iterator it = plugins_.find(message_handler_type);
        if (it == plugins_.end() || !(*it).second.first) {
            if (ec.category() != hpx::get_lightweight_hpx_category())
            {
                // we don't know anything about this component
                std::ostringstream strm;
                strm << "attempt to create message handler plugin instance of "
                        "invalid/unknown type: " << message_handler_type;
                l.unlock();
                HPX_THROWS_IF(ec, hpx::bad_plugin_type,
                    "runtime_support::create_message_handler",
                    strm.str());
            }
            else
            {
                // lightweight error handling
                HPX_THROWS_IF(ec, hpx::bad_plugin_type,
                    "runtime_support::create_message_handler",
                    "attempt to create message handler plugin instance of "
                    "invalid/unknown type");
            }
            return 0;
        }

        l.unlock();

        // create new component instance
        boost::shared_ptr<plugins::message_handler_factory_base> factory(
            boost::static_pointer_cast<plugins::message_handler_factory_base>(
                (*it).second.first));

        parcelset::policies::message_handler* mh = factory->create(action,
            pp, num_messages, interval);
        if (0 == mh) {
            std::ostringstream strm;
            strm << "couldn't create message handler plugin of type: "
                 << message_handler_type;
            HPX_THROWS_IF(ec, hpx::bad_plugin_type,
                "runtime_support::create_message_handler",
                strm.str());
            return 0;
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
        typedef boost::unique_lock<plugin_map_mutex_type> plugin_map_scoped_lock;
        plugin_map_scoped_lock l(p_mtx_);

        plugin_map_type::const_iterator it = plugins_.find(binary_filter_type);
        if (it == plugins_.end() || !(*it).second.first) {
            // we don't know anything about this component
            std::ostringstream strm;
            strm << "attempt to create binary filter plugin instance of "
                    "invalid/unknown type: " << binary_filter_type;
            HPX_THROWS_IF(ec, hpx::bad_plugin_type,
                "runtime_support::create_binary_filter",
                strm.str());
            return 0;
        }

        l.unlock();

        // create new component instance
        boost::shared_ptr<plugins::binary_filter_factory_base> factory(
            boost::static_pointer_cast<plugins::binary_filter_factory_base>(
                (*it).second.first));

        serialization::binary_filter* bf = factory->create(compress, next_filter);
        if (0 == bf) {
            std::ostringstream strm;
            strm << "couldn't to create binary filter plugin of type: "
                 << binary_filter_type;
            HPX_THROWS_IF(ec, hpx::bad_plugin_type,
                "runtime_support::create_binary_filter",
                strm.str());
            return 0;
        }

        if (&ec != &throws)
            ec = make_success_code();

        // log result if requested
        LRT_(info) << "successfully binary filter handler plugin of type: "
                    << binary_filter_type;
        return bf;
    }

    ///////////////////////////////////////////////////////////////////////////
    bool runtime_support::load_component_static(
        util::section& ini, std::string const& instance,
        std::string const& component, boost::filesystem::path const& lib,
        naming::gid_type const& prefix, naming::resolver_client& agas_client,
        bool isdefault, bool isenabled,
        boost::program_options::options_description& options,
        std::set<std::string>& startup_handled)
    {
        try {
            // initialize the factory instance using the preferences from the
            // ini files
            util::section const* glob_ini = NULL;
            if (ini.has_section("settings"))
                glob_ini = ini.get_section("settings");

            util::section const* component_ini = NULL;
            std::string component_section("hpx.components." + instance);
            if (ini.has_section(component_section))
                component_ini = ini.get_section(component_section);

            error_code ec(lightweight);
            if (0 == component_ini ||
                "0" == component_ini->get_entry("no_factory", "0"))
            {
                util::plugin::get_plugins_list_type f;
                if (!components::get_static_factory(instance, f)) {
                    LRT_(warning) << "static loading failed: " << lib.string()
                                    << ": " << instance << ": couldn't find "
                                    << "factory in global static factory map";
                    return false;
                }

                // get the factory
                hpx::util::plugin::static_plugin_factory<
                    component_factory_base> pf (f);

                // create the component factory object, if not disabled
                boost::shared_ptr<component_factory_base> factory (
                    pf.create(instance, ec, glob_ini, component_ini, isenabled));
                if (ec) {
                    LRT_(warning) << "static loading failed: " << lib.string()
                                    << ": " << instance << ": "
                                    << get_error_what(ec);
                    return false;
                }

                component_type t = factory->get_component_type(
                    prefix, agas_client);
                if (0 == t) {
                    LRT_(info) << "component refused to load: "  << instance;
                    return false;   // module refused to load
                }

                // store component factory and module for later use
                boost::lock_guard<component_map_mutex_type> l(cm_mtx_);

                component_factory_type data(factory, isenabled);
                std::pair<component_map_type::iterator, bool> p =
                    components_.insert(component_map_type::value_type(t, data));

                if (components::get_derived_type(t) != 0) {
                // insert three component types, the base type, the derived
                // type and the combined one.
                    if (p.second) {
                        p = components_.insert(component_map_type::value_type(
                                components::get_derived_type(t), data));
                    }
                    if (p.second) {
                        components_.insert(component_map_type::value_type(
                                components::get_base_type(t), data));
                    }
                }

                if (!p.second) {
                    LRT_(fatal) << "duplicate component id: " << instance
                        << ": " << components::get_component_type_name(t);
                    return false;   // duplicate component id?
                }

                LRT_(info) << "static loading succeeded: " << lib.string()
                            << ": " << instance << ": "
                            << components::get_component_type_name(t);
            }

            // make sure startup/shutdown registration is called once for each
            // module, same for plugins
            if (startup_handled.find(component) == startup_handled.end()) {
                startup_handled.insert(component);
                load_commandline_options_static(component, options, ec);
                if (ec) ec = error_code(lightweight);
                load_startup_shutdown_functions_static(component, ec);
            }
        }
        catch (hpx::exception const&) {
            throw;
        }
        catch (std::logic_error const& e) {
            LRT_(warning) << "static loading failed: " << lib.string()
                          << ": " << instance << ": " << e.what();
            return false;
        }
        catch (std::exception const& e) {
            LRT_(warning) << "static loading failed: " << lib.string()
                          << ": " << instance << ": " << e.what();
            return false;
        }
        return true;    // component got loaded
    }

    ///////////////////////////////////////////////////////////////////////////
    // Load all components from the ini files found in the configuration
    int runtime_support::load_components(util::section& ini,
        naming::gid_type const& prefix, naming::resolver_client& agas_client,
        boost::program_options::options_description& options,
        std::set<std::string>& startup_handled)
    {
        // load all components as described in the configuration information
        if (!ini.has_section("hpx.components")) {
            LRT_(info) << "No components found/loaded, HPX will be mostly "
                          "non-functional (no section [hpx.components] found).";
            return 0;     // no components to load
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
        if (NULL == sec)
        {
            LRT_(error) << "NULL section found";
            return 0;     // something bad happened
        }

        util::section::section_map const& s = (*sec).get_sections();
        typedef util::section::section_map::const_iterator iterator;
        iterator end = s.end();
        for (iterator i = s.begin (); i != end; ++i)
        {
            namespace fs = boost::filesystem;

            // the section name is the instance name of the component
            util::section const& sect = i->second;
            std::string instance (sect.get_name());
            std::string component;

            if (sect.has_entry("name"))
                component = sect.get_entry("name");
            else
                component = instance;

            bool isenabled = true;
            if (sect.has_entry("enabled")) {
                std::string tmp = sect.get_entry("enabled");
                boost::algorithm::to_lower (tmp);
                if (tmp == "no" || tmp == "false" || tmp == "0") {
                    LRT_(info) << "component factory disabled: " << instance;
                    isenabled = false;     // this component has been disabled
                }
            }

            // test whether this component section was generated
            bool isdefault = false;
            if (sect.has_entry("isdefault")) {
                std::string tmp = sect.get_entry("isdefault");
                boost::algorithm::to_lower (tmp);
                if (tmp == "true")
                    isdefault = true;
            }

            fs::path lib;
            try {
                std::string component_path;
                if (sect.has_entry("path"))
                    component_path = sect.get_entry("path");
                else
                    component_path = HPX_DEFAULT_COMPONENT_PATH;

                typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
                boost::char_separator<char> sep(HPX_INI_PATH_DELIMITER);
                tokenizer tokens(component_path, sep);
                boost::system::error_code fsec;
                for(tokenizer::iterator it = tokens.begin(); it != tokens.end(); ++it)
                {
                    lib = hpx::util::create_path(*it);
                    fs::path lib_path = lib / std::string(HPX_MAKE_DLL_STRING(component));
                    if(fs::exists(lib_path, fsec))
                    {
                        break;
                    }
                    lib.clear();
                }

                if (sect.get_entry("static", "0") == "1") {
                    load_component_static(ini, instance,
                        component, lib, prefix, agas_client, isdefault,
                        isenabled, options, startup_handled);
                }
                else {
#if defined(HPX_HAVE_STATIC_LINKING)
                    HPX_THROW_EXCEPTION(service_unavailable,
                        "runtime_support::load_components",
                        "static linking configuration does not support dynamic "
                        "loading of component '" + instance + "'");
#else
                    load_component_dynamic(ini, instance,
                        component, lib, prefix, agas_client, isdefault,
                        isenabled, options, startup_handled);
#endif
                }
            }
            catch (hpx::exception const& e) {
                LRT_(warning) << "caught exception while loading " << instance
                              << ", " << e.get_error_code().get_message()
                              << ": " << e.what();
                if (e.get_error_code().value() == hpx::commandline_option_error)
                {
                    std::cerr << "runtime_support::load_components: "
                              << "invalid command line option(s) to "
                              << instance << " component: " << e.what()
                              << std::endl;
                }
            }
        } // for

        return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    bool runtime_support::load_startup_shutdown_functions_static(
        std::string const& module, error_code& ec)
    {
        try {
            // get the factory, may fail
            util::plugin::get_plugins_list_type f;
            if (!components::get_static_startup_shutdown(module, f))
            {
                LRT_(debug) << "static loading of startup/shutdown functions "
                    "failed: " << module << ": couldn't find module in global "
                        << "static startup/shutdown functions data map";
                return false;
            }

            util::plugin::static_plugin_factory<
                component_startup_shutdown_base> pf (f);

            // create the startup_shutdown object
            boost::shared_ptr<component_startup_shutdown_base>
                startup_shutdown(pf.create("startup_shutdown", ec));
            if (ec) {
                LRT_(debug) << "static loading of startup/shutdown functions "
                    "failed: " << module << ": " << get_error_what(ec);
                return false;
            }

            startup_function_type startup;
            bool pre_startup = true;
            if (startup_shutdown->get_startup_function(startup, pre_startup))
            {
                if (pre_startup)
                    pre_startup_functions_.push_back(startup);
                else
                    startup_functions_.push_back(startup);
            }

            shutdown_function_type s;
            bool pre_shutdown = false;
            if (startup_shutdown->get_shutdown_function(s, pre_shutdown))
            {
                if (pre_shutdown)
                    pre_shutdown_functions_.push_back(s);
                else
                    shutdown_functions_.push_back(s);
            }
        }
        catch (hpx::exception const&) {
            throw;
        }
        catch (std::logic_error const& e) {
            LRT_(debug) << "static loading of startup/shutdown functions failed: "
                        << module << ": " << e.what();
            return false;
        }
        catch (std::exception const& e) {
            LRT_(debug) << "static loading of startup/shutdown functions failed: "
                        << module << ": " << e.what();
            return false;
        }
        return true;    // startup/shutdown functions got registered
    }

    ///////////////////////////////////////////////////////////////////////////
    bool runtime_support::load_commandline_options_static(
        std::string const& module,
        boost::program_options::options_description& options, error_code& ec)
    {
        try {
            util::plugin::get_plugins_list_type f;
            if (!components::get_static_commandline(module, f))
            {
                LRT_(debug) << "static loading of command-line options failed: "
                        << module << ": couldn't find module in global "
                        << "static command line data map";
                return false;
            }

            // get the factory, may fail
            hpx::util::plugin::static_plugin_factory<
                component_commandline_base> pf (f);

            // create the startup_shutdown object
            boost::shared_ptr<component_commandline_base>
                commandline_options(pf.create("commandline_options", ec));
            if (ec) {
                LRT_(debug) << "static loading of command-line options failed: "
                            << module << ": " << get_error_what(ec);
                return false;
            }

            options.add(commandline_options->add_commandline_options());
        }
        catch (hpx::exception const&) {
            throw;
        }
        catch (std::logic_error const& e) {
            LRT_(debug) << "static loading of command-line options failed: "
                        << module << ": " << e.what();
            return false;
        }
        catch (std::exception const& e) {
            LRT_(debug) << "static loading of command-line options failed: "
                        << module << ": " << e.what();
            return false;
        }
        return true;    // startup/shutdown functions got registered
    }

#if !defined(HPX_HAVE_STATIC_LINKING)
    bool runtime_support::load_component_dynamic(
        util::section& ini, std::string const& instance,
        std::string const& component, boost::filesystem::path lib,
        naming::gid_type const& prefix, naming::resolver_client& agas_client,
        bool isdefault, bool isenabled,
        boost::program_options::options_description& options,
        std::set<std::string>& startup_handled)
    {
        modules_map_type::iterator it = modules_.find(HPX_MANGLE_STRING(component));
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
        if (ec) {
            // build path to component to load
            std::string libname(HPX_MAKE_DLL_STRING(component));
            lib /= hpx::util::create_path(libname);
            d.load_library(ec);
            if (ec) {
                LRT_(warning) << "dynamic loading failed: " << lib.string()
                                << ": " << instance << ": " << get_error_what(ec);
                return false;   // next please :-P
            }
        }

        // now, instantiate the requested factory
        if (!load_component(d, ini, instance, component, lib, prefix,
                agas_client, isdefault, isenabled, options,
                startup_handled))
        {
            return false;   // next please :-P
        }

        modules_.insert(std::make_pair(HPX_MANGLE_STRING(component), d));
        return true;
    }

    bool runtime_support::load_startup_shutdown_functions(hpx::util::plugin::dll& d,
        error_code& ec)
    {
        try {
            // get the factory, may fail
            hpx::util::plugin::plugin_factory<component_startup_shutdown_base> pf (d,
                "startup_shutdown");

            // create the startup_shutdown object
            boost::shared_ptr<component_startup_shutdown_base>
                startup_shutdown(pf.create("startup_shutdown", ec));
            if (ec) {
                LRT_(debug) << "loading of startup/shutdown functions failed: "
                            << d.get_name() << ": " << get_error_what(ec);
                return false;
            }

            startup_function_type startup;
            bool pre_startup = true;
            if (startup_shutdown->get_startup_function(startup, pre_startup))
            {
                if (pre_startup)
                    pre_startup_functions_.push_back(startup);
                else
                    startup_functions_.push_back(startup);
            }

            shutdown_function_type s;
            bool pre_shutdown = false;
            if (startup_shutdown->get_shutdown_function(s, pre_shutdown))
            {
                if (pre_shutdown)
                    pre_shutdown_functions_.push_back(s);
                else
                    shutdown_functions_.push_back(s);
            }
        }
        catch (hpx::exception const&) {
            throw;
        }
        catch (std::logic_error const& e) {
            LRT_(debug) << "loading of startup/shutdown functions failed: "
                        << d.get_name() << ": " << e.what();
            return false;
        }
        catch (std::exception const& e) {
            LRT_(debug) << "loading of startup/shutdown functions failed: "
                        << d.get_name() << ": " << e.what();
            return false;
        }
        return true;    // startup/shutdown functions got registered
    }

    bool runtime_support::load_commandline_options(hpx::util::plugin::dll& d,
        boost::program_options::options_description& options, error_code& ec)
    {
        try {
            // get the factory, may fail
            hpx::util::plugin::plugin_factory<component_commandline_base> pf (d,
                "commandline_options");

            // create the startup_shutdown object
            boost::shared_ptr<component_commandline_base>
                commandline_options(pf.create("commandline_options", ec));
            if (ec) {
                LRT_(debug) << "loading of command-line options failed: "
                            << d.get_name() << ": " << get_error_what(ec);
                return false;
            }

            options.add(commandline_options->add_commandline_options());
        }
        catch (hpx::exception const&) {
            throw;
        }
        catch (std::logic_error const& e) {
            LRT_(debug) << "loading of command-line options failed: "
                        << d.get_name() << ": " << e.what();
            return false;
        }
        catch (std::exception const& e) {
            LRT_(debug) << "loading of command-line options failed: "
                        << d.get_name() << ": " << e.what();
            return false;
        }
        return true;    // startup/shutdown functions got registered
    }

    ///////////////////////////////////////////////////////////////////////////
    bool runtime_support::load_component(
        hpx::util::plugin::dll& d, util::section& ini,
        std::string const& instance, std::string const& component,
        boost::filesystem::path const& lib, naming::gid_type const& prefix,
        naming::resolver_client& agas_client, bool isdefault, bool isenabled,
        boost::program_options::options_description& options,
        std::set<std::string>& startup_handled)
    {
        try {
            // initialize the factory instance using the preferences from the
            // ini files
            util::section const* glob_ini = NULL;
            if (ini.has_section("settings"))
                glob_ini = ini.get_section("settings");

            util::section const* component_ini = NULL;
            std::string component_section("hpx.components." + instance);
            if (ini.has_section(component_section))
                component_ini = ini.get_section(component_section);

            error_code ec(lightweight);
            if (0 == component_ini ||
                "0" == component_ini->get_entry("no_factory", "0"))
            {
                // get the factory
                hpx::util::plugin::plugin_factory<component_factory_base> pf (d,
                    "factory");

                // create the component factory object, if not disabled
                boost::shared_ptr<component_factory_base> factory (
                    pf.create(instance, ec, glob_ini, component_ini, isenabled));
                if (ec) {
                    LRT_(warning) << "dynamic loading failed: " << lib.string()
                                  << ": " << instance << ": "
                                  << get_error_what(ec);
                    return false;
                }

                component_type t = factory->get_component_type(
                    prefix, agas_client);
                if (0 == t) {
                    LRT_(info) << "component refused to load: "  << instance;
                    return false;   // module refused to load
                }

                // store component factory and module for later use
                boost::lock_guard<component_map_mutex_type> l(cm_mtx_);

                component_factory_type data(factory, d, isenabled);
                std::pair<component_map_type::iterator, bool> p =
                    components_.insert(component_map_type::value_type(t, data));

                if (components::get_derived_type(t) != 0) {
                // insert three component types, the base type, the derived
                // type and the combined one.
                    if (p.second) {
                        p = components_.insert(component_map_type::value_type(
                                components::get_derived_type(t), data));
                    }
                    if (p.second) {
                        components_.insert(component_map_type::value_type(
                                components::get_base_type(t), data));
                    }
                }

                if (!p.second) {
                    LRT_(fatal) << "duplicate component id: " << instance
                        << ": " << components::get_component_type_name(t);
                    return false;   // duplicate component id?
                }

                LRT_(info) << "dynamic loading succeeded: " << lib.string()
                            << ": " << instance << ": "
                            << components::get_component_type_name(t);
            }

            // make sure startup/shutdown registration is called once for each
            // module, same for plugins
            if (startup_handled.find(d.get_name()) == startup_handled.end()) {
                startup_handled.insert(d.get_name());
                load_commandline_options(d, options, ec);
                if (ec) ec = error_code(lightweight);
                load_startup_shutdown_functions(d, ec);
            }
        }
        catch (hpx::exception const&) {
            throw;
        }
        catch (std::logic_error const& e) {
            LRT_(warning) << "dynamic loading failed: " << lib.string()
                          << ": " << instance << ": " << e.what();
            return false;
        }
        catch (std::exception const& e) {
            LRT_(warning) << "dynamic loading failed: " << lib.string()
                          << ": " << instance << ": " << e.what();
            return false;
        }
        return true;    // component got loaded
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // Load all components from the ini files found in the configuration
    bool runtime_support::load_plugins(util::section& ini,
        boost::program_options::options_description& options,
        std::set<std::string>& startup_handled)
    {
        // load all components as described in the configuration information
        if (!ini.has_section("hpx.plugins")) {
            LRT_(info) << "No plugins found/loaded.";
            return true;     // no plugins to load
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
        if (NULL == sec)
        {
            LRT_(error) << "NULL section found";
            return false;     // something bad happened
        }

        util::section::section_map const& s = (*sec).get_sections();
        typedef util::section::section_map::const_iterator iterator;
        iterator end = s.end();
        for (iterator i = s.begin (); i != end; ++i)
        {
            namespace fs = boost::filesystem;

            // the section name is the instance name of the component
            util::section const& sect = i->second;
            std::string instance (sect.get_name());
            std::string component;

            if (i->second.has_entry("name"))
                component = sect.get_entry("name");
            else
                component = instance;

            bool isenabled = true;
            if (sect.has_entry("enabled")) {
                std::string tmp = sect.get_entry("enabled");
                boost::algorithm::to_lower (tmp);
                if (tmp == "no" || tmp == "false" || tmp == "0") {
                    LRT_(info) << "plugin factory disabled: " << instance;
                    isenabled = false;     // this component has been disabled
                }
            }

            fs::path lib;
            try {
                std::string component_path;
                if (sect.has_entry("path"))
                    component_path = sect.get_entry("path");
                else
                    component_path = HPX_DEFAULT_COMPONENT_PATH;

                typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
                boost::char_separator<char> sep(HPX_INI_PATH_DELIMITER);
                tokenizer tokens(component_path, sep);
                boost::system::error_code fsec;
                for(tokenizer::iterator it = tokens.begin(); it != tokens.end(); ++it)
                {
                    lib = hpx::util::create_path(*it);
                    fs::path lib_path = lib / std::string(HPX_MAKE_DLL_STRING(component));
                    if(fs::exists(lib_path, fsec))
                    {
                        break;
                    }
                    lib.clear();
                }

                if (sect.get_entry("static", "0") == "1") {
                    // FIXME: implement statically linked plugins
                    HPX_THROW_EXCEPTION(service_unavailable,
                        "runtime_support::load_plugins",
                        "static linking configuration does not support static "
                        "loading of plugin '" + instance + "'");
                }
                else {
#if defined(HPX_HAVE_STATIC_LINKING)
                    HPX_THROW_EXCEPTION(service_unavailable,
                        "runtime_support::load_plugins",
                        "static linking configuration does not support dynamic "
                        "loading of plugin '" + instance + "'");
#else
                    // first, try using the path as the full path to the library
                    load_plugin_dynamic(ini, instance, component, lib,
                        isenabled, options, startup_handled);
#endif
                }
            }
            catch (hpx::exception const& e) {
                LRT_(warning) << "caught exception while loading " << instance
                              << ", " << e.get_error_code().get_message()
                              << ": " << e.what();
                if (e.get_error_code().value() == hpx::commandline_option_error)
                {
                    std::cerr << "runtime_support::load_pluginss: "
                              << "invalid command line option(s) to "
                              << instance << " component: " << e.what()
                              << std::endl;
                }
            }
        } // for
        return true;
    }

#if !defined(HPX_HAVE_STATIC_LINKING)
    bool runtime_support::load_plugin(hpx::util::plugin::dll& d,
        util::section& ini,
        std::string const& instance, std::string const& plugin,
        boost::filesystem::path const& lib, bool isenabled,
        boost::program_options::options_description& options,
        std::set<std::string>& startup_handled)
    {
        try {
            // initialize the factory instance using the preferences from the
            // ini files
            util::section const* glob_ini = NULL;
            if (ini.has_section("settings"))
                glob_ini = ini.get_section("settings");

            util::section const* plugin_ini = NULL;
            std::string plugin_section("hpx.plugins." + instance);
            if (ini.has_section(plugin_section))
                plugin_ini = ini.get_section(plugin_section);

            error_code ec(lightweight);
            if (0 == plugin_ini ||
                "0" == plugin_ini->get_entry("no_factory", "0"))
            {
                // get the factory
                hpx::util::plugin::plugin_factory<plugins::plugin_factory_base>
                    pf (d, "factory");

                // create the component factory object, if not disabled
                boost::shared_ptr<plugins::plugin_factory_base> factory (
                    pf.create(instance, ec, glob_ini, plugin_ini, isenabled));
                if (!ec)
                {
                    // store component factory and module for later use
                    plugin_factory_type data(factory, d, isenabled);
                    std::pair<plugin_map_type::iterator, bool> p =
                        plugins_.insert(plugin_map_type::value_type(instance, data));

                    if (!p.second) {
                        LRT_(fatal) << "duplicate plugin type: " << instance;
                        return false;
                    }

                    LRT_(info) << "dynamic loading succeeded: " << lib.string()
                               << ": " << instance;
                }
                else
                {
                    LRT_(warning) << "dynamic loading of plugin factory failed: "
                        << lib.string() << ": " << instance << ": "
                        << get_error_what(ec);
                }
            }

            // make sure startup/shutdown registration is called once for each
            // module, same for plugins
            if (startup_handled.find(d.get_name()) == startup_handled.end()) {
                startup_handled.insert(d.get_name());
                load_commandline_options(d, options, ec);
                if (ec) ec = error_code(lightweight);
                load_startup_shutdown_functions(d, ec);
            }
        }
        catch (hpx::exception const&) {
            throw;
        }
        catch (std::logic_error const& e) {
            LRT_(warning) << "dynamic loading failed: " << lib.string()
                          << ": " << instance << ": " << e.what();
            return false;
        }
        catch (std::exception const& e) {
            LRT_(warning) << "dynamic loading failed: " << lib.string()
                          << ": " << instance << ": " << e.what();
            return false;
        }
        return true;
    }

    bool runtime_support::load_plugin_dynamic(util::section& ini,
        std::string const& instance, std::string const& plugin,
        boost::filesystem::path lib, bool isenabled,
        boost::program_options::options_description& options,
        std::set<std::string>& startup_handled)
    {
        modules_map_type::iterator it = modules_.find(HPX_MANGLE_STRING(plugin));
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
        if (ec) {
            // build path to component to load
            std::string libname(HPX_MAKE_DLL_STRING(plugin));
            lib /= hpx::util::create_path(libname);
            d.load_library(ec);
            if (ec) {
                LRT_(warning) << "dynamic loading failed: " << lib.string()
                                << ": " << instance << ": " << get_error_what(ec);
                return false;   // next please :-P
            }
        }

        // now, instantiate the requested factory
        if (!load_plugin(d, ini, instance, plugin, lib, isenabled, options,
                startup_handled))
        {
            return false;   // next please :-P
        }

        modules_.insert(std::make_pair(HPX_MANGLE_STRING(plugin), d));
        return true;    // plugin got loaded
    }
#endif

#if defined(HPX_HAVE_SECURITY)
    components::security::capability
        runtime_support::get_factory_capabilities(components::component_type type)
    {
        components::security::capability caps;

        boost::unique_lock<component_map_mutex_type> l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            std::ostringstream strm;
            strm << "attempt to extract capabilities for component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component type not found in map)";

            l.unlock();
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::get_factory_capabilities",
                strm.str());
            return caps;
        }

        if (!(*it).second.first) {
            std::ostringstream strm;
            strm << "attempt to extract capabilities for component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";

            l.unlock();
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::get_factory_capabilities",
                strm.str());
            return caps;
        }

        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::unlock_guard<boost::unique_lock<component_map_mutex_type> > ul(l);
            caps = factory->get_required_capabilities();
        }
        return caps;
    }
#endif
}}}

