//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/filesystem_compatibility.hpp>
#include <hpx/util/unlock_lock.hpp>

#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/runtime/components/server/create_component.hpp>
#include <hpx/runtime/components/server/memory_block.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/component_factory_base.hpp>
#include <hpx/runtime/components/component_registry_base.hpp>
#include <hpx/runtime/components/component_startup_shutdown_base.hpp>
#include <hpx/runtime/components/component_commandline_base.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/lcos/future_wait.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/parse_command_line.hpp>

#include <hpx/plugins/message_handler_factory_base.hpp>
#include <hpx/plugins/binary_filter_factory_base.hpp>

#include <algorithm>
#include <set>

#include <boost/foreach.hpp>
#include <boost/assert.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/convenience.hpp>
#include <boost/algorithm/string/case_conv.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the runtime_support actions
HPX_REGISTER_ACTION(
    hpx::components::server::runtime_support::factory_properties_action,
    factory_properties_action)
HPX_REGISTER_ACTION(
    hpx::components::server::runtime_support::bulk_create_components_action,
    bulk_create_components_action)
HPX_REGISTER_ACTION(
    hpx::components::server::runtime_support::create_memory_block_action,
    create_memory_block_action)
HPX_REGISTER_ACTION(
    hpx::components::server::runtime_support::load_components_action,
    load_components_action)
HPX_REGISTER_ACTION(
    hpx::components::server::runtime_support::call_startup_functions_action,
    call_startup_functions_action)
HPX_REGISTER_ACTION(
    hpx::components::server::runtime_support::call_shutdown_functions_action,
    call_shutdown_functions_action)
HPX_REGISTER_ACTION(
    hpx::components::server::runtime_support::free_component_action,
    free_component_action)
HPX_REGISTER_ACTION(
    hpx::components::server::runtime_support::shutdown_action,
    shutdown_action)
HPX_REGISTER_ACTION(
    hpx::components::server::runtime_support::shutdown_all_action,
    shutdown_all_action)
HPX_REGISTER_ACTION(
    hpx::components::server::runtime_support::terminate_action,
    terminate_action)
HPX_REGISTER_ACTION(
    hpx::components::server::runtime_support::terminate_all_action,
    terminate_all_action)
HPX_REGISTER_ACTION(
    hpx::components::server::runtime_support::get_config_action,
    get_config_action)
HPX_REGISTER_ACTION(
    hpx::components::server::runtime_support::update_agas_cache_entry_action,
    update_agas_cache_entry_action)
HPX_REGISTER_ACTION(
    hpx::components::server::runtime_support::garbage_collect_action,
    garbage_collect_action)
HPX_REGISTER_ACTION(
    hpx::components::server::runtime_support::create_performance_counter_action,
    create_performance_counter_action)
HPX_REGISTER_ACTION(
    hpx::components::server::runtime_support::get_instance_count_action,
    get_instance_count_action)
HPX_REGISTER_ACTION(
    hpx::components::server::runtime_support::remove_from_connection_cache_action,
    remove_from_connection_cache_action)

///////////////////////////////////////////////////////////////////////////////
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::components::server::runtime_support,
    hpx::components::component_runtime_support)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    runtime_support::runtime_support()
      : stopped_(false), terminated_(false)
    {}

    ///////////////////////////////////////////////////////////////////////////
    // return, whether more than one instance of the given component can be
    // created at the same time
    int runtime_support::factory_properties(components::component_type type)
    {
        // locate the factory for the requested component type
        component_map_mutex_type::scoped_lock l(cm_mtx_);

        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end() || !(*it).second.first) {
            // we don't know anything about this component
            hpx::util::osstream strm;
            strm << "attempt to query factory properties for components "
                    "invalid/unknown type: "
                 << components::get_component_type_name(type);
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::factory_properties",
                hpx::util::osstream_get_string(strm));
            return factory_invalid;
        }

    // ask for the factory's capabilities
        return (*it).second.first->get_factory_properties();
    }

    /// \brief Action to create N new default constructed components
    std::vector<naming::gid_type> runtime_support::bulk_create_components(
        components::component_type type, std::size_t count)
    {
        // locate the factory for the requested component type
        component_map_mutex_type::scoped_lock l(cm_mtx_);

        std::vector<naming::gid_type> ids;

        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end() || !(*it).second.first) {
            // we don't know anything about this component
            hpx::util::osstream strm;
            strm << "attempt to create component instance of invalid/unknown type: "
                 << components::get_component_type_name(type);
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::bulk_create_components",
                hpx::util::osstream_get_string(strm));
            return ids;
        }

        l.unlock();

    // create new component instance
        boost::shared_ptr<component_factory_base> factory((*it).second.first);

        ids.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
            ids.push_back(factory->create());

    // log result if requested
        if (LHPX_ENABLED(info))
        {
            LRT_(info) << "successfully created " << count << " components "
                        << " of type: "
                        << components::get_component_type_name(type);
        }
        return ids;
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
            LRT_(info) << "successfully created memory block of size " << count
                       << ": " << gid;
            return gid;
        }

        delete c;

        hpx::util::osstream strm;
        strm << "global id " << gid << " is already bound to a different "
                "component instance";
        HPX_THROW_EXCEPTION(hpx::duplicate_component_address,
            "runtime_support::create_memory_block",
            hpx::util::osstream_get_string(strm));

        return naming::invalid_gid;
    }

    ///////////////////////////////////////////////////////////////////////////
    // delete an existing instance of a component
    void runtime_support::free_component(
        components::component_type type, naming::gid_type const& gid,
        naming::gid_type const& count)
    {
        // Special case: component_memory_block.
        if (type == components::component_memory_block) {
            applier::applier& appl = hpx::applier::get_applier();

            for (naming::gid_type i(0, 0); i < count; ++i)
            {
                naming::gid_type target = gid + i;

                // retrieve the local address bound to the given global id
                naming::address addr;
                if (!appl.get_agas_client().resolve(target, addr))
                {
                    hpx::util::osstream strm;
                    strm << "global id " << target << " is not bound to any "
                            "component instance";
                    // FIXME: If this throws then we leak the rest of count.
                    // What should we do instead?
                    HPX_THROW_EXCEPTION(hpx::unknown_component_address,
                        "runtime_support::free_component",
                        hpx::util::osstream_get_string(strm));
                    return;
                }

                // make sure this component is located here
                if (appl.here() != addr.locality_)
                {
                    // FIXME: should the component be re-bound ?
                    hpx::util::osstream strm;
                    strm << "global id " << target << " is not bound to any "
                            "local component instance";
                    // FIXME: If this throws then we leak the rest of count.
                    // What should we do instead?
                    HPX_THROW_EXCEPTION(hpx::unknown_component_address,
                        "runtime_support::free_component",
                        hpx::util::osstream_get_string(strm));
                    return;
                }

                // free the memory block
                components::server::memory_block::destroy(
                    reinterpret_cast<components::server::memory_block*>(addr.address_));

                LRT_(info) << "successfully destroyed memory block " << target;
            }

            return;
        }

        // locate the factory for the requested component type
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            // we don't know anything about this component
            hpx::util::osstream strm;

            error_code ec(lightweight);
            strm << "attempt to destroy component " << gid
                 << " of invalid/unknown type: "
                 << components::get_component_type_name(type) << " ("
                 << naming::get_agas_client().get_component_type_name(type, ec)
                 << ")" << std::endl;

            strm << "list of registered components: \n";
            component_map_type::iterator end = components_.end();
            for (component_map_type::iterator cit = components_.begin(); cit!= end; ++cit)
            {
                strm << "  " << components::get_component_type_name((*cit).first)
                     << " (" << naming::get_agas_client().get_component_type_name((*cit).first, ec)
                      << ")" << std::endl;
            }

            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::free_component",
                hpx::util::osstream_get_string(strm));
            return;
        }


        for (naming::gid_type i(0, 0); i < count; ++i)
        {
            naming::gid_type target = gid + i;

            // FIXME: If this throws then we leak the rest of count.
            // What should we do instead?
            // destroy the component instance
            (*it).second.first->destroy(target);

            LRT_(info) << "successfully destroyed component " << target
                << " of type: " << components::get_component_type_name(type);
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
            if (agas::is_local_address(respond_to, addr)) {
                // execute locally, action is executed immediately as it is
                // a direct_action
                hpx::applier::detail::apply_l<action_type>(addr);
            }
            else {
                // apply remotely, parcel is sent synchronously
                hpx::applier::detail::apply_r_sync<action_type>(addr,
                    respond_to);
            }
        }

        std::abort();
    }

    ///////////////////////////////////////////////////////////////////////////
    // initiate system shutdown for all localities
    void invoke_shutdown_functions(
        std::vector<naming::gid_type> const& prefixes, bool pre_shutdown)
    {
        std::vector<lcos::future<void> > lazy_actions;
        BOOST_FOREACH(naming::gid_type const& gid, prefixes)
        {
            using components::stubs::runtime_support;
            naming::id_type id(gid, naming::id_type::unmanaged);
            lazy_actions.push_back(
                runtime_support::call_shutdown_functions_async(id, pre_shutdown));
        }

        // wait for all localities to finish executing their registered
        // shutdown functions
        lcos::wait(lazy_actions);
    }

    void runtime_support::shutdown_all(double timeout)
    {
        std::vector<naming::gid_type> locality_ids;
        applier::applier& appl = hpx::applier::get_applier();
        appl.get_agas_client().get_localities(locality_ids);
        std::reverse(locality_ids.begin(), locality_ids.end());

        // execute registered shutdown functions on all localities
        invoke_shutdown_functions(locality_ids, true);
        invoke_shutdown_functions(locality_ids, false);

        // shut down all localities except the the local one
        {
            boost::uint32_t locality_id = get_locality_id();
            std::vector<lcos::future<void> > lazy_actions;

            BOOST_FOREACH(naming::gid_type gid, locality_ids)
            {
                if (locality_id != naming::get_locality_id_from_gid(gid))
                {
                    using components::stubs::runtime_support;
                    naming::id_type id(gid, naming::id_type::unmanaged);
                    lazy_actions.push_back(runtime_support::shutdown_async(id, timeout));
                }
            }

            // wait for all localities to be stopped
            lcos::wait(lazy_actions);
        }

        // now make sure this local locality gets shut down as well.
        stop(timeout, naming::invalid_id, false);    // no need to respond
    }

    ///////////////////////////////////////////////////////////////////////////
    // initiate system shutdown for all localities
    void runtime_support::terminate_all()
    {
        std::vector<naming::gid_type> locality_ids;
        applier::applier& appl = hpx::applier::get_applier();
        appl.get_agas_client().get_localities(locality_ids);
        std::reverse(locality_ids.begin(), locality_ids.end());

        // terminate all localities except the the local one
        {
            boost::uint32_t locality_id = get_locality_id();
            std::vector<lcos::future<void> > lazy_actions;

            BOOST_FOREACH(naming::gid_type gid, locality_ids)
            {
                if (locality_id != naming::get_locality_id_from_gid(gid))
                {
                    using components::stubs::runtime_support;
                    naming::id_type id(gid, naming::id_type::unmanaged);
                    lazy_actions.push_back(runtime_support::terminate_async(id));
                }
            }

            // wait for all localities to be stopped
            lcos::wait(lazy_actions);
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
        component_map_mutex_type::scoped_lock l(cm_mtx_);

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
    }

    ///////////////////////////////////////////////////////////////////////////
    long runtime_support::get_instance_count(components::component_type type)
    {
        component_map_mutex_type::scoped_lock l(cm_mtx_);

        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end() || !(*it).second.first) {
            // we don't know anything about this component
            hpx::util::osstream strm;
            strm << "attempt to query instance count for components of "
                    "invalid/unknown type: "
                 << components::get_component_type_name(type);
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::factory_properties",
                hpx::util::osstream_get_string(strm));
            return factory_invalid;
        }

        // ask for the factory's capabilities
        return (*it).second.first->instance_count();
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Remove the given locality from our connection cache
    void runtime_support::remove_from_connection_cache(naming::locality const& l)
    {
        runtime* rt = get_runtime_ptr();
        if (rt == 0) return;

        // instruct our connection cache to drop all connections it is holding
        rt->get_parcel_handler().remove_from_connection_cache(l);
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime_support::run()
    {
        mutex_type::scoped_lock l(mtx_);
        stopped_ = false;
        terminated_ = false;
    }

    void runtime_support::wait()
    {
        mutex_type::scoped_lock l(mtx_);
        if (!stopped_) {
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
        BOOST_ASSERT(0 != self);    // needs to be executed by a PX thread

        // give the scheduler some time to work on remaining tasks
        {
            util::unlock_the_lock<Lock> ul(l);
            self->yield(threads::pending);
        }

        // get rid of all terminated threads
        tm.cleanup_terminated(true);
    }

    void runtime_support::stop(double timeout,
        naming::id_type const& respond_to, bool remove_from_remote_caches)
    {
        mutex_type::scoped_lock l(mtx_);
        if (!stopped_) {
            // push pending logs
            components::cleanup_logging();

            BOOST_ASSERT(!terminated_);

            stopped_ = true;

            applier::applier& appl = hpx::applier::get_applier();
            threads::threadmanager_base& tm = appl.get_thread_manager();

            util::high_resolution_timer t;
            double start_time = t.elapsed();
            bool timed_out = false;

            while (tm.get_thread_count() > 1) {
                // let thread-manager clean up threads
                cleanup_threads(tm, l);

                // obey timeout
                if ((std::abs(timeout - 1.) < 1e-16)  && timeout < (t.elapsed() - start_time)) {
                    // we waited long enough
                    timed_out = true;
                    break;
                }
            }

            // If it took longer than expected, kill all suspended threads as
            // well.
            if (timed_out) {
                // now we have to wait for all threads to be aborted
                while (tm.get_thread_count() > 1) {
                    // abort all suspended threads
                    tm.abort_all_suspended_threads();

                    // let thread-manager clean up threads
                    cleanup_threads(tm, l);
                }
            }

            //remove all entries for this locality from AGAS
            naming::resolver_client& agas_client =
                get_runtime().get_agas_client();

            error_code ec(lightweight);

            // Drop the locality from the partition table.
            agas_client.unregister_locality(appl.here(), ec);

            // unregister fixed components
            agas_client.unbind(appl.get_runtime_support_raw_gid(), ec);
            agas_client.unbind(appl.get_memory_raw_gid(), ec);

            if (remove_from_remote_caches)
                remove_here_from_connection_cache();

            if (respond_to) {
                // respond synchronously
                typedef lcos::base_lco_with_value<void> void_lco_type;
                typedef void_lco_type::set_event_action action_type;

                naming::address addr;
                if (agas::is_local_address(respond_to, addr)) {
                    // execute locally, action is executed immediately as it is
                    // a direct_action
                    hpx::applier::detail::apply_l<action_type>(addr);
                }
                else {
                    // apply remotely, parcel is sent synchronously
                    hpx::applier::detail::apply_r_sync<action_type>(addr,
                        respond_to);
                }
            }

            wait_condition_.notify_all();
            stop_condition_.wait(l);        // wait for termination
        }
    }

    // this will be called after the thread manager has exited
    void runtime_support::stopped()
    {
        mutex_type::scoped_lock l(mtx_);
        if (!terminated_) {
            stop_condition_.notify_all();   // finished cleanup/termination
            terminated_ = true;
        }
    }

    bool runtime_support::load_components()
    {
        // load components now that AGAS is up
        util::runtime_configuration& ini = get_runtime().get_config();
        ini.load_components();

        naming::resolver_client& client = get_runtime().get_agas_client();
        bool result = load_components(ini, client.get_local_locality(), client);

        return load_plugins(ini) && result;
    }

    void runtime_support::call_startup_functions(bool pre_startup)
    {
        if (pre_startup) {
            BOOST_FOREACH(HPX_STD_FUNCTION<void()> const& f, pre_startup_functions_)
            {
                f();
            }
        }
        else {
            BOOST_FOREACH(HPX_STD_FUNCTION<void()> const& f, startup_functions_)
            {
                f();
            }
        }
    }

    void runtime_support::call_shutdown_functions(bool pre_shutdown)
    {
        if (pre_shutdown) {
            BOOST_FOREACH(HPX_STD_FUNCTION<void()> const& f, pre_shutdown_functions_)
            {
                f();
            }
        }
        else {
            BOOST_FOREACH(HPX_STD_FUNCTION<void()> const& f, shutdown_functions_)
            {
                f();
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    bool runtime_support::keep_factory_alive(component_type type)
    {
        component_map_mutex_type::scoped_lock l(cm_mtx_);

        // Only after releasing the components we are allowed to release
        // the modules. This is done in reverse order of loading.
        component_map_type::iterator it = components_.find(type);
        if (it == components_.end() || !(*it).second.first)
            return false;

        (*it).second.second.keep_alive();
        return true;
    }

    void runtime_support::remove_here_from_connection_cache()
    {
        runtime* rt = get_runtime_ptr();
        if (rt == 0)
            return;

        std::vector<naming::id_type> locality_ids = find_remote_localities();

        typedef server::runtime_support::remove_from_connection_cache_action action_type;
        action_type act;
        BOOST_FOREACH(naming::id_type const& id, locality_ids)
        {
            apply(act, id, rt->here());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    parcelset::policies::message_handler*
    runtime_support::create_message_handler(
        char const* message_handler_type, char const* action,
        parcelset::parcelport* pp, std::size_t num_messages,
        std::size_t interval, error_code& ec)
    {
        // locate the factory for the requested plugin type
        plugin_map_mutex_type::scoped_lock l(p_mtx_);

        plugin_map_type::const_iterator it = plugins_.find(message_handler_type);
        if (it == plugins_.end() || !(*it).second.first) {
            // we don't know anything about this component
            hpx::util::osstream strm;
            strm << "attempt to create message handler plugin instance of "
                    "invalid/unknown type: " << message_handler_type;
            HPX_THROWS_IF(ec, hpx::bad_plugin_type,
                "runtime_support::create_message_handler",
                hpx::util::osstream_get_string(strm));
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
            hpx::util::osstream strm;
            strm << "couldn't to create message handler plugin of type: " 
                 << message_handler_type;
            HPX_THROWS_IF(ec, hpx::bad_plugin_type,
                "runtime_support::create_message_handler",
                hpx::util::osstream_get_string(strm));
            return 0;
        }

        if (&ec != &throws)
            ec = make_success_code();

        // log result if requested
        if (LHPX_ENABLED(info))
        {
            LRT_(info) << "successfully created message handler plugin of type: "
                       << message_handler_type;
        }
        return mh;
    }

    util::binary_filter* runtime_support::create_binary_filter(
        char const* binary_filter_type, bool compress, error_code& ec)
    {
        // locate the factory for the requested plugin type
        plugin_map_mutex_type::scoped_lock l(p_mtx_);

        plugin_map_type::const_iterator it = plugins_.find(binary_filter_type);
        if (it == plugins_.end() || !(*it).second.first) {
            // we don't know anything about this component
            hpx::util::osstream strm;
            strm << "attempt to create binary filter plugin instance of "
                    "invalid/unknown type: " << binary_filter_type;
            HPX_THROWS_IF(ec, hpx::bad_plugin_type,
                "runtime_support::create_binary_filter",
                hpx::util::osstream_get_string(strm));
            return 0;
        }

        l.unlock();

        // create new component instance
        boost::shared_ptr<plugins::binary_filter_factory_base> factory(
            boost::static_pointer_cast<plugins::binary_filter_factory_base>(
                (*it).second.first));

        util::binary_filter* bf = factory->create(compress);
        if (0 == bf) {
            hpx::util::osstream strm;
            strm << "couldn't to create binary filter plugin of type: " 
                 << binary_filter_type;
            HPX_THROWS_IF(ec, hpx::bad_plugin_type,
                "runtime_support::create_binary_filter",
                hpx::util::osstream_get_string(strm));
            return 0;
        }

        if (&ec != &throws)
            ec = make_success_code();

        // log result if requested
        if (LHPX_ENABLED(info))
        {
            LRT_(info) << "successfully binary filter handler plugin of type: "
                       << binary_filter_type;
        }
        return bf;
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

    ///////////////////////////////////////////////////////////////////////////
    // Load all components from the ini files found in the configuration
    bool runtime_support::load_components(util::section& ini,
        naming::gid_type const& prefix, naming::resolver_client& agas_client)
    {
        // load all components as described in the configuration information
        if (!ini.has_section("hpx.components")) {
            LRT_(info) << "No components found/loaded, HPX will be mostly "
                          "non-functional (no section [hpx.components] found).";
            return true;     // no components to load
        }

        // each shared library containing components may have an ini section
        //
        // # mandatory section describing the component module
        // [hpx.components.instance_name]
        //  name = ...           # the name of this component module
        //  path = ...           # the path where to find this component module
        //  enabled = false      # optional (default is assumed to be true)
        //
        // # optional section defining additional properties for this module
        // [hpx.components.instance_name.settings]
        //  key = value
        //
        util::section* sec = ini.get_section("hpx.components");
        if (NULL == sec)
        {
            LRT_(error) << "NULL section found";
            return false;     // something bad happened
        }

        // make sure every component module gets asked for startup/shutdown
        // functions only once
        std::set<std::string> startup_handled;

        // collect additional command-line options
        boost::program_options::options_description options;

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
                if (sect.has_entry("path"))
                    lib = hpx::util::create_path(sect.get_entry("path"));
                else
                    lib = hpx::util::create_path(HPX_DEFAULT_COMPONENT_PATH);

                // first, try using the path as the full path to the library
                if (!load_component(ini, instance, component, lib, prefix,
                        agas_client, isdefault, isenabled, options,
                        startup_handled))
                {
                    // build path to component to load
                    std::string libname(HPX_MAKE_DLL_STRING(component));
                    lib /= hpx::util::create_path(libname);
                    if (!load_component(ini, instance, component, lib, prefix,
                            agas_client, isdefault, isenabled, options,
                            startup_handled))
                    {
                        continue;   // next please :-P
                    }
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
                              << instance << " component: " << e.what() << std::endl;
                }
            }
        } // for

        // do secondary command line processing, check validity of options only
        try {
            std::string unknown_cmd_line(ini.get_entry("hpx.unknown_cmd_line", ""));
            if (!unknown_cmd_line.empty()) {
                std::string runtime_mode(ini.get_entry("hpx.runtime_mode", ""));
                boost::program_options::variables_map vm;

                util::parse_commandline(ini, options, unknown_cmd_line, vm,
                    std::size_t(-1), util::rethrow_on_error,
                    get_runtime_mode_from_name(runtime_mode));
            }

            std::string fullhelp(ini.get_entry("hpx.cmd_line_help", ""));
            if (!fullhelp.empty()) {
                std::string help_option(ini.get_entry("hpx.cmd_line_help_option", ""));
                if (0 == std::string("full").find(help_option)) {
                    std::cout << decode_string(fullhelp);
                    std::cout << options << std::endl;
                }
                else {
                    throw std::logic_error("unknown help option: " + help_option);
                }
                return false;
            }

            // secondary command line handling, looking for --exit option
            std::string cmd_line(ini.get_entry("hpx.cmd_line", ""));
            if (!cmd_line.empty()) {
                std::string runtime_mode(ini.get_entry("hpx.runtime_mode", ""));
                boost::program_options::variables_map vm;

                util::parse_commandline(ini, options, cmd_line, vm, std::size_t(-1),
                    util::allow_unregistered, get_runtime_mode_from_name(runtime_mode));

                if (vm.count("hpx:exit"))
                    return false;
            }
        }
        catch (std::exception const& e) {
            std::cerr << "runtime_support::load_components: "
                      << "command line processing: " << e.what() << std::endl;
            return false;
        }
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    bool runtime_support::load_startup_shutdown_functions(hpx::util::plugin::dll& d)
    {
        try {
            // get the factory, may fail
            hpx::util::plugin::plugin_factory<component_startup_shutdown_base> pf (d,
                "startup_shutdown");

            // create the startup_shutdown object
            boost::shared_ptr<component_startup_shutdown_base>
                startup_shutdown(pf.create("startup_shutdown"));

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

    ///////////////////////////////////////////////////////////////////////////
    bool runtime_support::load_commandline_options(hpx::util::plugin::dll& d,
        boost::program_options::options_description& options)
    {
        try {
            // get the factory, may fail
            hpx::util::plugin::plugin_factory<component_commandline_base> pf (d,
                "commandline_options");

            // create the startup_shutdown object
            boost::shared_ptr<component_commandline_base>
                commandline_options(pf.create("commandline_options"));

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
    bool runtime_support::load_component(util::section& ini,
        std::string const& instance, std::string const& component,
        boost::filesystem::path lib, naming::gid_type const& prefix,
        naming::resolver_client& agas_client, bool isdefault, bool isenabled,
        boost::program_options::options_description& options,
        std::set<std::string>& startup_handled)
    {
        namespace fs = boost::filesystem;
        if (fs::extension(lib) != HPX_SHARED_LIB_EXTENSION)
        {
            //LRT_(info) << lib.string() << " is not a shared object: " << instance;
            return false;
        }

        try {
            // get the handle of the library
            hpx::util::plugin::dll d(lib.string(), HPX_MANGLE_STRING(component));

            // initialize the factory instance using the preferences from the
            // ini files
            util::section const* glob_ini = NULL;
            if (ini.has_section("settings"))
                glob_ini = ini.get_section("settings");

            util::section const* component_ini = NULL;
            std::string component_section("hpx.components." + instance);
            if (ini.has_section(component_section))
                component_ini = ini.get_section(component_section);

            if (0 == component_ini || "0" == component_ini->get_entry("no_factory", "0"))
            {
                // get the factory
                hpx::util::plugin::plugin_factory<component_factory_base> pf (d,
                    "factory");

                // create the component factory object, if not disabled
                boost::shared_ptr<component_factory_base> factory (
                    pf.create(instance, glob_ini, component_ini, isenabled));

                component_type t = factory->get_component_type(
                    prefix, agas_client);
                if (0 == t) {
                    LRT_(info) << "component refused to load: "  << instance;
                    return false;   // module refused to load
                }

                // store component factory and module for later use
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
                load_commandline_options(d, options);
                load_startup_shutdown_functions(d);
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

    ///////////////////////////////////////////////////////////////////////////
    // Load all components from the ini files found in the configuration
    bool runtime_support::load_plugins(util::section& ini)
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
                if (sect.has_entry("path"))
                    lib = hpx::util::create_path(sect.get_entry("path"));
                else
                    lib = hpx::util::create_path(HPX_DEFAULT_COMPONENT_PATH);

                // first, try using the path as the full path to the library
                if (!load_plugin(ini, instance, component, lib, isenabled))
                {
                    // build path to component to load
                    std::string libname(HPX_MAKE_DLL_STRING(component));
                    lib /= hpx::util::create_path(libname);
                    if (!load_plugin(ini, instance, component, lib, isenabled))
                    {
                        continue;   // next please :-P
                    }
                }
            }
            catch (hpx::exception const& e) {
                LRT_(warning) << "caught exception while loading " << instance
                              << ", " << e.get_error_code().get_message()
                              << ": " << e.what();
            }
        } // for
        return true;
    }

    bool runtime_support::load_plugin(util::section& ini,
        std::string const& instance, std::string const& plugin,
        boost::filesystem::path lib, bool isenabled)
    {
        namespace fs = boost::filesystem;
        if (fs::extension(lib) != HPX_SHARED_LIB_EXTENSION)
            return false;

        try {
            // get the handle of the library
            hpx::util::plugin::dll d(lib.string(), HPX_MANGLE_STRING(plugin));

            // initialize the factory instance using the preferences from the
            // ini files
            util::section const* glob_ini = NULL;
            if (ini.has_section("settings"))
                glob_ini = ini.get_section("settings");

            util::section const* plugin_ini = NULL;
            std::string plugin_section("hpx.plugins." + instance);
            if (ini.has_section(plugin_section))
                plugin_ini = ini.get_section(plugin_section);

            if (0 != plugin_ini && "0" != plugin_ini->get_entry("no_factory", "0"))
                return false;

            // get the factory
            hpx::util::plugin::plugin_factory<plugins::plugin_factory_base>
                pf (d, "factory");

            // create the component factory object, if not disabled
            boost::shared_ptr<plugins::plugin_factory_base> factory (
                pf.create(instance, glob_ini, plugin_ini, isenabled));

            // store component factory and module for later use
            plugin_factory_type data(factory, d, isenabled);
            std::pair<plugin_map_type::iterator, bool> p =
                plugins_.insert(plugin_map_type::value_type(instance, data));

            if (!p.second) {
                LRT_(fatal) << "duplicate plugin type: " << instance;
                return false;   // duplicate component id?
            }

            LRT_(info) << "dynamic loading succeeded: " << lib.string()
                        << ": " << instance;
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
}}}

