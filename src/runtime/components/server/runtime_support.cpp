//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/util.hpp>
#include <hpx/util/logging.hpp>

#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/runtime/components/server/manage_component.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/assert.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/convenience.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the runtime_support actions
HPX_REGISTER_ACTION(hpx::components::server::runtime_support::factory_properties_action);
HPX_REGISTER_ACTION(hpx::components::server::runtime_support::create_component_action);
HPX_REGISTER_ACTION(hpx::components::server::runtime_support::free_component_action);
HPX_REGISTER_ACTION(hpx::components::server::runtime_support::shutdown_action);
HPX_REGISTER_ACTION(hpx::components::server::runtime_support::shutdown_all_action);

///////////////////////////////////////////////////////////////////////////////
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::components::server::runtime_support);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    // return, whether more than one instance of the given component can be 
    // created at the same time
    threads::thread_state runtime_support::factory_properties(
        int* factoryprops, components::component_type type)
    {
    // locate the factory for the requested component type
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            // we don't know anything about this component
            HPX_OSSTREAM strm;
            strm << "attempt to query factory properties for components "
                    "invalid/unknown type: "
                 << components::get_component_type_name(type);
            HPX_THROW_EXCEPTION(hpx::bad_component_type, 
                HPX_OSSTREAM_GETSTRING(strm));
            return threads::terminated;
        }

    // ask for the factory's capabilities
        *factoryprops = (*it).second->get_factory_properties();
        return threads::terminated;
    }

    // create a new instance of a component
    threads::thread_state runtime_support::create_component(
        naming::id_type* gid, components::component_type type, std::size_t count)
    {
    // locate the factory for the requested component type
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            // we don't know anything about this component
            HPX_OSSTREAM strm;
            strm << "attempt to create component instance of invalid/unknown type: "
                 << components::get_component_type_name(type);
            HPX_THROW_EXCEPTION(hpx::bad_component_type, 
                HPX_OSSTREAM_GETSTRING(strm));
            return threads::terminated;
        }

    // create new component instance
        naming::id_type id = (*it).second->create(count);

    // set result if requested
        if (0 != gid)
            *gid = id;

        if (LHPX_ENABLED(info)) {
            if ((*it).second->get_factory_properties() & factory_instance_count_is_size) 
            {
                LRT_(info) << "successfully created component " << *gid 
                           << " of type: " 
                           << components::get_component_type_name(type) 
                           << " (size: " << count << ")";
            }
            else {
                LRT_(info) << "successfully created " << count 
                           << " component(s) " << *gid << " of type: " 
                           << components::get_component_type_name(type);
            }
        }
        return threads::terminated;
    }

    // delete an existing instance of a component
    threads::thread_state runtime_support::free_component(
        components::component_type type, naming::id_type const& gid)
    {
    // locate the factory for the requested component type
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            // we don't know anything about this component
            HPX_OSSTREAM strm;
            strm << "attempt to destroy component " << gid 
                 << " of invalid/unknown type: " 
                 << components::get_component_type_name(type);
            HPX_THROW_EXCEPTION(hpx::bad_component_type, 
                HPX_OSSTREAM_GETSTRING(strm));
            return threads::terminated;
        }

    // destroy the component instance
        (*it).second->destroy(gid);

        LRT_(info) << "successfully destroyed component " << gid 
            << " of type: " << components::get_component_type_name(type);
        return threads::terminated;
    }

    // Action: shut down this runtime system instance
    threads::thread_state runtime_support::shutdown()
    {
        // initiate system shutdown
        stop();
        return threads::terminated;
    }

    // initiate system shutdown for all localities
    threads::thread_state runtime_support::shutdown_all()
    {
        std::vector<naming::id_type> prefixes;
        applier::applier& appl = hpx::applier::get_applier();
        appl.get_agas_client().get_prefixes(prefixes);

        // shut down all localities except the the local one
        std::vector<naming::id_type>::iterator end = prefixes.end();
        for (std::vector<naming::id_type>::iterator it = prefixes.begin(); 
             it != end; ++it)
        {
            if (naming::get_prefix_from_id(appl.get_prefix()) !=
                naming::get_prefix_from_id(*it))
            {
                components::stubs::runtime_support::shutdown(*it);
            }
        }

        // now make sure the local locality gets shut down as well.
        stop();
        return threads::terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime_support::wait()
    {
        mutex_type::scoped_lock l(mtx_);
        if (!stopped_) {
            LRT_(info) << "runtime_support: about to enter wait state";
            condition_.wait(l);
            LRT_(info) << "runtime_support: exiting wait state";
        }
    }

    void runtime_support::stop()
    {
        mutex_type::scoped_lock l(mtx_);
        if (!stopped_) {
            condition_.notify_all();
            stopped_ = true;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Load all components from the ini files found in the configuration
    void runtime_support::load_components(util::section& ini, 
        naming::id_type const& prefix, naming::resolver_client& agas_client)
    {
        // load all components as described in the configuration information
        if (!ini.has_section("hpx.components")) {
            LRT_(info) << "No components found/loaded, HPX will be mostly "
                          "non-functional (no section [hpx.components] found).";
            return;     // no components to load
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
            return;     // something bad happened

        util::section::section_map const& s = (*sec).get_sections();

        typedef util::section::section_map::const_iterator iterator;
        iterator end = s.end();
        for (iterator i = s.begin (); i != end; ++i) 
        {
            namespace fs = boost::filesystem;

            // the section name is the instance name of the component
            std::string instance (i->second.get_name());
            std::string component;

            if (i->second.has_entry("name"))
                component = HPX_MANGLE_COMPONENT_NAME_STR(i->second.get_entry("name"));
            else
                component = HPX_MANGLE_COMPONENT_NAME_STR(instance);

            if (i->second.has_entry("enabled")) {
                std::string tmp = i->second.get_entry("enabled");
                boost::to_lower (tmp);
                if (tmp == "no" || tmp == "false" || tmp == "0") {
                    LRT_(info) << "dynamic loading disabled: " << instance;
                    continue;     // this component has been disabled
                }
            }

            // test whether this component section was generated 
            bool isdefault = false;
            if (i->second.has_entry("isdefault")) {
                std::string tmp = i->second.get_entry("isdefault");
                boost::to_lower (tmp);
                if (tmp == "true") 
                    isdefault = true;
            }

            fs::path lib;
            try {
                if (i->second.has_entry("path"))
                    lib = fs::path(i->second.get_entry("path"), fs::native);
                else
                    lib = fs::path(HPX_DEFAULT_COMPONENT_PATH, fs::native);

                if (!load_component(ini, instance, component, lib, prefix, agas_client, isdefault)) {
                    // build path to component to load
                    std::string libname(component + HPX_SHARED_LIB_EXTENSION);
                    lib /= fs::path(libname, fs::native);
                    if (!load_component (ini, instance, component, lib, prefix, agas_client, isdefault))
                        continue;   // next please :-P
                }
            } 
            catch (hpx::exception const& /*e*/) {
                ; // FIXME: use default component location
            }
        } // for
    }

    bool runtime_support::load_component(util::section& ini, 
        std::string const& instance, std::string const& component, 
        boost::filesystem::path lib, naming::id_type const& prefix, 
        naming::resolver_client& agas_client, bool isdefault)
    {
        namespace fs = boost::filesystem;
        if (fs::extension(lib) != HPX_SHARED_LIB_EXTENSION)
            return false;

        try {
            // get the handle of the library 
            boost::plugin::dll d (lib.string(), component);

            // get the factory
            boost::plugin::plugin_factory<component_factory_base> pf (d, 
                BOOST_PP_STRINGIZE(HPX_MANGLE_COMPONENT_NAME(factory)));

            // initialize the factory instance using the preferences from the 
            // ini files
            util::section const* glob_ini = NULL;
            if (ini.has_section("settings"))
                glob_ini = ini.get_section("settings");

            util::section const* component_ini = NULL;
            std::string component_section("hpx.components." + instance);
            if (ini.has_section(component_section))
                component_ini = ini.get_section(component_section);

            // create the component factory object
            component_factory_type factory (
                pf.create(instance, glob_ini, component_ini)); 

            component_type t = factory->get_component_type(prefix, agas_client);
            if (0 == t) {
                LRT_(info) << "component refused to load: "  << instance;
                return false;   // module refused to load
            }

            // store component factory for later use
            std::pair<component_map_type::iterator, bool> p = 
                components_.insert(component_map_type::value_type(t, factory));

            if (!p.second) {
                LRT_(error) << "duplicate component id: " << instance
                           << ": " << components::get_component_type_name(t);
                return false;   // duplicate component id?
            }

            // Store the reference to the shared library if everything is fine.
            // We store the library at front of the list so we can unload the 
            // modules in reverse order.
            modules_.push_front(d); 

            LRT_(info) << "dynamic loading succeeded: " << lib.string() 
                       << ": " << instance << ": " 
                       << components::get_component_type_name(t);
        }
        catch (std::logic_error const& e) {
            if (!isdefault) {
                LRT_(warning) << "dynamic loading failed: " << lib.string() 
                            << ": " << instance << ": " << e.what();
            }
            return false;
        }
        return true;    // component got loaded
    }

}}}

