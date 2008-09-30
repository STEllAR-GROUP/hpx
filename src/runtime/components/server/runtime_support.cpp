//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/init_ini_data.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/runtime/components/server/manage_component.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/convenience.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the runtime_support actions
HPX_SERIALIZE_ACTION(hpx::components::server::runtime_support::create_component_action);
HPX_SERIALIZE_ACTION(hpx::components::server::runtime_support::free_component_action);
HPX_SERIALIZE_ACTION(hpx::components::server::runtime_support::shutdown_action);
HPX_SERIALIZE_ACTION(hpx::components::server::runtime_support::shutdown_all_action);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    // create a new instance of a component
    threads::thread_state runtime_support::create_component(
        threads::thread_self& self, applier::applier& appl,
        naming::id_type* gid, components::component_type type,
        std::size_t count)
    {
    // create new component instance
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            // we don't know anything about this component
            boost::throw_exception(hpx::exception(hpx::bad_component_type,
                std::string("attempt to create component instance of invalid type: ") + 
                    boost::lexical_cast<std::string>((int)type)));
            return threads::terminated;
        }
        naming::id_type id = (*it).second->create(appl, count);

    // set result if requested
        if (0 != gid)
            *gid = id;
        return threads::terminated;
    }

    // delete an existing instance of a component
    threads::thread_state runtime_support::free_component(
        threads::thread_self& self, applier::applier& appl,
        components::component_type type, naming::id_type const& gid,
        std::size_t count)
    {
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            // we don't know anything about this component
            boost::throw_exception(hpx::exception(hpx::bad_component_type,
                std::string("attempt to destroy component instance of invalid type: ") + 
                    boost::lexical_cast<std::string>((int)type)));
            return threads::terminated;
        }

        (*it).second->destroy(appl, gid, count);
        return threads::terminated;
    }

    /// \brief Action shut down this runtime system instance
    threads::thread_state runtime_support::shutdown(
        threads::thread_self& self, applier::applier& app)
    {
        // initiate system shutdown
        stop();
        return threads::terminated;
    }

    // initiate system shutdown for all localities
    threads::thread_state runtime_support::shutdown_all(
        threads::thread_self& self, applier::applier& app)
    {
        std::vector<naming::id_type> prefixes;
        app.get_dgas_client().get_prefixes(prefixes);

        // shut down all localities except the the local one
        components::stubs::runtime_support rts(app);
        std::vector<naming::id_type>::iterator end = prefixes.end();
        for (std::vector<naming::id_type>::iterator it = prefixes.begin(); 
             it != end; ++it)
        {
            if (naming::get_prefix_from_id(app.get_prefix()) !=
                naming::get_prefix_from_id(*it))
            {
                rts.shutdown(*it);
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
        stopped_ = false;
        condition_.wait(l);
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
    // iterate over all shared libraries in the given directory and construct
    // default ini settings assuming all of those are components
    inline void init_ini_data_default(std::string const& libs, util::section& ini)
    {
        namespace fs = boost::filesystem;

        try {
            fs::directory_iterator nodir;
            fs::path libs_path (libs, fs::native);

            if (!fs::exists(libs_path)) 
                return;     // give directory doesn't exist

            for (fs::directory_iterator dir(libs_path); dir != nodir; ++dir)
            {
                fs::path curr(*dir);
                if (fs::extension(curr) != HPX_SHARED_LIB_EXTENSION) 
                    continue;

                // instance name and module name are the same
                std::string name (fs::basename(curr));
#if defined(BOOST_WINDOWS) && defined(_DEBUG)
                // remove the 'd' suffix 
                if (name[name.size()-1] == 'd')
                    name = name.substr(0, name.size()-1);
#endif

                if (!ini.has_section("hpx.components")) {
                    util::section* hpx_sec = ini.get_section("hpx");
                    BOOST_ASSERT(NULL != hpx_sec);

                    util::section comp_sec;
                    hpx_sec->add_section("components", comp_sec);
                }

                util::section* components_sec = ini.get_section("hpx.components");
                util::section sec;
                sec.add_entry("name", name);
                sec.add_entry("path", libs);
                components_sec->add_section(name, sec);
            }
        }
        catch (fs::filesystem_error const& /*e*/) {
            ;
        }
     }

    ///////////////////////////////////////////////////////////////////////////
    // Load all components from the ini files found in the configuration
    void runtime_support::load_components(naming::resolver_client& dgas_client)
    {
        util::section ini;
        if (!util::init_ini_data(ini)) {
            // no ini files found, try to build default ini structure from
            // shared libraries in default installation location
            init_ini_data_default(HPX_DEFAULT_COMPONENT_PATH, ini);
        }
        else {
            // merge all found ini files of all components
            util::merge_component_inis(ini);

            // read system and user ini files _again_, to allow the user to 
            // overwrite the settings from the default component ini's. 
            util::init_ini_data_base(ini);
        }

        // now, load all components
        if (!ini.has_section("hpx.components"))
            return;     // no components to load

        // each shared library containing components must have a ini section
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

            bool enabled = true;
            if (i->second.has_entry("enabled")) {
                std::string tmp = i->second.get_entry("enabled");
                boost::to_lower (tmp);
                if (tmp == "no" || tmp == "false" || tmp == "0")
                    enabled = false;
            }

            fs::path lib;
            try {
                if (i->second.has_entry("path"))
                    lib = fs::path(i->second.get_entry("path"), fs::native);
                else
                    lib = fs::path(HPX_DEFAULT_COMPONENT_PATH, fs::native);

                if (!enabled) 
                    continue;

                if (!load_component(ini, instance, component, lib, dgas_client)) {
                    // build path to component to load
                    std::string libname(component + HPX_SHARED_LIB_EXTENSION);
                    lib /= fs::path(libname, fs::native);
                    if (!load_component (ini, instance, component, lib, dgas_client))
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
        boost::filesystem::path lib, naming::resolver_client& dgas_client)
    {
        namespace fs = boost::filesystem;
        if (fs::extension(lib) != HPX_SHARED_LIB_EXTENSION)
            return false;

        try {
            // get the handle of the library 
            boost::plugin::dll d (lib.string(), component);

            // get the factory
            boost::plugin::plugin_factory<component_factory_base> pf (d, "factory");

            // initialize the factory instance using the preferences from the 
            // ini files
            util::section const* glob_ini = NULL;
            if (ini.has_section("settings"))
                glob_ini = ini.get_section("settings");

            util::section const* component_ini = NULL;
            std::string component_section("hpx.components." + instance);
            if (ini.has_section(component_section))
                component_ini = ini.get_section(component_section);

            std::string component_name(component);

#if defined(BOOST_WINDOWS) && defined(_DEBUG)
            // remove the 'd' suffix 
            if (component_name[component_name.size()-1] == 'd')
                component_name = component_name.substr(0, component_name.size()-1);
#endif

            // create the component factory object
            component_factory_type factory (
                pf.create(component_name, glob_ini, component_ini)); 

            component_type t = factory->get_component_type(dgas_client);
            if (0 == t) 
                return false;   // module refused to load

            // store component factory for later use
            std::pair<component_map_type::iterator, bool> p = 
                components_.insert(component_map_type::value_type(t, factory));

            if (p.second) 
                return false;   // duplicate component id?

            // store the reference to the shared library if everything is fine
            modules_.push_back (d); 
        }
        catch (std::logic_error const& /*e*/) {
            return false;
        }
        return true;    // component got loaded
    }

}}}

