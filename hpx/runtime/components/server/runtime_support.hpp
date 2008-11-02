//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_SUPPORT_JUN_02_2008_1145AM)
#define HPX_RUNTIME_SUPPORT_JUN_02_2008_1145AM

#include <map>
#include <list>

#include <boost/thread.hpp>
#include <boost/thread/condition.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/plugin.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/component_factory_base.hpp>
#include <hpx/runtime/actions/action.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class runtime_support
    {
    private:
        typedef boost::mutex mutex_type;
        typedef std::list<boost::plugin::dll> module_list_type;
        typedef boost::shared_ptr<component_factory_base> component_factory_type;
        typedef std::map<component_type, component_factory_type> component_map_type;

    public:
        // parcel action code: the action to be performed on the destination 
        // object 
        enum actions
        {
            runtime_support_has_multi_instance_factory = 0,  ///< return whether more than 
                                                    ///< one instance of a component can be created at once
            runtime_support_create_component = 1,   ///< create new components
            runtime_support_free_component = 2,     ///< delete existing components
            runtime_support_shutdown = 3,           ///< shut down this runtime instance
            runtime_support_shutdown_all = 4,       ///< shut down the runtime instances of all localities
        };

        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        static component_type get_component_type() 
        { 
            return component_runtime_support; 
        }

        // constructor
        runtime_support(util::section& ini, naming::resolver_client& dgas_client) 
          : stopped_(false)
        {
            load_components(ini, dgas_client);
        }

        ~runtime_support()
        {
            tidy();
        }

        void tidy()
        {
            components_.clear();    // make sure components get released first

            // Only after releasing the components we are allowed to release 
            // the modules. This is done in reverse order of loading.
            module_list_type::iterator end = modules_.end();
            for (module_list_type::iterator it = modules_.begin(); it != end; /**/)
            {
                module_list_type::iterator curr = it;
                ++it;
                modules_.erase(curr);
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// \brief Action to figure out, whether we can create more than one 
        ///        instance at once
        threads::thread_state has_multi_instance_factory(
            threads::thread_self& self, applier::applier& app,
            bool* has_multi_instance_factory, components::component_type type); 

        /// \brief Action to create new components
        threads::thread_state create_component(
            threads::thread_self& self, applier::applier& app,
            naming::id_type* gid, components::component_type type, 
            std::size_t count); 

        /// \brief Action to delete existing components
        threads::thread_state free_component(
            threads::thread_self& self, applier::applier& app,
            components::component_type type, naming::id_type const& gid,
            std::size_t count); 

        /// \brief Action shut down this runtime system instance
        threads::thread_state shutdown(
            threads::thread_self& self, applier::applier& app);

        /// \brief Action shut down runtime system instances on all localities
        threads::thread_state shutdown_all(
            threads::thread_self& self, applier::applier& app);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into a action
        // type, allowing to generate all require boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action1<
            runtime_support, bool, runtime_support_has_multi_instance_factory, 
            components::component_type, 
            &runtime_support::has_multi_instance_factory
        > has_multi_instance_factory_action;

        typedef hpx::actions::result_action2<
            runtime_support, naming::id_type, runtime_support_create_component, 
            components::component_type, std::size_t, 
            &runtime_support::create_component
        > create_component_action;

        typedef hpx::actions::action3<
            runtime_support, runtime_support_free_component, 
            components::component_type, naming::id_type const&, std::size_t, 
            &runtime_support::free_component
        > free_component_action;

        typedef hpx::actions::action0<
            runtime_support, runtime_support_shutdown, 
            &runtime_support::shutdown
        > shutdown_action;

        typedef hpx::actions::action0<
            runtime_support, runtime_support_shutdown_all, 
            &runtime_support::shutdown_all
        > shutdown_all_action;

        /// \brief Wait for the runtime_support component to notify the calling
        ///        thread.
        ///
        /// This function will be called from the main thread, causing it to
        /// block while the HPX functionality is executed. The main thread will
        /// block until the shutdown_action is executed, which in turn notifies
        /// all waiting threads.
        void wait();

        /// \brief Notify all waiting (blocking) threads allowing the system to 
        ///        be properly stopped.
        ///
        /// \note      This function can be called from any thread.
        void stop();

    protected:
        // Load all components from the ini files found in the configuration
        void load_components(util::section& ini, 
            naming::resolver_client& dgas_client);
        bool load_component(util::section& ini, std::string const& instance, 
            std::string const& component, boost::filesystem::path lib,
            naming::resolver_client& dgas_client, bool isdefault);

    private:
        mutex_type mtx_;
        boost::condition condition_;
        bool stopped_;

        component_map_type components_;
        module_list_type modules_;
    };

}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
