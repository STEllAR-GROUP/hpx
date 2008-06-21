//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_FACTORY_JUN_02_2008_1145AM)
#define HPX_COMPONENTS_FACTORY_JUN_02_2008_1145AM

#include <boost/thread.hpp>
#include <boost/thread/condition.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/components/component_type.hpp>
#include <hpx/components/action.hpp>

namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class runtime_support
    {
    private:
        typedef boost::mutex mutex_type;

    public:
        // parcel action code: the action to be performed on the destination 
        // object 
        enum actions
        {
            runtime_support_create_component = 0,   ///< create new components
            runtime_support_free_component = 1,     ///< delete existing components
            runtime_support_shutdown = 2,           ///< shut down this runtime instance
            runtime_support_shutdown_all = 3,       ///< shut down the runtime instances of all localities
        };

        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = component_runtime_support };

        // constructor
        runtime_support()
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// \brief Action to create new components
        threadmanager::thread_state create_component(
            threadmanager::px_thread_self& self, applier::applier& app,
            naming::id_type* gid, components::component_type type, 
            std::size_t count); 

        /// \brief Action to delete existing components
        threadmanager::thread_state free_component(
            threadmanager::px_thread_self& self, applier::applier& app,
            components::component_type type, naming::id_type const& gid,
            std::size_t count); 

        /// \brief Action shut down this runtime system instance
        threadmanager::thread_state shutdown(
            threadmanager::px_thread_self& self, applier::applier& app)
        {
            // initiate system shutdown
            condition_.notify_all();
            return threadmanager::terminated;
        }

        /// \brief Action shut down runtime system instances on all localities
        threadmanager::thread_state shutdown_all(
            threadmanager::px_thread_self& self, applier::applier& app);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into a action
        // type, allowing to generate all require boilerplate code for threads,
        // serialization, etc.
        typedef result_action2<
            runtime_support, naming::id_type, runtime_support_create_component, 
            components::component_type, std::size_t, 
            &runtime_support::create_component
        > create_component_action;

        typedef action3<
            runtime_support, runtime_support_free_component, 
            components::component_type, naming::id_type const&, std::size_t, 
            &runtime_support::free_component
        > free_component_action;

        typedef action0<
            runtime_support, runtime_support_shutdown, 
            &runtime_support::shutdown
        > shutdown_action;

        typedef action0<
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
        void wait()
        {
            mutex_type::scoped_lock l(mtx_);
            condition_.wait(l);
        }

    private:
        mutex_type mtx_;
        boost::condition condition_;
    };

}}}

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the runtime_support actions
HPX_SERIALIZE_ACTION(hpx::components::server::runtime_support::create_component_action);
HPX_SERIALIZE_ACTION(hpx::components::server::runtime_support::free_component_action);
HPX_SERIALIZE_ACTION(hpx::components::server::runtime_support::shutdown_action);
HPX_SERIALIZE_ACTION(hpx::components::server::runtime_support::shutdown_all_action);

#endif
