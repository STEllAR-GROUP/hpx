//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_RUNTIME_SUPPORT_JUN_09_2008_0503PM)
#define HPX_COMPONENTS_STUBS_RUNTIME_SUPPORT_JUN_09_2008_0503PM

#include <boost/bind.hpp>

#include <hpx/runtime/applier/applier.hpp>
#include <hpx/components/server/runtime_support.hpp>
#include <hpx/lcos/simple_future.hpp>

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////
    // The \a runtime_support class is the client side representation of a 
    // \a server#runtime_support component
    class runtime_support
    {
    public:
        /// Create a client side representation for any existing 
        /// \a server#runtime_support instance
        runtime_support(applier::applier& app) 
          : app_(app)
        {}

        ~runtime_support() 
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Create a new component \a type using the runtime_support with the 
        /// given \a targetgid. This is a non-blocking call. The caller needs 
        /// to call \a simple_future#get_result on the result of this function 
        /// to obtain the global id of the newly created object.
        static lcos::simple_future<naming::id_type> create_component_async(
            applier::applier& appl, naming::id_type const& targetgid, 
            components::component_type type, std::size_t count = 1) 
        {
            // Create a simple_future, execute the required action and wait 
            // for the result to be returned to the future.
            lcos::simple_future<naming::id_type> lco;

            // The simple_future instance is associated with the following 
            // apply action by sending it along as its continuation
            appl.apply<server::runtime_support::create_component_action>(
                new components::continuation(lco.get_gid(appl)), 
                targetgid, type, count);

            // we simply return the initialized simple_future, the caller needs
            // to call get_result() on the return value to obtain the result
            return lco;
        }

        /// Create a new component \a type using the runtime_support with the 
        /// given \a targetgid. Block for the creation to finish.
        static naming::id_type create_component(
            threadmanager::px_thread_self& self, applier::applier& appl, 
            naming::id_type const& targetgid, components::component_type type,
            std::size_t count = 1) 
        {
            // The following get_result yields control while the action above 
            // is executed and the result is returned to the simple_future
            return create_component_async(appl, targetgid, type, count)
                .get_result(self);
        }

        ///
        lcos::simple_future<naming::id_type> create_component_async(
            naming::id_type const& targetgid, components::component_type type,
            std::size_t count = 1) 
        {
            return create_component_async(app_, targetgid, type, count);
        }

        /// 
        naming::id_type create_component(threadmanager::px_thread_self& self,
            naming::id_type const& targetgid, components::component_type type,
            std::size_t count = 1) 
        {
            return create_component(self, app_, targetgid, type, count);
        }

        /// Destroy an existing component
        static void free_component(applier::applier& appl, 
            naming::id_type const& targetgid, components::component_type type, 
            naming::id_type const& gid, std::size_t count = 1) 
        {
            typedef server::runtime_support::free_component_action action_type;
            appl.apply<action_type>(targetgid, type, gid, count);
        }

        void free_component(naming::id_type const& targetgid, 
            components::component_type type, naming::id_type const& gid,
            std::size_t count = 1)
        {
            free_component(app_, targetgid, type, gid, count);
        }

        /// \brief Shutdown the given runtime system
        static void 
        shutdown(applier::applier& appl, naming::id_type const& targetgid)
        {
            appl.apply<server::runtime_support::shutdown_action>(targetgid);
        }

        void shutdown(naming::id_type const& targetgid)
        {
            shutdown(app_, targetgid);
        }

        /// \brief Shutdown the runtime systems of all localities
        static void 
        shutdown_all(applier::applier& appl, naming::id_type const& targetgid)
        {
            appl.apply<server::runtime_support::shutdown_all_action>(targetgid);
        }

        void shutdown_all(naming::id_type const& targetgid)
        {
            shutdown_all(app_, targetgid);
        }

    private:
        applier::applier& app_;
    };

}}}

#endif
