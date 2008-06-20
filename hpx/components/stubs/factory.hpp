//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_FACTORY_JUN_09_2008_0503PM)
#define HPX_COMPONENTS_STUBS_FACTORY_JUN_09_2008_0503PM

#include <boost/bind.hpp>

#include <hpx/runtime/applier/applier.hpp>
#include <hpx/components/server/factory.hpp>
#include <hpx/lcos/simple_future.hpp>

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////
    // The \a factory class is the client side representation of a 
    // \a server#factory component
    class factory
    {
    public:
        /// Create a client side representation for any existing 
        /// \a server#factory instance
        factory(applier::applier& app) 
          : app_(app)
        {}

        ~factory() 
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Create a new component \a type using the factory with the given \a 
        /// targetgid. This is a non-blocking call. The caller needs to call 
        /// \a simple_future#get_result on the result of this function to 
        /// obtain the global id of the newly created object.
        static lcos::simple_future<naming::id_type> create_async(
            threadmanager::px_thread_self& self, applier::applier& appl, 
            naming::id_type const& targetgid, components::component_type type,
            std::size_t count = 1) 
        {
            // Create a simple_future, execute the required action and wait 
            // for the result to be returned to the future.
            lcos::simple_future<naming::id_type> lco (self);

            // The simple_future instance is associated with the following 
            // apply action by sending it along as its continuation
            appl.apply<server::factory::create_action>(
                new components::continuation(lco.get_gid(appl)), 
                targetgid, type, count);

            // we simply return the initialized simple_future, the caller needs
            // to call get_result() on the return value to obtain the result
            return lco;
        }

        /// Create a new component \a type using the factory with the given \a 
        /// targetgid. Block for the creation to finish.
        static naming::id_type create(
            threadmanager::px_thread_self& self, applier::applier& appl, 
            naming::id_type const& targetgid, components::component_type type,
            std::size_t count = 1) 
        {
            // The following get_result yields control while the action above 
            // is executed and the result is returned to the simple_future
            return create_async(self, appl, targetgid, type, count).get_result();
        }

        ///
        lcos::simple_future<naming::id_type> create_async(
            threadmanager::px_thread_self& self, 
            naming::id_type const& targetgid, components::component_type type,
            std::size_t count = 1) 
        {
            return create_async(self, app_, targetgid, type, count);
        }

        /// 
        naming::id_type create(threadmanager::px_thread_self& self,
            naming::id_type const& targetgid, components::component_type type,
            std::size_t count = 1) 
        {
            return create(self, app_, targetgid, type, count);
        }

        /// Destroy an existing component
        static void free(applier::applier& appl, 
            naming::id_type const& targetgid, components::component_type type, 
            naming::id_type const& gid, std::size_t count = 1) 
        {
            appl.apply<server::factory::free_action>(targetgid, type, gid, count);
        }

        void free (naming::id_type const& targetgid, 
            components::component_type type, naming::id_type const& gid,
            std::size_t count = 1)
        {
            free(app_, targetgid, type, gid, count);
        }

    private:
        applier::applier& app_;
    };

}}}

#endif
