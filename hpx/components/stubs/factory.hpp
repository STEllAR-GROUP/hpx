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
        /// targetgid
        static naming::id_type create(
            threadmanager::px_thread_self& self, applier::applier& appl, 
            naming::id_type targetgid, components::component_type type) 
        {
            // Create a simple_future, execute the required action and wait 
            // for the result to be returned to the future.
            lcos::simple_future<naming::id_type> lco (self);

            // The simple_future instance is associated with the following 
            // apply action by sending it along as its continuation
            appl.apply<server::factory::create_action>(
                new components::continuation(lco.get_gid()), targetgid, type);

            // The following get_result unconditionally yields control while
            // the action above is executed and the result is returned to the 
            // simple_future
            return lco.get_result();
        }

        naming::id_type create(threadmanager::px_thread_self& self,
            naming::id_type targetgid, components::component_type type) 
        {
            return create(self, app_, targetgid, type);
        }

    private:
        applier::applier& app_;
    };

}}}

#endif
