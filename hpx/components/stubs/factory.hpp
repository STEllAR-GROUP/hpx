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

        ///////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Create a new component using the factory with the given \a 
        /// targetgid
        static naming::id_type create(
            threadmanager::px_thread_self& self, applier::applier& appl, 
            naming::id_type targetgid, components::component_type type) 
        {
            // the following assignment of the function pointer to a variable
            // is needed to help th ecompiler to deduce the correct required 
            // function to bind to the supplied arguments.
            parcelset::parcel_id (applier::applier::*f)(
                    components::continuation*, naming::id_type, 
                    components::component_type const&) = 
                &applier::applier::apply<server::factory::create_action, 
                    components::component_type>;

            // create a simple_future, which executes the supplied function
            // during construction time
            lcos::simple_future<naming::id_type> lco (
                self, boost::bind(f, boost::ref(appl), _1, targetgid, type)
            );
            return lco.get_result(self);
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
