//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_FACTORY_JUN_09_2008_0503PM)
#define HPX_COMPONENTS_STUBS_FACTORY_JUN_09_2008_0503PM

#include <hpx/runtime/runtime.hpp>
#include <hpx/components/server/factory.hpp>

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    // The \a factory class is the client side representation of a 
    // \a server#factory component
    class factory
    {
    public:
        /// Create a client side representation for any existing 
        /// \a server#factory instance
        factory(runtime& rt) 
          : rt_(rt)
        {}
        
        ~factory() 
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Create a new component using the factory with the given \a targetgid
        void create(naming::id_type targetgid, components::component_type type, 
            naming::id_type newgid) 
        {
            rt_.get_applier().apply<server::factory::create_action>(
                targetgid, type, newgid);
        }

    private:
        runtime& rt_;
    };

}}}

#endif
