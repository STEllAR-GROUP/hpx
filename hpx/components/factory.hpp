//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_FACTORY_JUN_03_2008_0438PM)
#define HPX_COMPONENTS_FACTORY_JUN_03_2008_0438PM

#include <hpx/naming/name.hpp>
#include <hpx/applier/applier.hpp>
#include <hpx/components/server/factory.hpp>

namespace hpx { namespace components 
{
    ///////////////////////////////////////////////////////////////////////////
    // The \a factory class is the client side representation of a 
    // \a server#factory component
    class factory
    {
    public:
        /// Create a client side representation for the existing
        /// \a server#accumulator instance with the given global id \a gid.
        factory(applier::applier& app, naming::id_type gid) 
          : app_(app), gid_(gid)
        {}
        
        ~factory() 
        {}
        
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        
        /// Create a new component using the factory 
        void create(components::component_type type, naming::id_type gid) 
        {
            app_.apply<server::factory::create_action>(gid_, type, gid);
        }
        
    private:
        naming::id_type gid_;
        applier::applier& app_;
    };
    
}}

#endif
