//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_FACTORY_JUN_03_2008_0438PM)
#define HPX_COMPONENTS_FACTORY_JUN_03_2008_0438PM

#include <hpx/naming/name.hpp>
#include <hpx/applier/applier.hpp>
#include <hpx/components/stubs/factory.hpp>

namespace hpx { namespace components 
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a factory class is the client side representation of a 
    /// \a server#factory component
    class factory : public stubs::factory
    {
        typedef stubs::factory base_type;
        
    public:
        /// Create a client side representation for the existing
        /// \a server#factory instance with the given global id \a gid.
        factory(applier::applier& app, naming::id_type gid) 
          : base_type(app), gid_(gid)
        {}
        
        ~factory() 
        {}
        
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        
        /// Create a new component using the factory 
        void create(components::component_type type, naming::id_type newgid) 
        {
            this->base_type::create(gid_, type, newgid);
        }
        
    private:
        naming::id_type gid_;
    };
    
}}

#endif
