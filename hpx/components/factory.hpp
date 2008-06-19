//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_FACTORY_JUN_03_2008_0438PM)
#define HPX_COMPONENTS_FACTORY_JUN_03_2008_0438PM

#include <hpx/runtime/applier/applier.hpp>
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
        naming::id_type create(threadmanager::px_thread_self& self,
            components::component_type type, std::size_t count = 1) 
        {
            return this->base_type::create(self, gid_, type, count);
        }

        /// Asynchronously create a new component using the factory 
        lcos::simple_future<naming::id_type> 
        create_async(threadmanager::px_thread_self& self,
            components::component_type type, std::size_t count = 1) 
        {
            return this->base_type::create_async(self, gid_, type, count);
        }

        /// Destroy an existing component
        void free (components::component_type type, naming::id_type const& gid)
        {
            this->base_type::free(gid_, type, gid);
        }

    private:
        naming::id_type gid_;
    };

}}

#endif
