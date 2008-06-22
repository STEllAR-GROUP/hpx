//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_RUNTIME_SUPPORT_JUN_03_2008_0438PM)
#define HPX_COMPONENTS_RUNTIME_SUPPORT_JUN_03_2008_0438PM

#include <hpx/runtime/applier/applier.hpp>
#include <hpx/components/stubs/runtime_support.hpp>

namespace hpx { namespace components 
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a runtime_support class is the client side representation of a 
    /// \a server#runtime_support component
    class runtime_support : public stubs::runtime_support
    {
        typedef stubs::runtime_support base_type;

    public:
        /// Create a client side representation for the existing
        /// \a server#runtime_support instance with the given global id \a gid.
        runtime_support(applier::applier& app, naming::id_type gid) 
          : base_type(app), gid_(gid)
        {
            BOOST_ASSERT(gid_);
        }

        ~runtime_support() 
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Create a new component using the runtime_support 
        naming::id_type create_component(threadmanager::px_thread_self& self,
            components::component_type type, std::size_t count = 1) 
        {
            return this->base_type::create_component(self, gid_, type, count);
        }

        /// Asynchronously create a new component using the runtime_support 
        lcos::simple_future<naming::id_type> 
        create_component_async(components::component_type type, 
            std::size_t count = 1) 
        {
            return this->base_type::create_component_async(gid_, type, count);
        }

        /// Destroy an existing component
        void free_component (components::component_type type, 
            naming::id_type const& gid, std::size_t count = 1)
        {
            this->base_type::free_component(gid_, type, gid, count);
        }

        /// \brief Shutdown the given runtime system
        void shutdown()
        {
            this->base_type::shutdown(gid_);
        }

        /// \brief Shutdown the runtime systems of all localities
        void shutdown_all()
        {
            this->base_type::shutdown_all(gid_);
        }

    private:
        naming::id_type gid_;
    };

}}

#endif
