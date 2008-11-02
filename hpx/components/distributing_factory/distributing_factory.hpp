//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_DISTRIBUTING_FACTORY_OCT_31_2008_0329PM)
#define HPX_COMPONENTS_DISTRIBUTING_FACTORY_OCT_31_2008_0329PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/components/distributing_factory/stubs/distributing_factory.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components 
{
    ///////////////////////////////////////////////////////////////////////////
    // The \a distributing_factory class is the client side representation of a 
    // concrete \a server#distributing_factory component
    class distributing_factory 
      : public client_base<distributing_factory, stubs::distributing_factory>
    {
    private:
        typedef 
            client_base<distributing_factory, stubs::distributing_factory>
        base_type;

    public:
        /// Create a client side representation for any existing 
        /// \a server#runtime_support instance with the given global id \a gid.
        distributing_factory(applier::applier& app, naming::id_type gid,
                bool freeonexit = false) 
          : base_type(app, gid, freeonexit)
        {
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        typedef base_type::result_type result_type;

        ///
        lcos::future_value<result_type> create_components_async(
            components::component_type type, std::size_t count = 1) 
        {
            return this->base_type::create_components_async(gid_, type, count);
        }

        /// 
        result_type create_components(threads::thread_self& self,
            components::component_type type, std::size_t count = 1) 
        {
            return this->base_type::create_components(self, gid_, type, count);
        }

        ///
        void free_components(result_type const& gids) 
        {
            this->base_type::free_components(gid_, gids);
        }
    };

}}

#endif
