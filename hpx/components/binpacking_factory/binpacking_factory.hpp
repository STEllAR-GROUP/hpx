//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_BINPACKING_FACTORY_MAY_23_2012_1122AM)
#define HPX_COMPONENTS_BINPACKING_FACTORY_MAY_23_2012_1122AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/components/binpacking_factory/stubs/binpacking_factory.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    // The \a binpacking_factory class is the client side representation of a
    // concrete \a server#binpacking_factory component
    class binpacking_factory
      : public client_base<binpacking_factory, stubs::binpacking_factory>
    {
    private:
        typedef client_base<binpacking_factory, stubs::binpacking_factory>
            base_type;

    public:
        binpacking_factory()
          : base_type(naming::invalid_id)
        {}

        /// Create a client side representation for any existing
        /// \a server#binpacking_factoryinstance with the given global id \a gid.
        binpacking_factory(naming::id_type gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        typedef base_type::result_type result_type;
        typedef base_type::remote_result_type remote_result_type;
        typedef base_type::iterator_type iterator_type;
        typedef base_type::iterator_range_type iterator_range_type;

        typedef lcos::future<result_type, remote_result_type>
            async_create_result_type;

        ///////////////////////////////////////////////////////////////////////
        ///
        async_create_result_type create_components_async(
            components::component_type type, std::size_t count = 1)
        {
            return this->base_type::create_components_async(gid_, type, count);
        }

        ///
        result_type create_components(components::component_type type,
            std::size_t count = 1)
        {
            return this->base_type::create_components(gid_, type, count);
        }
    };
}}

#endif
