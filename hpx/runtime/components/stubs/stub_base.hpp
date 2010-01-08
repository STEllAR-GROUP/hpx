//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_STUB_BASE_CLIENT_OCT_31_2008_0441PM)
#define HPX_COMPONENTS_STUBS_STUB_BASE_CLIENT_OCT_31_2008_0441PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ServerComponent>
    struct stub_base 
    {
        ///////////////////////////////////////////////////////////////////////
        /// Asynchronously create a new instance of a distributing_factory
        static lcos::future_value<naming::id_type>
        create_async(naming::id_type const& gid, 
            component_type type, std::size_t count = 1)
        {
            return stubs::runtime_support::create_component_async(gid, type, count);
        }

        static lcos::future_value<naming::id_type>
        create_async(naming::id_type const& gid, std::size_t count = 1)
        {
            return create_async(gid, get_component_type<ServerComponent>(), count);
        }

        /// Create a new instance of an simple_accumulator
        static naming::id_type 
        create(naming::id_type const& gid, component_type type, 
            std::size_t count = 1)
        {
            return stubs::runtime_support::create_component(gid, type, count);
        }

        static naming::id_type 
        create(naming::id_type const& gid, std::size_t count = 1)
        {
            return create(gid, get_component_type<ServerComponent>(), count);
        }

        /// Delete an existing component
        static void
        free(component_type type, naming::id_type const& gid)
        {
            stubs::runtime_support::free_component(type, gid);
        }

        static void
        free(naming::id_type const& gid)
        {
            free(get_component_type<ServerComponent>(), gid);
        }

        static void
        free_sync(component_type type, naming::id_type const& gid)
        {
            stubs::runtime_support::free_component_sync(type, gid);
        }

        static void
        free_sync(naming::id_type const& gid)
        {
            free_sync(get_component_type<ServerComponent>(), gid);
        }
    };

}}}

#endif

