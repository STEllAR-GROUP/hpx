//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_STUB_BASE_CLIENT_OCT_31_2008_0441PM)
#define HPX_COMPONENTS_STUBS_STUB_BASE_CLIENT_OCT_31_2008_0441PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ServerComponent>
    struct stub_base
    {
        typedef ServerComponent server_component_type;

        ///////////////////////////////////////////////////////////////////////
        // expose component type of the corresponding server component
        static components::component_type get_component_type()
        {
            return components::get_component_type<ServerComponent>();
        }

        ///////////////////////////////////////////////////////////////////////
        /// Asynchronously create a new instance of a component
        static lcos::promise<naming::id_type, naming::gid_type>
        create_async(naming::gid_type const& gid,
            component_type type, std::size_t count = 1)
        {
            return stubs::runtime_support::create_component_async(gid, type, count);
        }

        static lcos::promise<naming::id_type, naming::gid_type>
        create_async(naming::gid_type const& gid, std::size_t count = 1)
        {
            return create_async(gid, get_component_type(), count);
        }

        static lcos::promise<naming::id_type, naming::gid_type>
        create_async(naming::id_type const& gid,
            component_type type, std::size_t count = 1)
        {
            return stubs::runtime_support::create_component_async(
                gid.get_gid(), type, count);
        }

        static lcos::promise<naming::id_type, naming::gid_type>
        create_async(naming::id_type const& gid, std::size_t count = 1)
        {
            return create_async(gid.get_gid(), get_component_type(), count);
        }

        /// Create a new instance of an simple_accumulator
        static naming::id_type
        create_sync(naming::gid_type const& gid, component_type type,
            std::size_t count = 1)
        {
            return stubs::runtime_support::create_component(gid, type, count);
        }

        static naming::id_type
        create_sync(naming::gid_type const& gid, std::size_t count = 1)
        {
            return create_sync(gid, get_component_type(), count);
        }

        static naming::id_type
        create_sync(naming::id_type const& gid, component_type type,
            std::size_t count = 1)
        {
            return stubs::runtime_support::create_component(
                gid.get_gid(), type, count);
        }

        static naming::id_type
        create_sync(naming::id_type const& gid, std::size_t count = 1)
        {
            return create_sync(gid.get_gid(), get_component_type(), count);
        }

        ///////////////////////////////////////////////////////////////////////
        /// Asynchronously create a new instance of a component while passing
        /// one argument to it's constructor
        template <typename Arg0>
        static lcos::promise<naming::id_type, naming::gid_type>
        create_one_async(naming::gid_type const& gid, component_type type,
            Arg0 const& arg0)
        {
            return stubs::runtime_support::create_one_component_async(gid, type, arg0);
        }

        template <typename Arg0>
        static lcos::promise<naming::id_type, naming::gid_type>
        create_one_async(naming::gid_type const& gid, Arg0 const& arg0)
        {
            return create_one_async(gid, get_component_type(), arg0);
        }

        template <typename Arg0>
        static lcos::promise<naming::id_type, naming::gid_type>
        create_one_async(naming::id_type const& gid, component_type type,
            Arg0 const& arg0)
        {
            return stubs::runtime_support::create_one_component_async(
                gid.get_gid(), type, arg0);
        }

        template <typename Arg0>
        static lcos::promise<naming::id_type, naming::gid_type>
        create_one_async(naming::id_type const& gid, Arg0 const& arg0)
        {
            return create_one_async(gid.get_gid(), get_component_type(), arg0);
        }

        template <typename Arg0>
        static naming::id_type
        create_one_sync(naming::gid_type const& gid, component_type type, Arg0 const& arg0)
        {
            return stubs::runtime_support::create_one_component(gid, type, arg0);
        }

        template <typename Arg0>
        static naming::id_type
        create_one_sync(naming::gid_type const& gid, Arg0 const& arg0)
        {
            return create_one_sync(gid, get_component_type(), arg0);
        }

        template <typename Arg0>
        static naming::id_type
        create_one_sync(naming::id_type const& gid, component_type type, Arg0 const& arg0)
        {
            return stubs::runtime_support::create_one_component(
                gid.get_gid(), type, arg0);
        }

        template <typename Arg0>
        static naming::id_type
        create_one_sync(naming::id_type const& gid, Arg0 const& arg0)
        {
            return create_one_sync(gid.get_gid(), get_component_type(), arg0);
        }

        /// Delete an existing component
        static void
        free(component_type type, naming::id_type& gid)
        {
            gid = naming::invalid_id;
        }

        static void
        free(naming::id_type& gid)
        {
            gid = naming::invalid_id;
        }

        static void
        free_sync(component_type type, naming::id_type& gid)
        {
            gid = naming::invalid_id;
        }

        static void
        free_sync(naming::id_type& gid)
        {
            gid = naming::invalid_id;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace stubs
    {
          // for backwards compatibility only
          using components::stub_base;
    }
}}

#endif

