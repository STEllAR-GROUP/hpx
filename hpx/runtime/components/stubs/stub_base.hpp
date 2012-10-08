//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUB_STUB_BASE_CLIENT_OCT_31_2008_0441PM)
#define HPX_COMPONENTS_STUB_STUB_BASE_CLIENT_OCT_31_2008_0441PM

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
        static lcos::future<naming::id_type, naming::gid_type>
        create_async(naming::id_type const& gid)
        {
            return stubs::runtime_support::create_component_async<ServerComponent>
                (gid);
        }

        static naming::id_type create(naming::id_type const& gid)
        {
            return stubs::runtime_support::create_component<ServerComponent>(gid);
        }

        ///////////////////////////////////////////////////////////////////////
#define HPX_STUB_BASE_CREATE(Z, N, D)                                        \
        template <BOOST_PP_ENUM_PARAMS(N, typename A)>                       \
        static lcos::future<naming::id_type, naming::gid_type>               \
        create_async(naming::id_type const& gid,                             \
            BOOST_PP_ENUM_BINARY_PARAMS(N, A, a))                            \
        {                                                                    \
            return stubs::runtime_support::create_component_async<ServerComponent>\
                (gid, HPX_ENUM_MOVE_IF_NO_REF_ARGS(N, A, a));                \
        }                                                                    \
                                                                             \
        template <BOOST_PP_ENUM_PARAMS(N, typename A)>                       \
        static naming::id_type create(                                       \
            naming::id_type const& gid,                                      \
                BOOST_PP_ENUM_BINARY_PARAMS(N, A, a))                        \
        {                                                                    \
            return stubs::runtime_support::create_component<ServerComponent> \
                (gid, HPX_ENUM_MOVE_IF_NO_REF_ARGS(N, A, a));                \
        }                                                                    \
    /**/

        BOOST_PP_REPEAT_FROM_TO(
            1
          , HPX_ACTION_ARGUMENT_LIMIT
          , HPX_STUB_BASE_CREATE
          , _
        )

#undef HPX_STUB_BASE_CREATE

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
    };
}}

#endif

