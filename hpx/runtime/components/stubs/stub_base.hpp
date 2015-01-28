//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUB_STUB_BASE_CLIENT_OCT_31_2008_0441PM)
#define HPX_COMPONENTS_STUB_STUB_BASE_CLIENT_OCT_31_2008_0441PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/util/move.hpp>

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
        template <typename ...Ts>
        static lcos::future<naming::id_type>
        create_async(naming::id_type const& gid, Ts&&... vs)
        {
            using stubs::runtime_support;
            return runtime_support::create_component_async<ServerComponent>(
                gid, std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        static lcos::future<std::vector<naming::id_type> >
        bulk_create_async(naming::id_type const& gid, std::size_t count,
            Ts&&... vs)
        {
            using stubs::runtime_support;
            return runtime_support::bulk_create_component_async<
                    ServerComponent
                >(gid, count, std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        static naming::id_type create(
            naming::id_type const& gid, Ts&&... vs)
        {
            using stubs::runtime_support;
            return runtime_support::create_component<ServerComponent>(
                gid, std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        static std::vector<naming::id_type> bulk_create(
            naming::id_type const& gid, std::size_t count, Ts&&... vs)
        {
            using stubs::runtime_support;
            return runtime_support::bulk_create_component<ServerComponent>(
                gid, count, std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        static lcos::future<naming::id_type>
        create_colocated_async(naming::id_type const& gid, Ts&&... vs)
        {
            using stubs::runtime_support;
            return runtime_support::create_component_colocated_async<
                ServerComponent>(gid, std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        static naming::id_type create_colocated(
            naming::id_type const& gid, Ts&&... vs)
        {
            using stubs::runtime_support;
            return runtime_support::create_component_colocated<
                ServerComponent>(gid, std::forward<Ts>(vs)...);
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
    };
}}

#endif

