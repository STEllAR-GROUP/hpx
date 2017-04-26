//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUB_STUB_BASE_CLIENT_OCT_31_2008_0441PM)
#define HPX_COMPONENTS_STUB_STUB_BASE_CLIENT_OCT_31_2008_0441PM

#include <hpx/config.hpp>
#include <hpx/lcos/async_fwd.hpp>
#include <hpx/lcos/detail/async_colocated_fwd.hpp>
#include <hpx/lcos/detail/async_implementations_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/throw_exception.hpp>

#include <cstddef>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    namespace server
    {
        template <typename Component, typename ...Ts>
        struct create_component_action;

        template <typename Component, typename ...Ts>
        struct bulk_create_component_action;
    }

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
            if (!naming::is_locality(gid))
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "stubs::runtime_support::create_component_async",
                    "The id passed as the first argument is not representing"
                        " a locality");
                return lcos::make_ready_future(naming::invalid_id);
            }

            typedef server::create_component_action<
                ServerComponent, typename hpx::util::decay<Ts>::type...
            > action_type;
            return hpx::async<action_type>(gid, std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        static lcos::future<std::vector<naming::id_type> >
        bulk_create_async(naming::id_type const& gid, std::size_t count,
            Ts&&... vs)
        {
            if (!naming::is_locality(gid))
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "stubs::runtime_support::bulk_create_component_async",
                    "The id passed as the first argument is not representing"
                        " a locality");
                return lcos::make_ready_future(std::vector<naming::id_type>());
            }

            typedef server::bulk_create_component_action<
                ServerComponent, typename hpx::util::decay<Ts>::type...
            > action_type;
            return hpx::async<action_type>(gid, count,
                std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        static naming::id_type create(
            naming::id_type const& gid, Ts&&... vs)
        {
            return create_async(gid, std::forward<Ts>(vs)...).get();
        }

        template <typename ...Ts>
        static std::vector<naming::id_type> bulk_create(
            naming::id_type const& gid, std::size_t count, Ts&&... vs)
        {
            return bulk_create_async(gid, count, std::forward<Ts>(vs)...).get();
        }

        template <typename ...Ts>
        static lcos::future<naming::id_type>
        create_colocated_async(naming::id_type const& gid, Ts&&... vs)
        {
            typedef server::create_component_action<
                ServerComponent, typename hpx::util::decay<Ts>::type...
            > action_type;
            return hpx::detail::async_colocated<action_type>(
                gid, std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        static naming::id_type create_colocated(
            naming::id_type const& gid, Ts&&... vs)
        {
            return create_colocated_async(gid, std::forward<Ts>(vs)...).get();
        }

        template <typename ...Ts>
        static lcos::future<std::vector<naming::id_type> >
        bulk_create_colocated_async(naming::id_type const& gid,
            std::size_t count, Ts&&... vs)
        {
            typedef server::bulk_create_component_action<
                ServerComponent, typename hpx::util::decay<Ts>::type...
            > action_type;

            return hpx::detail::async_colocated<action_type>(gid, count,
                std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        static std::vector<naming::id_type>
        bulk_create_colocated(naming::id_type const& id, std::size_t count,
            Ts&&... vs)
        {
            return bulk_create_colocated_async(id, count,
                std::forward<Ts>(vs)...).get();
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

