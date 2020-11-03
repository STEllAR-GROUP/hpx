//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_colocated/async_colocated_fwd.hpp>
#include <hpx/async_distributed/detail/async_implementations_fwd.hpp>
#include <hpx/async_local/async_fwd.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/naming_base/id_type.hpp>

#include <cstddef>
#include <utility>
#include <type_traits>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components {

    namespace server {
        template <typename Component, typename... Ts>
        struct create_component_action;

        template <typename Component, typename... Ts>
        struct bulk_create_component_action;
    }    // namespace server

    ///////////////////////////////////////////////////////////////////////////
    /// Asynchronously create a new instance of a component
    template <typename Component, typename... Ts>
    future<naming::id_type> create_async(naming::id_type const& gid, Ts&&... vs)
    {
        if (!naming::is_locality(gid))
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "stubs::runtime_support::create_component_async",
                "The id passed as the first argument is not representing"
                " a locality");
            return make_ready_future(naming::invalid_id);
        }

        using action_type = server::create_component_action<Component,
            typename std::decay<Ts>::type...>;

        return hpx::async<action_type>(gid, std::forward<Ts>(vs)...);
    }

    template <typename Component, typename... Ts>
    future<std::vector<naming::id_type>> bulk_create_async(
        naming::id_type const& gid, std::size_t count, Ts&&... vs)
    {
        if (!naming::is_locality(gid))
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "stubs::runtime_support::bulk_create_component_async",
                "The id passed as the first argument is not representing"
                " a locality");
            return make_ready_future(std::vector<naming::id_type>());
        }

        using action_type = server::bulk_create_component_action<Component,
            typename std::decay<Ts>::type...>;

        return hpx::async<action_type>(gid, count, std::forward<Ts>(vs)...);
    }

    template <typename Component, typename... Ts>
    naming::id_type create(naming::id_type const& gid, Ts&&... vs)
    {
        return create_async<Component>(gid, std::forward<Ts>(vs)...).get();
    }

    template <typename Component, typename... Ts>
    std::vector<naming::id_type> bulk_create(
        naming::id_type const& gid, std::size_t count, Ts&&... vs)
    {
        return bulk_create_async<Component>(gid, count, std::forward<Ts>(vs)...)
            .get();
    }

    template <typename Component, typename... Ts>
    future<naming::id_type> create_colocated_async(
        naming::id_type const& gid, Ts&&... vs)
    {
        using action_type = server::create_component_action<Component,
            typename std::decay<Ts>::type...>;

        return hpx::detail::async_colocated<action_type>(
            gid, std::forward<Ts>(vs)...);
    }

    template <typename Component, typename... Ts>
    static naming::id_type create_colocated(
        naming::id_type const& gid, Ts&&... vs)
    {
        return create_colocated_async(gid, std::forward<Ts>(vs)...).get();
    }

    template <typename Component, typename... Ts>
    static future<std::vector<naming::id_type>> bulk_create_colocated_async(
        naming::id_type const& gid, std::size_t count, Ts&&... vs)
    {
        using action_type = server::bulk_create_component_action<Component,
            typename std::decay<Ts>::type...>;

        return hpx::detail::async_colocated<action_type>(
            gid, count, std::forward<Ts>(vs)...);
    }

    template <typename Component, typename... Ts>
    std::vector<naming::id_type> bulk_create_colocated(
        naming::id_type const& id, std::size_t count, Ts&&... vs)
    {
        return bulk_create_colocated_async<Component>(
            id, count, std::forward<Ts>(vs)...)
            .get();
    }
}}    // namespace hpx::components
