//  Copyright (c) 2020-2024 Hartmut Kaiser
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
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::components {

    namespace server {

        template <typename Component, typename... Ts>
        struct create_component_action;

        template <bool WithCount, typename Component, typename... Ts>
        struct bulk_create_component_action;
    }    // namespace server

    ///////////////////////////////////////////////////////////////////////////
    /// Asynchronously create a new instance of a component
    template <typename Component, typename... Ts>
    future<hpx::id_type> create_async(hpx::id_type const& gid, Ts&&... vs)
    {
        if (!naming::is_locality(gid))
        {
            HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                "stubs::runtime_support::create_component_async",
                "The id passed as the first argument is not representing "
                "a locality");
        }

        using action_type =
            server::create_component_action<Component, std::decay_t<Ts>...>;

        return hpx::async<action_type>(gid, HPX_FORWARD(Ts, vs)...);
    }

    template <bool WithCount, typename Component, typename... Ts>
    future<std::vector<hpx::id_type>> bulk_create_async(
        hpx::id_type const& gid, std::size_t count, Ts&&... vs)
    {
        if (!naming::is_locality(gid))
        {
            HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                "stubs::runtime_support::bulk_create_component_async",
                "The id passed as the first argument is not representing "
                "a locality");
        }

        using action_type = server::bulk_create_component_action<WithCount,
            Component, std::decay_t<Ts>...>;

        return hpx::async<action_type>(gid, count, HPX_FORWARD(Ts, vs)...);
    }

    template <typename Component, typename... Ts>
    hpx::id_type create(hpx::id_type const& gid, Ts&&... vs)
    {
        return create_async<Component>(gid, HPX_FORWARD(Ts, vs)...).get();
    }

    template <bool WithCount, typename Component, typename... Ts>
    std::vector<hpx::id_type> bulk_create(
        hpx::id_type const& gid, std::size_t count, Ts&&... vs)
    {
        return bulk_create_async<WithCount, Component>(
            gid, count, HPX_FORWARD(Ts, vs)...)
            .get();
    }

    template <typename Component, typename... Ts>
    future<hpx::id_type> create_colocated_async(
        hpx::id_type const& gid, Ts&&... vs)
    {
        using action_type =
            server::create_component_action<Component, std::decay_t<Ts>...>;

        return hpx::detail::async_colocated<action_type>(
            gid, HPX_FORWARD(Ts, vs)...);
    }

    template <typename Component, typename... Ts>
    static hpx::id_type create_colocated(hpx::id_type const& gid, Ts&&... vs)
    {
        return create_colocated_async(gid, HPX_FORWARD(Ts, vs)...).get();
    }

    template <bool WithCount, typename Component, typename... Ts>
    static future<std::vector<hpx::id_type>> bulk_create_colocated_async(
        hpx::id_type const& gid, std::size_t count, Ts&&... vs)
    {
        using action_type = server::bulk_create_component_action<WithCount,
            Component, std::decay_t<Ts>...>;

        return hpx::detail::async_colocated<action_type>(
            gid, count, HPX_FORWARD(Ts, vs)...);
    }

    template <bool WithCount, typename Component, typename... Ts>
    std::vector<hpx::id_type> bulk_create_colocated(
        hpx::id_type const& id, std::size_t count, Ts&&... vs)
    {
        return bulk_create_colocated_async<WithCount, Component>(
            id, count, HPX_FORWARD(Ts, vs)...)
            .get();
    }
}    // namespace hpx::components
