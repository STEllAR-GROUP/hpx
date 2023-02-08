//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file migrate_component.hpp

#pragma once

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/actions_base/plain_action.hpp>
#include <hpx/actions_base/traits/is_distribution_policy.hpp>
#include <hpx/async_colocated/async_colocated.hpp>
#include <hpx/async_distributed/async.hpp>
#include <hpx/components/client_base.hpp>
#include <hpx/components_base/traits/is_component.hpp>
#include <hpx/distribution_policies/target_distribution_policy.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime_distributed/server/migrate_component.hpp>

#include <type_traits>

namespace hpx { namespace components {

    /// Migrate the given component to the specified target locality
    ///
    /// The function \a migrate<Component> will migrate the component
    /// referenced by \a to_migrate to the locality specified with
    /// \a target_locality. It returns a future referring to the migrated
    /// component instance.
    ///
    /// \param to_migrate      [in] The client side representation of the
    ///                        component to migrate.
    /// \param policy          [in] A distribution policy which will be used to
    ///                        determine the locality to migrate this object to.
    ///
    /// \tparam  Component     Specifies the component type of the
    ///                        component to migrate.
    /// \tparam  DistPolicy    Specifies the distribution policy to use to
    ///                        determine the destination locality.
    ///
    /// \returns A future representing the global id of the migrated
    ///          component instance. This should be the same as \a migrate_to.
    ///
    template <typename Component, typename DistPolicy>
#if defined(DOXYGEN)
    future<hpx::id_type>
#else
    inline std::enable_if_t<traits::is_component<Component>::value &&
            traits::is_distribution_policy<DistPolicy>::value,
        future<hpx::id_type>>
#endif
    migrate(hpx::id_type const& to_migrate, DistPolicy const& policy)
    {
        using action_type =
            server::perform_migrate_component_action<Component, DistPolicy>;
        return hpx::detail::async_colocated<action_type>(
            to_migrate, to_migrate, policy);
    }

    /// Migrate the given component to the specified target locality
    ///
    /// The function \a migrate<Component> will migrate the component
    /// referenced by \a to_migrate to the locality specified with
    /// \a target_locality. It returns a future referring to the migrated
    /// component instance.
    ///
    /// \param to_migrate      [in] The client side representation of the
    ///                        component to migrate.
    /// \param policy          [in] A distribution policy which will be used to
    ///                        determine the locality to migrate this object to.
    ///
    /// \tparam  Derived       Specifies the component type of the
    ///                        component to migrate.
    /// \tparam  DistPolicy    Specifies the distribution policy to use to
    ///                        determine the destination locality.
    ///
    /// \returns A future representing the global id of the migrated
    ///          component instance. This should be the same as \a migrate_to.
    ///
    template <typename Derived, typename Stub, typename Data,
        typename DistPolicy>
#if defined(DOXYGEN)
    Derived
#else
    std::enable_if_t<traits::is_distribution_policy_v<DistPolicy>, Derived>
#endif
    migrate(client_base<Derived, Stub, Data> const& to_migrate,
        DistPolicy const& policy)
    {
        using component_type =
            typename client_base<Derived, Stub, Data>::server_component_type;
        return Derived(migrate<component_type>(to_migrate.get_id(), policy));
    }

    /// \cond NODETAIL
    // overload to be used for polymorphic objects
    template <typename Component, typename Derived, typename Stub,
        typename Data, typename DistPolicy>
#if defined(DOXYGEN)
    Derived
#else
    std::enable_if_t<traits::is_distribution_policy_v<DistPolicy>, Derived>
#endif
    migrate(client_base<Derived, Stub, Data> const& to_migrate,
        DistPolicy const& policy)
    {
        return Derived(migrate<Component>(to_migrate.get_id(), policy));
    }
    /// \endcond

    /// Migrate the component with the given id to the specified target locality
    ///
    /// The function \a migrate<Component> will migrate the component
    /// referenced by \a to_migrate to the locality specified with
    /// \a target_locality. It returns a future referring to the migrated
    /// component instance.
    ///
    /// \param to_migrate      [in] The global id of the component to migrate.
    /// \param target_locality [in] The locality where the component should be
    ///                        migrated to.
    ///
    /// \tparam  Component     Specifies the component type of the
    ///          component to migrate.
    ///
    /// \returns A future representing the global id of the migrated
    ///          component instance. This should be the same as \a migrate_to.
    ///
    template <typename Component>
#if defined(DOXYGEN)
    future<hpx::id_type>
#else
    inline std::enable_if_t<traits::is_component<Component>::value,
        future<hpx::id_type>>
#endif
    migrate(hpx::id_type const& to_migrate, hpx::id_type const& target_locality)
    {
        return migrate<Component>(to_migrate, hpx::target(target_locality));
    }

    /// Migrate the given component to the specified target locality
    ///
    /// The function \a migrate<Component> will migrate the component
    /// referenced by \a to_migrate to the locality specified with
    /// \a target_locality. It returns a future referring to the migrated
    /// component instance.
    ///
    /// \param to_migrate      [in] The client side representation of the
    ///                        component to migrate.
    /// \param target_locality [in] The id of the locality to migrate
    ///                        this object to.
    ///
    /// \tparam  Derived       Specifies the component type of the
    ///                        component to migrate.
    ///
    /// \returns A client side representation of representing of the migrated
    ///          component instance. This should be the same as \a migrate_to.
    ///
    template <typename Derived, typename Stub, typename Data>
    Derived migrate(client_base<Derived, Stub, Data> const& to_migrate,
        hpx::id_type const& target_locality)
    {
        using component_type =
            typename client_base<Derived, Stub, Data>::server_component_type;
        return Derived(
            migrate<component_type>(to_migrate.get_id(), target_locality));
    }

    /// \cond NODETAIL
    // overload to be used for polymorphic objects
    template <typename Component, typename Derived, typename Stub,
        typename Data>
    Derived migrate(client_base<Derived, Stub, Data> const& to_migrate,
        hpx::id_type const& target_locality)
    {
        return Derived(
            migrate<Component>(to_migrate.get_id(), target_locality));
    }
    /// \endcond
}}    // namespace hpx::components
#endif
