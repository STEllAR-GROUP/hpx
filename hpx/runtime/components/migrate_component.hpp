//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file migrate_component.hpp

#if !defined(HPX_RUNTIME_COMPONENTS_COPY_MIGRATE_COMPONENT_JAN_31_2014_1009AM)
#define HPX_RUNTIME_COMPONENTS_COPY_MIGRATE_COMPONENT_JAN_31_2014_1009AM

#include <hpx/config.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/lcos/detail/async_colocated.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/server/migrate_component.hpp>
#include <hpx/runtime/components/targeting_distribution_policy.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/traits/is_component.hpp>
#include <hpx/traits/is_distribution_policy.hpp>

#include <boost/utility/enable_if.hpp>

namespace hpx { namespace components
{
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
    future<naming::id_type>
#else
    inline typename boost::enable_if_c<
        traits::is_component<Component>::value &&
            traits::is_distribution_policy<DistPolicy>::value,
        future<naming::id_type>
    >::type
#endif
    migrate(naming::id_type const& to_migrate, DistPolicy const& policy)
    {
        typedef server::perform_migrate_component_action<
                Component, DistPolicy
            > action_type;
        return hpx::detail::async_colocated<action_type>(to_migrate,
            to_migrate, policy);
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
    template <typename Derived, typename Stub, typename DistPolicy>
#if defined(DOXYGEN)
    Derived
#else
    inline typename boost::enable_if_c<
        traits::is_distribution_policy<DistPolicy>::value, Derived
    >::type
#endif
    migrate(client_base<Derived, Stub> const& to_migrate,
        DistPolicy const& policy)
    {
        typedef typename client_base<Derived, Stub>::server_component_type
            component_type;
        return Derived(migrate<component_type>(to_migrate.get_id(), policy));
    }

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
    future<naming::id_type>
#else
    inline typename boost::enable_if<
        traits::is_component<Component>, future<naming::id_type>
    >::type
#endif
    migrate(naming::id_type const& to_migrate,
        naming::id_type const& target_locality)
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
    template <typename Derived, typename Stub>
    inline Derived
    migrate(client_base<Derived, Stub> const& to_migrate,
        naming::id_type const& target_locality)
    {
        typedef typename client_base<Derived, Stub>::server_component_type
            component_type;
        return Derived(migrate<component_type>(to_migrate.get_id(),
            target_locality));
    }
}}

#endif
