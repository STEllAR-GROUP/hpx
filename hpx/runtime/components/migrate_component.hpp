//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file migrate_component.hpp

#if !defined(HPX_RUNTIME_COMPONENTS_COPY_MIGRATE_COMPONENT_JAN_31_2014_1009AM)
#define HPX_RUNTIME_COMPONENTS_COPY_MIGRATE_COMPONENT_JAN_31_2014_1009AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/server/migrate_component.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/lcos/detail/async_colocated.hpp>
#include <hpx/traits/is_component.hpp>

#include <boost/utility/enable_if.hpp>

namespace hpx { namespace components
{
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
    /// \tparam  The only template argument specifies the component type of the
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
        typedef server::perform_migrate_component_action<Component> action_type;
        return hpx::detail::async_colocated<action_type>(to_migrate,
            to_migrate, target_locality);
    }

    /// Migrate the given component to the specified target locality
    ///
    /// The function \a migrate_to_storage will migrate the component
    /// referenced by \a to_migrate to the storage facility specified with
    /// \a target_storage. It returns a future referring to the migrated
    /// component instance.
    ///
    /// \param to_migrate      [in] The client side representation of the
    ///                        component to migrate.
    /// \param target_storage  [in] The id of the storage facility to migrate
    ///                        this object to.
    ///
    /// \returns A client side representation of representing of the migrated
    ///          component instance. This should be the same as \a migrate_to.
    ///
    template <typename Derived, typename Stub>
    inline Derived
    migrate(client_base<Derived, Stub> const& to_migrate,
        naming::id_type const& target_storage)
    {
        typedef typename client_base<Derived, Stub>::server_component_type
            component_type;
        return Derived(migrate<component_type>(to_migrate.get_id(),
            target_storage));
    }
}}

#endif
