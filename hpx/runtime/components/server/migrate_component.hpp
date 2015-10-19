//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_COMPONENTS_SERVER_MIGRATE_COMPONENT_JAN_30_2014_0737AM)
#define HPX_RUNTIME_COMPONENTS_SERVER_MIGRATE_COMPONENT_JAN_30_2014_0737AM

#include <hpx/config.hpp>
#include <hpx/traits/is_component.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/get_ptr.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/agas/interface.hpp>

namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    // Migrate given component to the specified target locality
    //
    // Object migration is performed in several steps:
    //
    // 1) The migration is triggered by invoking the trigger_migration_action
    //    on the locality which is responsible for managing the address
    //    resolution for the object which has to be migrated.
    // 2) The trigger_migration_action performs 3 steps:
    //    a) Invoke agas::begin_migration, which marks the global id in AGAS,
    //       deferring all address resolution requests until end_migration is
    //       called.
    //    b) Invoke the actual migration operation (see step 3)
    //    c) Invoke end_migration, which un-marks the global id and releases
    //       all pending address resolution requests. Those requests now return
    //       the new object location.
    // 3) The actual migration (migrate_component_action) is executed on the
    //    locality where the object is currently located. This involves several
    //    steps as well:
    //    a) Retrieve the (shared-) pointer to the object, this pins the
    //       object. The object is unpinned by the deleter associated with the
    //       shared pointer.
    //    b) Invoke the action runtime_support::migrate_component on the
    //       locality where the object has to be moved to. This passes
    //       along the shared pointer to the object and recreates the object
    //       on the target locality and updates the association of the object's
    //       global id with the new local virtual address in AGAS.
    //    c) Mark the old object (through the original shared pointer) as
    //       migrated which will delete it once the shared pointer goes out of
    //       scope.
    //
    namespace detail
    {
        // clean up (source) memory of migrated object
        template <typename Component>
        naming::id_type migrate_component_cleanup(
            future<naming::id_type> && f,
            boost::shared_ptr<Component> ptr,
            naming::id_type const& to_migrate)
        {
            ptr->mark_as_migrated();
            return f.get();
        }

        // trigger the actual migration
        template <typename Component>
        future<naming::id_type> migrate_component_postproc(
            boost::shared_ptr<Component> const& ptr,
            naming::id_type const& to_migrate,
            naming::id_type const& target_locality)
        {
            using components::stubs::runtime_support;

            boost::uint32_t pin_count = ptr->pin_count();

            if (pin_count == ~0x0u)
            {
                HPX_THROW_EXCEPTION(invalid_status,
                    "hpx::components::server::migrate_component",
                    "attempting to migrate an instance of a component which was "
                    "already migrated");
                return make_ready_future(naming::invalid_id);
            }

            if (pin_count > 1)
            {
                HPX_THROW_EXCEPTION(invalid_status,
                    "hpx::components::server::migrate_component",
                    "attempting to migrate an instance of a component which is "
                    "currently pinned");
                return make_ready_future(naming::invalid_id);
            }

            return runtime_support::migrate_component_async<Component>(
                        target_locality, ptr, to_migrate)
                .then(util::bind(
                    &detail::migrate_component_cleanup<Component>,
                    util::placeholders::_1, ptr, to_migrate));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // This will be executed on the locality where the object lives which is
    // to be migrated
    template <typename Component>
    future<naming::id_type> migrate_component(
        naming::id_type const& to_migrate,
        naming::address const& addr,
        naming::id_type const& target_locality)
    {
        // 'migration' to same locality as before is a no-op
        if (target_locality == hpx::find_here())
        {
            return make_ready_future(to_migrate);
        }

        if (!Component::supports_migration())
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "hpx::components::server::migrate_component",
                "attempting to migrate an instance of a component which "
                "does not support migration");
            return make_ready_future(naming::invalid_id);
        }

        // retrieve pointer to object (must be local)
        boost::shared_ptr<Component> ptr =
            hpx::detail::get_ptr_for_migration<Component>(addr, to_migrate);

        // perform actual migration by sending data over to target locality
        return detail::migrate_component_postproc<Component>(
            ptr, to_migrate, target_locality);
    }

    template <typename Component>
    struct migrate_component_action
      : ::hpx::actions::action<
            future<naming::id_type> (*)(naming::id_type const&,
                naming::address const&, naming::id_type const&)
          , &migrate_component<Component>
          , migrate_component_action<Component> >
    {};

    ///////////////////////////////////////////////////////////////////////////
    // This is executed on the locality responsible for managing the address
    // resolution for the given object.
    template <typename Component>
    future<naming::id_type> trigger_migrate_component(
        naming::id_type const& to_migrate,
        naming::id_type const& target_locality)
    {
        if (!Component::supports_migration())
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "hpx::components::server::trigger_migrate_component",
                "attempting to migrate an instance of a component which "
                "does not support migration");
            return make_ready_future(naming::invalid_id);
        }

        if (naming::get_locality_id_from_id(to_migrate) != get_locality_id())
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "hpx::components::server::trigger_migrate_component",
                "this function has to be executed on the locality responsible "
                "for managing the address of the given object");
            return make_ready_future(naming::invalid_id);
        }

        return agas::begin_migration(to_migrate, target_locality)
            .then(
                [to_migrate, target_locality](
                    future<std::pair<naming::id_type, naming::address> > && f)
                        -> future<naming::id_type>
                {
                    // rethrow errors
                    std::pair<naming::id_type, naming::address> r = f.get();

                    // perform actual object migration
                    typedef server::migrate_component_action<Component>
                        action_type;
                    return async<action_type>(r.first, to_migrate, r.second,
                        target_locality);
                })
            .then(
                [to_migrate](future<naming::id_type> && f) -> naming::id_type
                {
                    agas::end_migration(to_migrate).get();
                    return f.get();
                });
    }

    template <typename Component>
    struct trigger_migrate_component_action
      : ::hpx::actions::action<
            future<naming::id_type> (*)(naming::id_type const&,
                naming::id_type const&)
          , &trigger_migrate_component<Component>
          , trigger_migrate_component_action<Component> >
    {};
}}}

#endif

