//  Copyright (c) 2007-2016 Hartmut Kaiser
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
    //
    // Migrate a given component instance to the specified target locality (the
    // component instance is called 'object' below).
    //
    // Migration of an object will involve at least two localities, but can
    // touch on up to 4 localities:
    //
    // A) The locality on which the migration operation is triggered. This is
    //    the locality where `hpx::components::migrate()` is invoked.
    // B) The locality where the object to be migrated is currently located.
    // C) The locality where the object should be moved (migrated) to.
    // D) The locality which hosts the AGAS instance responsible for resolving
    //    the global address of the object to be migrated.
    //
    //    The localities B and C will be different, while the localities A and
    //    D could be the same as either of the others.
    //
    // Object migration is performed in several steps:
    //
    // 1) The migration is triggered on locality A by invoking the `migrate()`
    //    function. This invokes the action `perform_migrate_component_action`
    //    on the locality where the object to migrate is currently located
    //    (locality B).
    //
    //    The action `perform_migrate_component_action` executes the following
    //    steps:
    //    a) It will delay the start of the migration operation until no more
    //       actions (threads) are pending or currently running for the object
    //       to be migrated (the object is unpinned).
    //    b) It marks the object which is about to be migrated as 'was
    //       migrated'. This information is stored in the AGAS client side
    //       representation on locality B. It is used to forward all incoming
    //       parcels to the object's new locality. It is also used to force any
    //       locally triggered actions for the object to go through the parcel
    //       layer. For this, the object to be migrated is removed from the
    //       local AGAS cache.
    //    c) It will trigger the actual migration operation (see step 2).
    //
    //    In order to keep track of any pending and currently running actions
    //    (threads) for the object to migrate, any thread which is being
    //    scheduled will pin the object. The object will be unpinned only once
    //    the scheduled thread has executed to completion. Any last unpinning
    //    of the object will release any possibly pending migration operations
    //    (see step 1a).
    //
    // 2) The migration is triggered by invoking the action
    //    `trigger_migration_component_action` on the locality which is
    //    responsible for managing the address resolution for the object which
    //    has to be migrated (locality D).
    //
    //    The action `trigger_migration_component_action` performs 3 steps:
    //    a) Invoke `agas::begin_migration`, which marks the global id in AGAS,
    //       deferring all address resolution requests until end_migration is
    //       called. Note that the future returned by agas::begin_migration
    //       will become ready only after all currently pending operations on
    //       the target object have finished executing.
    //    b) Invoke the actual migration operation (see step 3)
    //    c) Invoke `agas::end_migration`, which un-marks the global id and
    //       releases all pending address resolution requests. Those requests
    //       now return the new object location.
    //
    // 3) The actual migration (`migrate_component_action`) is executed on the
    //    locality where the object is currently located (locality B). This
    //    involves several steps as well:
    //    a) Retrieve the (shared-) pointer to the object, this pins the
    //       object. The object is unpinned by the deleter associated with the
    //       shared pointer.
    //    b) Invoke the action `runtime_support::migrate_component` on the
    //       locality where the object has to be moved to. This passes
    //       along the shared pointer to the object and recreates the object
    //       on the target locality and updates the association of the object's
    //       global id with the new local virtual address in AGAS.
    //    c) Mark the old object (through the original shared pointer) as
    //       migrated which will delete it once the shared pointer goes out of
    //       scope.
    //
    //    The entry in the AGAS client side representation on locality B which
    //    marks the object as 'was migrated' will be left untouched (for now).
    //    This is necessary to allow for all parcels which where still resolved
    //    to the old locality will be properly forwarded to the new location
    //    of the object. Eventually this entry will have to be cleaned up no
    //    later than when the object is destroyed.
    //
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // This is step 3 of the migration process
    //
    // This will be executed on the locality where the object lives which is
    // to be migrated
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
        template <typename Component, typename DistPolicy>
        future<naming::id_type> migrate_component_postproc(
            boost::shared_ptr<Component> const& ptr,
            naming::id_type const& to_migrate, DistPolicy const& policy)
        {
            using components::stubs::runtime_support;

            boost::uint32_t pin_count = ptr->pin_count();

            if (pin_count == ~0x0u)
            {
                return make_exceptional_future<hpx::id_type>(
                    HPX_GET_EXCEPTION(invalid_status,
                        "hpx::components::server::migrate_component",
                        "attempting to migrate an instance of a component "
                        "which was already migrated"));
            }

            if (pin_count > 1)
            {
                return make_exceptional_future<hpx::id_type>(
                    HPX_GET_EXCEPTION(invalid_status,
                        "hpx::components::server::migrate_component",
                        "attempting to migrate an instance of a component "
                        "which is currently pinned"));
            }

            return runtime_support::migrate_component_async<Component>(
                        policy, ptr, to_migrate)
                .then(util::bind(
                    &detail::migrate_component_cleanup<Component>,
                    util::placeholders::_1, ptr, to_migrate));
        }
    }

    template <typename Component, typename DistPolicy>
    future<naming::id_type> migrate_component(
        naming::id_type const& to_migrate, naming::address const& addr,
        DistPolicy const& policy)
    {
        // 'migration' to same locality as before is a no-op
        if (policy.get_next_target() == hpx::find_here())
        {
            return make_ready_future(to_migrate);
        }

        if (!Component::supports_migration())
        {
            return make_exceptional_future<hpx::id_type>(
                HPX_GET_EXCEPTION(invalid_status,
                    "hpx::components::server::migrate_component",
                    "attempting to migrate an instance of a component which "
                    "does not support migration"));
        }

        // retrieve pointer to object (must be local)
        boost::shared_ptr<Component> ptr =
            hpx::detail::get_ptr_for_migration<Component>(addr, to_migrate);

        // perform actual migration by sending data over to target locality
        return detail::migrate_component_postproc<Component>(
            ptr, to_migrate, policy);
    }

    template <typename Component, typename DistPolicy>
    struct migrate_component_action
      : ::hpx::actions::action<
            future<naming::id_type> (*)(naming::id_type const&,
                naming::address const&, DistPolicy const&)
          , &migrate_component<Component, DistPolicy>
          , migrate_component_action<Component, DistPolicy> >
    {};

    ///////////////////////////////////////////////////////////////////////////
    // This is step 2 of the migration process
    //
    // This is executed on the locality responsible for managing the address
    // resolution for the given object.
    template <typename Component, typename DistPolicy>
    future<naming::id_type> trigger_migrate_component(
        naming::id_type const& to_migrate, DistPolicy const& policy)
    {
        if (!Component::supports_migration())
        {
            return make_exceptional_future<naming::id_type>(
                HPX_GET_EXCEPTION(invalid_status,
                    "hpx::components::server::trigger_migrate_component",
                    "attempting to migrate an instance of a component which "
                    "does not support migration"));
        }

        if (naming::get_locality_id_from_id(to_migrate) != get_locality_id())
        {
            return make_exceptional_future<naming::id_type>(
                HPX_GET_EXCEPTION(invalid_status,
                    "hpx::components::server::trigger_migrate_component",
                    "this function has to be executed on the locality "
                    "responsible for managing the address of the given object"));
        }

        return agas::begin_migration(to_migrate)
            .then(
                [=](future<std::pair<naming::id_type, naming::address> > && f)
                    ->  future<naming::id_type>
                {
                    // rethrow errors
                    std::pair<naming::id_type, naming::address> r = f.get();

                    // perform actual object migration
                    typedef migrate_component_action<
                            Component, DistPolicy
                        > action_type;
                    return async<action_type>(r.first, to_migrate, r.second,
                        policy);
                })
            .then(
                [to_migrate](future<naming::id_type> && f) -> naming::id_type
                {
                    agas::end_migration(to_migrate).get();
                    return f.get();
                });
    }

    template <typename Component, typename DistPolicy>
    struct trigger_migrate_component_action
      : ::hpx::actions::action<
            future<naming::id_type> (*)(naming::id_type const&, DistPolicy const&)
          , &trigger_migrate_component<Component, DistPolicy>
          , trigger_migrate_component_action<Component, DistPolicy> >
    {};

    ///////////////////////////////////////////////////////////////////////////
    // This is step 1 of the migration process
    //
    // This is executed on the locality where the object to migrate is
    // currently located.
    template <typename Component, typename DistPolicy>
    future<naming::id_type> perform_migrate_component(
        naming::id_type const& to_migrate, DistPolicy const& policy)
    {
        if (!Component::supports_migration())
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "hpx::components::server::perform_migrate_component",
                "attempting to migrate an instance of a component which "
                "does not support migration");
            return make_ready_future(naming::invalid_id);
        }

        // retrieve pointer to object (must be local)
        return hpx::get_ptr<Component>(to_migrate)
            .then(
                [=](hpx::future<boost::shared_ptr<Component> > && f)
                    ->  hpx::future<hpx::id_type>
                {
                    hpx::future<void> trigger_migration;

                    {
                        boost::shared_ptr<Component> ptr = f.get();

                        // Delay the start of the migration operation until no
                        // more actions (threads) are pending or currently
                        // running for the given object (until the object is
                        // unpinned).
                        trigger_migration = ptr->mark_as_migrated(to_migrate);

                        // Unpin the object, will trigger migration if this is
                        // the only pin-count.
                    }

                    // Once the migration is possible (object is not pinned
                    // anymore trigger the necessary actions)
                    return trigger_migration
                        .then(
                            launch::async,  // run on separate thread
                            [=](hpx::future<void> && f)
                                ->  hpx::future<hpx::id_type>
                            {
                                f.get();        // rethrow exceptions

                                // now trigger 2nd step of migration
                                typedef trigger_migrate_component_action<
                                        Component, DistPolicy
                                    > action_type;

                                return async<action_type>(
                                    naming::get_locality_from_id(to_migrate),
                                    to_migrate, policy);
                            });
                });
    }

    template <typename Component, typename DistPolicy>
    struct perform_migrate_component_action
      : ::hpx::actions::action<
            future<naming::id_type> (*)(naming::id_type const&, DistPolicy const&)
          , &perform_migrate_component<Component, DistPolicy>
          , perform_migrate_component_action<Component, DistPolicy> >
    {};
}}}

#endif

