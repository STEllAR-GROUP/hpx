//  Copyright (c) 2015-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_MIGRATE_TO_STORAGE_SERVER_FEB_04_2015_1021AM)
#define HPX_MIGRATE_TO_STORAGE_SERVER_FEB_04_2015_1021AM

#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/naming.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/include/util.hpp>

#include <hpx/components/component_storage/export_definitions.hpp>
#include <hpx/components/component_storage/server/component_storage.hpp>

namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    //
    // Migrate given component to the component storage
    //
    // Migration of an object to storage will involve at least one locality,
    // but can touch on up to 4 localities:
    //
    // A) The locality on which the migration operation is triggered. This is
    //    the locality where `hpx::components::migrate_to_storage()` is invoked.
    // B) The locality where the object to be migrated is currently located.
    // C) The locality where the storage object lives the object should be
    //    moved (migrated) to.
    // D) The locality which hosts the AGAS instance responsible for resolving
    //    the global address of the object to be migrated.
    //
    // Object migration is performed in several steps:
    //
    // 1) The migration is triggered on locality A by invoking the
    //    `migrate_to_storage()` function. This invokes the action
    //    `perform_migrate_to_storage_action` on the locality where the object
    //    to migrate is currently located (locality B).
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
    // 2) The migration is triggered by invoking the
    //    `trigger_migrate_to_storage_action` on the locality which is
    //    responsible for managing the address resolution for the object which
    //    has to be migrated (locality D).
    //
    //    The `trigger_migrate_to_storage_action` performs 3 steps:
    //    a) Invoke `agas::begin_migration`, which marks the global id in AGAS,
    //       deferring all address resolution requests until end_migration is
    //       called.
    //    b) Invoke the actual migration operation (see step 3)
    //    c) Invoke `agas::end_migration`, which un-marks the global id and
    //       releases all pending address resolution requests. Those requests
    //       now return the new object location (dormant in storage).
    //
    // 3) The actual migration (`migrate_to_storage_action`) is executed on
    //    the locality where the object is currently located (locality B). This
    //    involves several steps as well:
    //
    //    a) Retrieve the (shared-) pointer to the object, this pins the
    //       object. The object is unpinned by the deleter associated with the
    //       shared pointer.
    //    b) Serialize the object to create a byte stream representing the
    //       state of the object.
    //    c) Invoke the action `component_storage::migrate_to_here_action` on
    //       the storage object which should receive the data. This passes
    //       along the serialized data and updates the association of the
    //       object's global id with the new local virtual address in AGAS.
    //    d) Mark the old object (through the original shared pointer) as
    //       migrated which will delete it once the shared pointer goes out of
    //       scope.
    //
    //    The entry in the AGAS client side representation on locality B which
    //    marks the object as 'was migrated' will be left untouched (for now).
    //    This is necessary to allow for all parcels which where still resolved
    //    to the old locality will be properly forwarded to the new location
    //    of the object. Eventually this entry will have to be cleaned up no
    //    later than when the object is eventually destroyed.
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
        id_type migrate_to_storage_cleanup(
            future<id_type> f,
            boost::shared_ptr<Component> ptr,
            id_type const& to_migrate)
        {
            ptr->mark_as_migrated();
            return f.get();
        }

        // trigger the actual migration to storage
        template <typename Component>
        future<id_type> migrate_to_storage_postproc(
            boost::shared_ptr<Component> const& ptr,
            id_type const& to_migrate,
            id_type const& target_storage)
        {
            boost::uint32_t pin_count = ptr->pin_count();

            if (pin_count == ~0x0u)
            {
                return hpx::make_exceptional_future<id_type>(
                    HPX_GET_EXCEPTION(invalid_status,
                        "hpx::components::server::migrate_to_storage_postproc",
                        "attempting to migrate an instance of a component "
                        "which was already migrated"));
            }

            if (pin_count > 1)
            {
                return hpx::make_exceptional_future<id_type>(
                    HPX_GET_EXCEPTION(invalid_status,
                        "hpx::components::server::migrate_to_storage_postproc",
                        "attempting to migrate an instance of a component "
                        "which is currently pinned"));
            }

            // serialize the given component
            std::vector<char> data;

            {
                serialization::output_archive archive(data);
                archive << ptr;
            }

            naming::address addr(ptr->get_current_address());

            typedef typename server::component_storage::migrate_to_here_action
                action_type;

            return hpx::async<action_type>(
                    target_storage, std::move(data), to_migrate, addr)
                .then(util::bind(
                    &migrate_to_storage_cleanup<Component>,
                    util::placeholders::_1, ptr, to_migrate));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // This will be executed on the locality where the object lives which is
    // to be migrated
    template <typename Component>
    future<id_type> migrate_to_storage_here(
        id_type const& to_migrate,
        naming::address const& addr,
        id_type const& target_storage)
    {
        if (!Component::supports_migration())
        {
            return hpx::make_exceptional_future<hpx::id_type>(
                HPX_GET_EXCEPTION(invalid_status,
                    "hpx::components::server::migrate_component",
                    "attempting to migrate an instance of a component which "
                    "does not support migration"));
        }

        // retrieve pointer to object (must be local)
        boost::shared_ptr<Component> ptr =
            hpx::detail::get_ptr_for_migration<Component>(addr, to_migrate);

        // perform actual migration by sending data over to target locality
        return detail::migrate_to_storage_postproc<Component>(
            ptr, to_migrate, target_storage);
    }

    template <typename Component>
    struct migrate_to_storage_action
      : ::hpx::actions::action<
            future<id_type> (*)(id_type const&,
                naming::address const&, id_type const&)
          , &migrate_to_storage_here<Component>
          , migrate_to_storage_action<Component> >
    {};

    ///////////////////////////////////////////////////////////////////////////
    // This is step 2 of the migration process
    //
    // This is executed on the locality responsible for managing the address
    // resolution for the given object.
    template <typename Component>
    future<id_type> trigger_migrate_to_storage(
        id_type const& to_migrate,
        id_type const& target_storage)
    {
        if (!Component::supports_migration())
        {
            return hpx::make_exceptional_future<id_type>(
                HPX_GET_EXCEPTION(invalid_status,
                    "hpx::components::server::trigger_migrate_to_storage",
                    "attempting to migrate an instance of a component which "
                    "does not support migration"));
        }

        if (naming::get_locality_id_from_id(to_migrate) != get_locality_id())
        {
            return hpx::make_exceptional_future<id_type>(
                HPX_GET_EXCEPTION(invalid_status,
                    "hpx::components::server::trigger_migrate_to_storage",
                    "this function has to be executed on the locality "
                    "responsible for managing the address of the given object"));
        }

        return agas::begin_migration(to_migrate)
            .then(
                [to_migrate, target_storage](
                    future<std::pair<id_type, naming::address> > && f)
                        -> future<id_type>
                {
                    // rethrow errors
                    std::pair<id_type, naming::address> r = f.get();

                    // perform actual object migration
                    typedef server::migrate_to_storage_action<Component>
                        action_type;
                    return async<action_type>(r.first, to_migrate, r.second,
                        target_storage);
                })
            .then(
                [to_migrate](future<id_type> && f) -> id_type
                {
                    agas::end_migration(to_migrate).get();
                    return f.get();
                });
    }

    template <typename Component>
    struct trigger_migrate_to_storage_action
      : ::hpx::actions::action<
            future<id_type> (*)(id_type const&, id_type const&)
          , &trigger_migrate_to_storage<Component>
          , trigger_migrate_to_storage_action<Component> >
    {};

    ///////////////////////////////////////////////////////////////////////////
    // This is step 1 of the migration process
    //
    // This is executed on the locality where the object to migrate is
    // currently located.
    template <typename Component>
    future<id_type> perform_migrate_to_storage(
        id_type const& to_migrate, id_type const& target_storage)
    {
        if (!Component::supports_migration())
        {
            return hpx::make_exceptional_future<id_type>(
                HPX_GET_EXCEPTION(invalid_status,
                    "hpx::components::server::perform_migrate_to_storage",
                    "attempting to migrate an instance of a component which "
                    "does not support migration"));
        }

        // retrieve pointer to object (must be local)
        return hpx::get_ptr<Component>(to_migrate)
            .then(
                [=](future<boost::shared_ptr<Component> > && f) -> future<id_type>
                {
                    future<void> trigger_migration;

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
                            [=](future<void> && f) -> future<id_type>
                            {
                                f.get();        // rethrow exceptions

                                // now trigger 2nd step of migration
                                typedef trigger_migrate_to_storage_action<
                                        Component
                                    > action_type;

                                return async<action_type>(
                                    naming::get_locality_from_id(to_migrate),
                                    to_migrate, target_storage);
                            });
                });
    }

    template <typename Component>
    struct perform_migrate_to_storage_action
      : ::hpx::actions::action<
            future<id_type> (*)(id_type const&, id_type const&)
          , &perform_migrate_to_storage<Component>
          , perform_migrate_to_storage_action<Component> >
    {};
}}}

#endif


