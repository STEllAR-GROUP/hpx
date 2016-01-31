//  Copyright (c) 2015-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_MIGRATE_FROM_STORAGE_SERVER_FEB_09_2015_0330PM)
#define HPX_MIGRATE_FROM_STORAGE_SERVER_FEB_09_2015_0330PM

#include <hpx/include/async.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/naming.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/include/util.hpp>

#include <hpx/components/component_storage/export_definitions.hpp>
#include <hpx/components/component_storage/server/component_storage.hpp>

namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    //
    // Migrate given component from the specified storage component
    //
    // Migration of an object to storage will involve at least one locality,
    // but can touch on up to 4 localities:
    //
    // A) The locality on which the migration operation is triggered. This is
    //    the locality where `hpx::components::migrate_from_storage()` is invoked.
    // b) The locality where the storage object lives the object should be
    //    moved (migrated) from.
    // C) The locality where the object should be migrated to.
    // D) The locality which hosts the AGAS instance responsible for resolving
    //    the global address of the object to be migrated.
    //
    // Object migration is performed from the storage in several steps:
    //
    // 1) The migration is triggered by invoking the
    //    `trigger_migrate_from_storage_action` on the locality which is
    //    responsible for managing the address resolution for the object which
    //    has to be migrated.
    //
    //   The `trigger_migrate_from_storage_action` performs 3 steps:
    //    a) Invoke agas::begin_migration, which marks the global id in AGAS,
    //       deferring all address resolution requests until end_migration is
    //       called.
    //    b) Invoke the actual migration operation (see step 3)
    //    c) Invoke end_migration, which un-marks the global id and releases
    //       all pending address resolution requests. Those requests now return
    //       the new object location.
    //
    // 2) The actual migration (`component_storage::migrate_from_here_action`)
    //    is executed on the storage facility where the object is currently
    //    stored (locality B). This involves several steps as well:
    //    a) Retrieve the byte stream representing the object from the storage
    //    b) Deserialize the byte stream to re-create the object. The newly
    //       recreated object is pinned immediately. The object is unpinned by
    //       the deleter associated with the shared pointer.
    //    c) Invoke the action `runtime_support::migrate_component` on the
    //       locality where the object has to be moved to (locality C). This
    //       passes along the shared pointer to the object and recreates the
    //       object on the target locality and updates the association of the
    //       object's global id with the new local virtual address in AGAS.
    //    d) Mark the old object (through the original shared pointer) as
    //       migrated which will delete it once the shared pointer goes out of
    //       scope.
    //
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // This is step 2 of the migration process
    //
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Component>
        future<id_type> migrate_from_storage_id(
            id_type const& target_locality,
            boost::shared_ptr<Component> const& ptr,
            id_type const& to_resurrect)
        {
            // and resurrect it on the specified locality
            using hpx::components::runtime_support;
            return runtime_support::migrate_component_async<Component>(
                        target_locality, ptr, to_resurrect)
                .then(util::bind(
                    &detail::migrate_component_cleanup<Component>,
                    util::placeholders::_1, ptr, to_resurrect));
        }

        template <typename Component>
        future<id_type> migrate_from_storage_addr(
            naming::address const& addr,
            boost::shared_ptr<Component> const& ptr,
            id_type const& to_resurrect)
        {
            id_type id(addr.locality_, id_type::unmanaged);
            return migrate_from_storage_id(id, ptr, to_resurrect);
        }

        // convert the extracted data into a living component instance
        template <typename Component>
        future<id_type> migrate_from_storage(
            future<std::vector<char> > && f,
            id_type const& to_resurrect, naming::address const& addr,
            id_type const& target_locality)
        {
            // recreate the object
            boost::shared_ptr<Component> ptr;

            {
                std::vector<char> data = f.get();
                serialization::input_archive archive(data, data.size(), 0);
                archive >> ptr;
            }

            // make sure the migration code works properly
            ptr->pin();

            // if target locality is not specified, use the address of the last
            // locality where the object was living before
            if (target_locality == naming::invalid_id)
            {
                return migrate_from_storage_addr(addr, ptr, to_resurrect);
            }

            // otherwise directly refer to the locality where the object should
            // be resurrected
            return migrate_from_storage_id(target_locality, ptr, to_resurrect);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // This is step 1 of the migration process
    //
    // This is executed on the locality responsible for managing the address
    // resolution for the given object.
    template <typename Component>
    future<id_type> trigger_migrate_from_storage(
        id_type const& to_resurrect,
        id_type const& target_locality)
    {
        if (!Component::supports_migration())
        {
            return make_exceptional_future<id_type>(
                HPX_GET_EXCEPTION(invalid_status,
                    "hpx::components::server::trigger_migrate_from_storage",
                    "attempting to migrate an instance of a component which "
                    "does not support migration"));
        }

        if (naming::get_locality_id_from_id(to_resurrect) != get_locality_id())
        {
            return make_exceptional_future<id_type>(
                HPX_GET_EXCEPTION(invalid_status,
                    "hpx::components::server::trigger_migrate_from_storage",
                    "this function has to be executed on the locality responsible "
                    "for managing the address of the given object"));
        }

        return agas::begin_migration(to_resurrect)
            .then(
                [to_resurrect, target_locality](
                    future<std::pair<id_type, naming::address> > && f)
                        -> future<id_type>
                {
                    // rethrow errors
                    std::pair<id_type, naming::address> r = f.get();

                    // retrieve the data from the given storage
                    typedef
                        typename server::component_storage::migrate_from_here_action
                        action_type;
                    return async<action_type>(r.first, to_resurrect.get_gid())
                        .then(util::bind(
                            &detail::migrate_from_storage<Component>,
                            util::placeholders::_1, to_resurrect,
                            r.second, target_locality));
                })
            .then(
                [to_resurrect](future<id_type> && f) -> id_type
                {
                    agas::end_migration(to_resurrect).get();
                    return f.get();
                });
    }

    template <typename Component>
    struct trigger_migrate_from_storage_action
      : ::hpx::actions::action<
            future<id_type> (*)(id_type const&,
                id_type const&)
          , &trigger_migrate_from_storage<Component>
          , trigger_migrate_from_storage_action<Component> >
    {};
}}}

#endif


