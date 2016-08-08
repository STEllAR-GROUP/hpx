//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_MIGRATE_TO_STORAGE_SERVER_FEB_04_2015_1021AM)
#define HPX_MIGRATE_TO_STORAGE_SERVER_FEB_04_2015_1021AM

#include <hpx/config.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/bind.hpp>

#include <hpx/components/component_storage/export_definitions.hpp>
#include <hpx/components/component_storage/server/component_storage.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    // Migrate given component to the component storage
    //
    // Object migration is performed in several steps:
    //
    // 1) The migration is triggered by invoking the
    //    trigger_migrate_to_storage_here_action
    //    on the locality which is responsible for managing the address
    //    resolution for the object which has to be migrated.
    // 2) The trigger_migrate_to_storage_here_action performs 3 steps:
    //    a) Invoke agas::begin_migration, which marks the global id in AGAS,
    //       deferring all address resolution requests until end_migration is
    //       called.
    //    b) Invoke the actual migration operation (see step 3)
    //    c) Invoke end_migration, which un-marks the global id and releases
    //       all pending address resolution requests. Those requests now return
    //       the new object location.
    // 3) The actual migration (migrate_to_storage_here_action) is executed on
    //    the locality where the object is currently located. This involves
    //    several steps as well:
    //    a) Retrieve the (shared-) pointer to the object, this pins the
    //       object. The object is unpinned by the deleter associated with the
    //       shared pointer.
    //    b) Serialize the object to create a byte stream representing the
    //       state of the object.
    //    c) Invoke the action component_storage::migrate_to_here_action on the
    //       storage object which should receive the data. This passes
    //       along the serialized data and updates the association of the
    //       object's global id with the new local virtual address in AGAS.
    //    d) Mark the old object (through the original shared pointer) as
    //       migrated which will delete it once the shared pointer goes out of
    //       scope.
    //
    namespace detail
    {
        // clean up (source) memory of migrated object
        template <typename Component>
        naming::id_type migrate_to_storage_here_cleanup(
            future<naming::id_type> f,
            std::shared_ptr<Component> ptr,
            naming::id_type const& to_migrate)
        {
            ptr->mark_as_migrated();
            return f.get();
        }

        // trigger the actual migration to storage
        template <typename Component>
        future<naming::id_type> migrate_to_storage_here_postproc(
            std::shared_ptr<Component> const& ptr,
            naming::id_type const& to_migrate,
            naming::id_type const& target_storage)
        {
            using components::stubs::runtime_support;

            boost::uint32_t pin_count = ptr->pin_count();

            if (pin_count == ~0x0u)
            {
                HPX_THROW_EXCEPTION(invalid_status,
                    "hpx::components::server::migrate_to_storage_here",
                    "attempting to migrate an instance of a component which was "
                    "already migrated");
                return make_ready_future(naming::invalid_id);
            }

            if (pin_count > 1)
            {
                HPX_THROW_EXCEPTION(invalid_status,
                    "hpx::components::server::migrate_to_storage_here",
                    "attempting to migrate an instance of a component which is "
                    "currently pinned");
                return make_ready_future(naming::invalid_id);
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
                    &migrate_to_storage_here_cleanup<Component>,
                    util::placeholders::_1, ptr, to_migrate));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // This will be executed on the locality where the object lives which is
    // to be migrated
    template <typename Component>
    future<naming::id_type> migrate_to_storage_here(
        naming::id_type const& to_migrate,
        naming::address const& addr,
        naming::id_type const& target_storage)
    {
        if (!traits::component_supports_migration<Component>::call())
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "hpx::components::server::migrate_to_storage_here",
                "attempting to migrate an instance of a component which "
                "does not support migration");
            return make_ready_future(naming::invalid_id);
        }

        // retrieve pointer to object (must be local)
        std::shared_ptr<Component> ptr =
            hpx::detail::get_ptr_for_migration<Component>(addr, to_migrate);

        // perform actual migration by sending data over to target locality
        return detail::migrate_to_storage_here_postproc<Component>(
            ptr, to_migrate, target_storage);
    }

    template <typename Component>
    struct migrate_to_storage_here_action
      : ::hpx::actions::action<
            future<naming::id_type> (*)(naming::id_type const&,
                naming::address const&, naming::id_type const&)
          , &migrate_to_storage_here<Component>
          , migrate_to_storage_here_action<Component> >
    {};

    ///////////////////////////////////////////////////////////////////////////
    // This is executed on the locality responsible for managing the address
    // resolution for the given object.
    template <typename Component>
    future<naming::id_type> trigger_migrate_to_storage_here(
        naming::id_type const& to_migrate,
        naming::id_type const& target_storage)
    {
        if (!traits::component_supports_migration<Component>::call())
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "hpx::components::server::trigger_migrate_to_storage_here",
                "attempting to migrate an instance of a component which "
                "does not support migration");
            return make_ready_future(naming::invalid_id);
        }

        if (naming::get_locality_id_from_id(to_migrate) != get_locality_id())
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "hpx::components::server::trigger_migrate_to_storage_here",
                "this function has to be executed on the locality responsible "
                "for managing the address of the given object");
            return make_ready_future(naming::invalid_id);
        }

        return agas::begin_migration(to_migrate)
            .then(
                [to_migrate, target_storage](
                    future<std::pair<naming::id_type, naming::address> > && f)
                        -> future<naming::id_type>
                {
                    // rethrow errors
                    std::pair<naming::id_type, naming::address> r = f.get();

                    // perform actual object migration
                    typedef server::migrate_to_storage_here_action<Component>
                        action_type;
                    return async<action_type>(r.first, to_migrate, r.second,
                        target_storage);
                })
            .then(
                [to_migrate](future<naming::id_type> && f) -> naming::id_type
                {
                    agas::end_migration(to_migrate).get();
                    return f.get();
                });
    }

    template <typename Component>
    struct trigger_migrate_to_storage_here_action
      : ::hpx::actions::action<
            future<naming::id_type> (*)(naming::id_type const&,
                naming::id_type const&)
          , &trigger_migrate_to_storage_here<Component>
          , trigger_migrate_to_storage_here_action<Component> >
    {};
}}}

#endif


