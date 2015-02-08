//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_MIGRATE_TO_STORAGE_SERVER_FEB_04_2015_1021AM)
#define HPX_MIGRATE_TO_STORAGE_SERVER_FEB_04_2015_1021AM

#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/naming.hpp>
#include <hpx/include/util.hpp>

#include <hpx/components/migrate_to_storage/export_definitions.hpp>
#include <hpx/components/migrate_to_storage/server/component_storage.hpp>

namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // clean up (source) memory of migrated object
        template <typename Component>
        naming::id_type migrate_to_storage_here_cleanup(
            future<naming::id_type> f,
            boost::shared_ptr<Component> ptr,
            naming::id_type const& to_migrate)
        {
            ptr->mark_as_migrated();
            return f.get();
        }

        // trigger the actual migration to storage
        template <typename Component>
        future<naming::id_type> migrate_to_storage_here_postproc(
            future<boost::shared_ptr<Component> > f,
            naming::id_type const& to_migrate,
            naming::id_type const& target_storage)
        {
            using components::stubs::runtime_support;

            boost::shared_ptr<Component> ptr = f.get();
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
                util::portable_binary_oarchive archive(
                    data, (util::binary_filter*)0, boost::archive::no_header);
                archive << ptr;
            }

            naming::address addr(ptr->get_current_address());

            typedef typename server::component_storage::migrate_to_here_action
                action_type;
            return hpx::async<action_type>(
                    target_storage, std::move(data), to_migrate, addr
                ).then(util::bind(
                    &migrate_to_storage_here_cleanup<Component>,
                    util::placeholders::_1, ptr, to_migrate)
                );
        }
    }

    template <typename Component>
    future<naming::id_type> migrate_to_storage_here(
        naming::id_type const& to_migrate,
        naming::id_type const& target_storage)
    {
        if (!Component::supports_migration())
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "hpx::components::server::migrate_to_storage_here",
                "attempting to migrate an instance of a component which is "
                "does not support migration");
            return make_ready_future(naming::invalid_id);
        }

        return hpx::detail::get_ptr_for_migration<Component>(to_migrate)
            .then(util::bind(&detail::migrate_to_storage_here_postproc<Component>,
                util::placeholders::_1, to_migrate, target_storage));
    }

    template <typename Component>
    struct migrate_to_storage_here_action
      : ::hpx::actions::action<
            future<naming::id_type> (*)(naming::id_type const&,
                naming::id_type const&)
          , &migrate_to_storage_here<Component>
          , migrate_to_storage_here_action<Component> >
    {};
}}}

HPX_REGISTER_PLAIN_ACTION_TEMPLATE(
    (template <typename Component>),
    (hpx::components::server::migrate_to_storage_here_action<Component>))

#endif


