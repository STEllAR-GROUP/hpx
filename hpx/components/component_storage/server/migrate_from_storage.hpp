//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_MIGRATE_FROM_STORAGE_SERVER_FEB_09_2015_0330PM)
#define HPX_MIGRATE_FROM_STORAGE_SERVER_FEB_09_2015_0330PM

#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/naming.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/include/util.hpp>

#include <hpx/components/component_storage/export_definitions.hpp>
#include <hpx/components/component_storage/server/component_storage.hpp>

namespace hpx { namespace components { namespace server
{
    namespace detail
    {
        // convert the extracted data into a living component instance
        template <typename Component>
        future<naming::id_type> migrate_from_storage_here_postproc(
            future<std::vector<char> > f,
            naming::id_type const& to_resurrect,
            naming::id_type const& target_locality)
        {
            // recreate the object
            boost::shared_ptr<Component> ptr;

            {
                std::vector<char> data = f.get();
                util::portable_binary_iarchive archive(
                    data, data.size(), boost::archive::no_header);
                archive >> ptr;
            }

            // make sure the migration code works properly
            ptr->pin();

            // and resurrect it on the specified locality
            using hpx::components::runtime_support;
            return runtime_support::migrate_component_async<Component>(
                        target_locality, ptr, to_resurrect)
                .then(util::bind(
                    &detail::migrate_component_cleanup<Component>,
                    util::placeholders::_1, ptr, to_resurrect));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    future<naming::id_type> migrate_from_storage_here(
        naming::id_type const& to_resurrect,
        naming::id_type const& source_storage,
        naming::id_type const& target_locality)
    {
        if (!Component::supports_migration())
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "hpx::components::server::migrate_from_storage_here",
                "attempting to migrate an instance of a component which "
                "does not support migration");
            return make_ready_future(naming::invalid_id);
        }

        // retrieve the data from the given storage
        typedef typename server::component_storage::migrate_from_here_action
            action_type;
        return async<action_type>(source_storage, to_resurrect.get_gid())
            .then(util::bind(
                &detail::migrate_from_storage_here_postproc<Component>,
                util::placeholders::_1, to_resurrect, target_locality));
    }

    template <typename Component>
    struct migrate_from_storage_here_action
      : ::hpx::actions::action<
            future<naming::id_type> (*)(naming::id_type const&,
                naming::id_type const&, naming::id_type const&)
          , &migrate_from_storage_here<Component>
          , migrate_from_storage_here_action<Component> >
    {};
}}}

HPX_REGISTER_PLAIN_ACTION_TEMPLATE(
    (template <typename Component>),
    (hpx::components::server::migrate_from_storage_here_action<Component>))

#endif


