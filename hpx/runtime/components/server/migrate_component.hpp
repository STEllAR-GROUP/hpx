//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_COMPONENTS_SERVER_MIGRATE_COMPONENT_JAN_30_2014_0737AM)
#define HPX_RUNTIME_COMPONENTS_SERVER_MIGRATE_COMPONENT_JAN_30_2014_0737AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/get_ptr.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>

#include <boost/serialization/shared_ptr.hpp>

namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Migrate given component to the specified target locality
    namespace detail
    {
        // clean up (source) memory of migrated object
        template <typename Component>
        naming::id_type migrate_component_cleanup(
            future<naming::id_type> f,
            boost::shared_ptr<Component> ptr,
            naming::id_type const& to_migrate)
        {
            ptr->mark_as_migrated();
            return to_migrate;
        }

        // trigger the actual migration
        template <typename Component>
        future<naming::id_type> migrate_component_postproc(
            future<boost::shared_ptr<Component> > f,
            naming::id_type const& to_migrate,
            naming::id_type const& target_locality)
        {
            using components::stubs::runtime_support;

            boost::shared_ptr<Component> ptr = f.get();
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
    template <typename Component>
    future<naming::id_type> migrate_component(
        naming::id_type const& to_migrate,
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
                "attempting to migrate an instance of a component which is "
                "does not support migration");
            return make_ready_future(naming::invalid_id);
        }

        return hpx::detail::get_ptr_for_migration<Component>(to_migrate)
            .then(util::bind(&detail::migrate_component_postproc<Component>,
                util::placeholders::_1, to_migrate, target_locality));
    }

    template <typename Component>
    struct migrate_component_action
      : ::hpx::actions::action<
            future<naming::id_type> (*)(naming::id_type const&, naming::id_type const&)
          , &migrate_component<Component>
          , migrate_component_action<Component> >
    {};
}}}

HPX_REGISTER_PLAIN_ACTION_TEMPLATE(
    (template <typename Component>),
    (hpx::components::server::migrate_component_action<Component>)
)

#endif

