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
        template <typename Component>
        unique_future<naming::id_type> migrate_component_postproc(
            unique_future<boost::shared_ptr<Component> > f,
            naming::id_type const& to_migrate,
            naming::id_type const& target_locality)
        {
            return stubs::runtime_support::migrate_component_async<Component>(
                target_locality, f.get(), to_migrate);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    unique_future<naming::id_type> migrate_component(
        naming::id_type const& to_migrate,
        naming::id_type const& target_locality)
    {
        if (target_locality == hpx::find_here())
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "hpx::components::server::migrate_component<Component>",
                "can't migrate component to same locality.");
            return make_ready_future(naming::invalid_id);
        }

        unique_future<boost::shared_ptr<Component> > f =
            get_ptr<Component>(to_migrate);

        return f.then(
            util::bind(&detail::migrate_component_postproc<Component>, 
                util::placeholders::_1, to_migrate, target_locality));
    }

    template <typename Component>
    struct migrate_component_action
      : ::hpx::actions::plain_result_action2<
            unique_future<naming::id_type>,
            naming::id_type const&, naming::id_type const&
          , &migrate_component<Component>
          , migrate_component_action<Component> >
    {};
}}}

HPX_REGISTER_PLAIN_ACTION_TEMPLATE(
    (template <typename Component>),
    (hpx::components::server::migrate_component_action<Component>)
)

#endif

