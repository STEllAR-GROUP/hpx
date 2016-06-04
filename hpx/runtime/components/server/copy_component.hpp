//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_COMPONENTS_SERVER_COPY_COMPONENT_SEP_20_2013_0825PM)
#define HPX_RUNTIME_COMPONENTS_SERVER_COPY_COMPONENT_SEP_20_2013_0825PM

#include <hpx/config.hpp>
#include <hpx/traits/get_remote_result.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/get_ptr.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>

#include <memory>

namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Copy given component to the specified target locality
    namespace detail
    {
        // If we know that the new component has to be created local to the old
        // one, we can avoid doing serialization.
        template <typename Component>
        naming::id_type copy_component_here_postproc(
            future<std::shared_ptr<Component> > f)
        {
            // This is executed on the locality where the component lives.
            hpx::components::server::runtime_support* rts =
                hpx::get_runtime_support_ptr();

            return traits::get_remote_result<
                    id_type, naming::gid_type
                >::call(rts->copy_create_component<Component>(f.get(), true));
        }

        template <typename Component>
        naming::id_type copy_component_postproc(
            future<std::shared_ptr<Component> > f,
            naming::id_type const& target_locality)
        {
            using stubs::runtime_support;

            std::shared_ptr<Component> ptr = f.get();
            if (!target_locality || target_locality == find_here())
            {
                // This is executed on the locality where the component lives,
                // if no target_locality is given we have to create the copy on
                // the locality of the component.
                hpx::components::server::runtime_support* rts =
                    hpx::get_runtime_support_ptr();

                return traits::get_remote_result<
                        id_type, naming::gid_type
                    >::call(rts->copy_create_component<Component>(ptr, true));
            }

            return runtime_support::copy_create_component<Component>(
                target_locality, ptr, false);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    future<naming::id_type> copy_component_here(naming::id_type const& to_copy)
    {
        future<std::shared_ptr<Component> > f =
            get_ptr<Component>(to_copy);
        return f.then(&detail::copy_component_here_postproc<Component>);
    }

    template <typename Component>
    future<naming::id_type> copy_component(naming::id_type const& to_copy,
        naming::id_type const& target_locality)
    {
        future<std::shared_ptr<Component> > f =
            get_ptr<Component>(to_copy);
        return f.then(util::bind(&detail::copy_component_postproc<Component>,
            util::placeholders::_1, target_locality));
    }

    template <typename Component>
    struct copy_component_action_here
      : ::hpx::actions::action<
            future<naming::id_type> (*)(naming::id_type const&)
          , &copy_component_here<Component>
          , copy_component_action_here<Component> >
    {};

    template <typename Component>
    struct copy_component_action
      : ::hpx::actions::action<
            future<naming::id_type> (*)(naming::id_type const&, naming::id_type const&)
          , &copy_component<Component>
          , copy_component_action<Component> >
    {};
}}}

#endif

