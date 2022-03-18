//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/plain_action.hpp>
#include <hpx/components/get_ptr.hpp>
#include <hpx/futures/traits/get_remote_result.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime_components/components_fwd.hpp>
#include <hpx/runtime_distributed/stubs/runtime_support.hpp>

#include <memory>

namespace hpx { namespace components { namespace server {

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Copy given component to the specified target locality
    namespace detail {
        // If we know that the new component has to be created local to the old
        // one, we can avoid doing serialization.
        template <typename Component>
        hpx::id_type copy_component_here_postproc(
            std::shared_ptr<Component> ptr)
        {
            // This is executed on the locality where the component lives.
            hpx::components::server::runtime_support* rts =
                hpx::get_runtime_support_ptr();

            return traits::get_remote_result<id_type, naming::gid_type>::call(
                rts->copy_create_component<Component>(ptr, true));
        }

        template <typename Component>
        hpx::id_type copy_component_postproc(
            std::shared_ptr<Component> ptr, hpx::id_type const& target_locality)
        {
            using stubs::runtime_support;

            if (!target_locality || target_locality == find_here())
            {
                // This is executed on the locality where the component lives,
                // if no target_locality is given we have to create the copy on
                // the locality of the component.
                hpx::components::server::runtime_support* rts =
                    hpx::get_runtime_support_ptr();

                return traits::get_remote_result<id_type, naming::gid_type>::
                    call(rts->copy_create_component<Component>(ptr, true));
            }

            return runtime_support::copy_create_component<Component>(
                target_locality, ptr, false);
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    future<hpx::id_type> copy_component_here(hpx::id_type const& to_copy)
    {
        future<std::shared_ptr<Component>> f = get_ptr<Component>(to_copy);
        return f.then(
            [=](future<std::shared_ptr<Component>> f) -> hpx::id_type {
                return detail::copy_component_here_postproc(f.get());
            });
    }

    template <typename Component>
    future<hpx::id_type> copy_component(
        hpx::id_type const& to_copy, hpx::id_type const& target_locality)
    {
        future<std::shared_ptr<Component>> f = get_ptr<Component>(to_copy);
        return f.then(
            [=](future<std::shared_ptr<Component>> f) -> hpx::id_type {
                return detail::copy_component_postproc(
                    f.get(), target_locality);
            });
    }

    template <typename Component>
    struct copy_component_action_here
      : ::hpx::actions::action<future<hpx::id_type> (*)(hpx::id_type const&),
            &copy_component_here<Component>,
            copy_component_action_here<Component>>
    {
    };

    template <typename Component>
    struct copy_component_action
      : ::hpx::actions::action<future<hpx::id_type> (*)(
                                   hpx::id_type const&, hpx::id_type const&),
            &copy_component<Component>, copy_component_action<Component>>
    {
    };
}}}    // namespace hpx::components::server
