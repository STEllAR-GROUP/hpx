//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/actions.hpp>
#include <hpx/modules/actions_base.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/modules/components_base.hpp>
#include <hpx/modules/synchronization.hpp>

#include <hpx/supervision/supervision_api.hpp>

#include <cstddef>
#include <functional>

namespace hpx::supervision::server {

    // Local counterpart of agent_component used by
    // register_target_activity_observer()/unregister_target_activity_observer().
    // Unlike agent_component, instances of this component are only ever
    // resolved and invoked locally (via hpx::get_ptr()), never through an
    // action: activity-observer delivery is currently only wired for local
    // registrations, so this class deliberately declares no actions of its
    // own. Cross-locality delivery is added when the remote
    // register_target_activity_observer()/unregister_target_activity_observer()
    // overloads are wired in a later substep.
    class HPX_EXPORT activity_agent_component
      : public hpx::components::component_base<activity_agent_component>
    {
    public:
        activity_agent_component() = default;

        explicit activity_agent_component(activity_callback f) noexcept
          : f_(HPX_MOVE(f))
        {
        }

        bool invoke_if_active(target_activity_notification const& notify);

        struct invoke_if_active_action
          : hpx::actions::make_action_t<
                decltype(&activity_agent_component::invoke_if_active),
                &activity_agent_component::invoke_if_active,
                invoke_if_active_action>
        {
        };

        void deactivate_and_wait();

        struct deactivate_and_wait_action
          : hpx::actions::make_action_t<
                decltype(&activity_agent_component::deactivate_and_wait),
                &activity_agent_component::deactivate_and_wait,
                deactivate_and_wait_action>
        {
        };

    private:
        void finish_delivery();

        activity_callback f_;

        hpx::spinlock mtx_;
        hpx::lcos::local::detail::condition_variable cv_;
        bool active_ = true;
        std::size_t in_flight_ = 0;
    };

    // Create a local activity agent wrapping the given function
    HPX_EXPORT hpx::id_type create_activity_agent(activity_callback f);

}    // namespace hpx::supervision::server

HPX_REGISTER_ACTION_DECLARATION(
    hpx::supervision::server::activity_agent_component::invoke_if_active_action,
    activity_agent_component_invoke_if_active_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::supervision::server::activity_agent_component::
        deactivate_and_wait_action,
    activity_agent_component_deactivate_and_wait_action)
