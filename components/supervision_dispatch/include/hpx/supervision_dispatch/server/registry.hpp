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
#include <hpx/modules/naming_base.hpp>
#include <hpx/modules/supervision.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/tracing.hpp>

#include <hpx/supervision_dispatch/export_definitions.hpp>

#include <map>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::supervision::server {

    ///////////////////////////////////////////////////////////////////////////
    class HPX_SUPERVISION_DISPATCH_EXPORT registry
      : public hpx::components::component_base<registry>
    {
    public:
        registry();

        /// \brief Joins a peer sentinel and creates or reuses its shadow.
        /// \param peer_sentinel The peer sentinel to observe.
        /// \param peer_locality The locality that owns \p peer_sentinel.
        hpx::id_type join(hpx::id_type const& peer_sentinel,
            hpx::id_type const& peer_locality);

        struct join_action
          : hpx::actions::make_action_t<decltype(&registry::join),
                &registry::join, join_action>
        {
        };

    protected:
        std::pair<hpx::id_type, bool> reserve_ownership(
            hpx::id_type const& peer_sentinel);
        std::pair<hpx::id_type, hpx::id_type> register_observers(
            hpx::id_type const& peer_sentinel,
            hpx::id_type const& peer_locality, hpx::id_type const& shadow);

    private:
        struct peer_entry
        {
            hpx::id_type shadow;
            hpx::id_type lifecycle_observer;
            hpx::id_type activity_observer;

            // False while a join() call has reserved ownership of this peer
            // sentinel (i.e. is in the process of performing the remote
            // observer registrations) but has not yet filled in the rest of
            // the entry. Concurrent join() calls for the same peer wait on
            // cond_ instead of racing to register duplicate observers.
            bool ready = false;
        };

        hpx::spinlock mtx_;
        hpx::lcos::local::detail::condition_variable cond_;

        std::map<hpx::id_type, peer_entry> peers_;
    };
}    // namespace hpx::supervision::server

HPX_REGISTER_ACTION_DECLARATION(hpx::supervision::server::registry::join_action,
    supervision_dispatch_registry_join_action)

#include <hpx/config/warnings_suffix.hpp>
