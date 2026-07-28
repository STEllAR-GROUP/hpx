//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/modules/runtime_components.hpp>
#include <hpx/modules/runtime_distributed.hpp>

#include <hpx/supervision_dispatch/registry.hpp>
#include <hpx/supervision_dispatch/server/registry.hpp>

#include <cstddef>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::supervision {

    registry::registry(hpx::id_type const& target_locality)
      : base_type(hpx::new_<server::registry>(
            target_locality ? target_locality : hpx::find_here()))
    {
    }

    registry::registry(hpx::future<hpx::id_type>&& f)
      : base_type(HPX_MOVE(f))
    {
    }

    hpx::future<hpx::id_type> registry::join(
        sentinel const& peer_sentinel, hpx::id_type const& peer_locality) const
    {
        using action_type = server::registry::join_action;
        return hpx::async<action_type>(
            this->get_id(), peer_sentinel.get_id(), peer_locality);
    }

    hpx::id_type registry::join(hpx::launch::sync_policy,
        sentinel const& peer_sentinel, hpx::id_type const& peer_locality,
        hpx::error_code& ec) const
    {
        using action_type = server::registry::join_action;
        return hpx::async<action_type>(
            this->get_id(), peer_sentinel.get_id(), peer_locality)
            .get(ec);
    }
}    // namespace hpx::supervision

#include <hpx/config/warnings_suffix.hpp>
