//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/async_base.hpp>
#include <hpx/modules/components.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/naming_base.hpp>

#include <hpx/supervision_dispatch/sentinel.hpp>
#include <hpx/supervision_dispatch/server/registry.hpp>

#include <cstddef>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::supervision {

    ///////////////////////////////////////////////////////////////////////////
    class HPX_SUPERVISION_DISPATCH_EXPORT registry
      : public hpx::components::client_base<registry, server::registry>
    {
        using base_type =
            hpx::components::client_base<registry, server::registry>;

    public:
        explicit registry(
            hpx::id_type const& target_locality = hpx::invalid_id);
        /* implicit */ registry(hpx::future<hpx::id_type>&& f);

        // Join a peer sentinel: create (or reuse) a local shadow target
        // that mirrors the peer's lifecycle state, and register this
        // registry as an observer of the peer's lifecycle/activity events.
        // Returns the id of the local shadow target.
        hpx::future<hpx::id_type> join(sentinel const& peer_sentinel,
            hpx::id_type const& peer_locality) const;
        hpx::id_type join(hpx::launch::sync_policy,
            sentinel const& peer_sentinel, hpx::id_type const& peer_locality,
            hpx::error_code& ec = hpx::throws) const;
    };
}    // namespace hpx::supervision

#include <hpx/config/warnings_suffix.hpp>
