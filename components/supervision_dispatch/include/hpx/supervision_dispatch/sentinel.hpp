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

#include <hpx/supervision_dispatch/server/sentinel.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::supervision {

    ///////////////////////////////////////////////////////////////////////////
    // A sentinel is a lightweight, self-supervising handle: constructing it
    // creates a component on the given target locality, and calling start()
    // publishes the `started` lifecycle event for that component to the
    // supervision manager running on the same locality. No registry lookup or
    // discovery step is required.
    class HPX_SUPERVISION_DISPATCH_EXPORT sentinel
      : public hpx::components::client_base<sentinel, server::sentinel>
    {
        using base_type =
            hpx::components::client_base<sentinel, server::sentinel>;

    public:
        explicit sentinel(
            hpx::id_type const& target_locality = hpx::invalid_id);
        /* implicit */ sentinel(hpx::future<hpx::id_type>&& f);

        // Publish the `started` lifecycle event (at the given epoch) for
        // this sentinel to the supervision manager running on its target
        // locality.
        hpx::future<publish_result> start(std::uint64_t epoch = 0) const;
        publish_result start(hpx::launch::sync_policy, std::uint64_t epoch = 0,
            hpx::error_code& ec = hpx::throws) const;
    };
}    // namespace hpx::supervision

#include <hpx/config/warnings_suffix.hpp>
