//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/timing.hpp>

#include <hpx/supervision_dispatch/export_definitions.hpp>
#include <hpx/supervision_dispatch/server/registry.hpp>

#include <chrono>
#include <vector>

namespace hpx::supervision::testing {

    HPX_SUPERVISION_DISPATCH_EXPORT std::vector<server::registry::peer_snapshot>
    local_snapshot_peers();

    // Overrides the timeout failure_detection_loop() passes to await_terminal()
    // on each sweep. Must be called before init() starts the loop to take
    // effect for that lifecycle; has no effect on a sweep already in flight.
    // Not part of the public dispatch API - exists only to make
    // failure-detection tests deterministic and fast instead of bound by the
    // real 60s default_discovery_timeout.
    HPX_SUPERVISION_DISPATCH_EXPORT void
    set_failure_detection_poll_timeout_for_testing(
        hpx::chrono::steady_duration timeout);
}    // namespace hpx::supervision::testing
