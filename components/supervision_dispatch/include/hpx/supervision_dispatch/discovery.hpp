//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/naming_base.hpp>
#include <hpx/modules/timing.hpp>

#include <hpx/supervision_dispatch/export_definitions.hpp>
#include <hpx/supervision_dispatch/registry.hpp>
#include <hpx/supervision_dispatch/sentinel.hpp>

#include <chrono>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::supervision {

    // Default bound for discover_peers(): matches the supervision module's own
    // default await_terminal() timeout (see default_await_terminal_timeout_ms
    // in supervision_manager_server.cpp) -- long enough to absorb
    // AGAS/peer-startup jitter, short enough to keep a one-time discovery pass
    // cheap and bounded.
    inline constexpr std::chrono::milliseconds default_discovery_timeout{60000};

    // A peer whose supervision_dispatch sentinel and registry basenames (see
    // sentinel::register_basename()/registry::register_basename()) were both
    // successfully resolved by discover_peers().
    struct discovered_peer
    {
        hpx::id_type locality;
        sentinel sentinel_client;
        registry registry_client;
    };

    /// Performs a one-time discovery pull for supervision_dispatch peers:
    /// concurrently resolves the pinned sentinel and registry basenames (see
    /// register_basename()) of every remote locality
    /// (hpx::find_remote_localities(), which already excludes this locality),
    /// bounded by a single wait for \a timeout across all candidates.
    ///
    /// Because constructing a sentinel/registry client from a symbolic name
    /// (see client_base's symbolic-name constructor) waits/resolves once the
    /// symbol is bound rather than failing fast on one that is not (yet)
    /// registered, localities that never called register_basename() (e.g.
    /// because they have not run init_supervision() yet) would otherwise hang
    /// this call indefinitely. Instead, once \a timeout elapses, only the
    /// candidates whose sentinel *and* registry clients had both resolved by
    /// then (checked via the non-blocking is_ready()) are returned; every other
    /// candidate is simply excluded as not currently participating in
    /// supervision -- discover_peers() itself never fails or throws on their
    /// account.
    ///
    /// \param timeout  The maximum duration to wait, across all candidates
    ///                 combined, for their sentinel and registry basenames to
    ///                 resolve.
    ///
    /// \return         The peers whose sentinel and registry clients both
    ///                 resolved within \a timeout, paired with the numeric
    ///                 locality id they are pinned to.
    HPX_SUPERVISION_DISPATCH_EXPORT std::vector<discovered_peer> discover_peers(
        hpx::chrono::steady_duration const& timeout =
            default_discovery_timeout);

    /// Reactively fans out join() calls from \a local_registry to every peer in
    /// \a peers (typically the result of a prior discover_peers() call):
    /// for each entry, this calls `local_registry.join(peer.sentinel_client,
    /// peer.locality)`.
    ///
    /// This reuses registry::join()'s existing reservation/idempotency
    /// machinery unchanged (see server::registry::reserve_ownership()) rather
    /// than duplicating any of it, so calling fan_out_join() more than once
    /// with (partially) overlapping peer lists -- or racing with an inbound
    /// join() call from one of those same peers, e.g. because that peer is
    /// concurrently running its own fan_out_join() against this locality -- is
    /// safe and never creates more than one shadow per peer sentinel.
    ///
    /// fan_out_join() itself is reactive only: it performs exactly one round of
    /// join() calls for the given \a peers and starts no background/polling
    /// activity of its own. Composing this with a discover_peers() call to
    /// produce \a peers, as a single entry point, is intentionally left to a
    /// later task.
    ///
    /// \param local_registry The registry to join every peer in \a peers to.
    /// \param peers          The peers to join, typically returned by a prior
    ///                       discover_peers() call.
    ///
    /// \return The shadow target id that \a local_registry created (or already
    ///         had) for each entry in \a peers, in the same order.
    HPX_SUPERVISION_DISPATCH_EXPORT std::vector<hpx::id_type> fan_out_join(
        registry const& local_registry,
        std::vector<discovered_peer> const& peers);
}    // namespace hpx::supervision

#include <hpx/config/warnings_suffix.hpp>
