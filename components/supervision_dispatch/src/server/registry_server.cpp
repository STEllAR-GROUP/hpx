//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/modules/naming_base.hpp>
#include <hpx/modules/runtime_distributed.hpp>
#include <hpx/modules/supervision.hpp>

#include <hpx/supervision_dispatch/server/registry.hpp>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

HPX_REGISTER_ACTION(hpx::supervision::server::registry::join_action,
    supervision_dispatch_registry_join_action)

namespace hpx::supervision::server {

    namespace {
        // Hand out a fresh, locally-unique id to serve as a "shadow" target: a
        // purely local lookup key that the supervision manager on this locality
        // uses to mirror a joined peer's lifecycle state (via
        // publish_event()/check_admission()). It is never resolved or
        // dereferenced as a component id, so it does not need to name a real,
        // live component (the public hpx::supervision API treats targets as
        // opaque lookup keys).
        hpx::id_type make_shadow_target()
        {
            static std::atomic<std::uint64_t> counter{1};
            hpx::naming::gid_type const gid(
                0x2ull, counter.fetch_add(1, std::memory_order_relaxed));
            return hpx::id_type(gid, hpx::id_type::management_type::unmanaged);
        }
    }    // namespace

    registry::registry() = default;

    // Register for lifecycle-event notifications published for the peer's
    // sentinel. Once the peer reaches a terminal event (completed or failed),
    // mirror it onto the local shadow at the same epoch (reusing the epoch
    // reported by the notification rather than inventing one), so that any
    // local consumer consulting the shadow (check_admission(),
    // await_terminal(), ...) observes the peer's terminal state without having
    // to talk to the peer's locality itself. `shadow` is captured by value
    // rather than looked up from `peers_` on every callback invocation: join()
    // is idempotent and never reassigns a peer's shadow once created (see the
    // re-join handling above), so the captured id can never go stale, and this
    // sidesteps the need to take `mtx_` from within a callback that may run
    // asynchronously with respect to join()/re-join races.
    std::pair<hpx::id_type, hpx::id_type> registry::register_observers(
        hpx::id_type const& peer_sentinel, hpx::id_type const& peer_locality,
        hpx::id_type const& shadow)
    {
        hpx::id_type activity_observer;
        hpx::id_type lifecycle_observer;
        try
        {
            // Register for lifecycle-event notifications published for the
            // peer's sentinel. Terminal notifications are re-published onto
            // the local shadow (see below).
            lifecycle_observer = hpx::supervision::register_observer(
                hpx::launch::sync, peer_locality, peer_sentinel,
                [shadow](hpx::supervision::lifecycle_event_notification const&
                        notification) {
                    if (hpx::supervision::is_terminal(notification.event))
                    {
                        // `event::completed` is only reachable from `running`
                        // or `suspending`, whereas this observer only mirrors
                        // the terminal notification itself, not the peer's full
                        // event history; bridge through an intermediate
                        // `running` transition (valid from the `started` seed
                        // above) at the same epoch before latching `completed`.
                        // `event::failed` needs no such bridge: it is directly
                        // reachable from `started`.
                        if (notification.event ==
                            hpx::supervision::event::completed)
                        {
                            hpx::supervision::publish_event(shadow,
                                hpx::supervision::event::running,
                                notification.epoch);
                        }
                        hpx::supervision::publish_event(
                            shadow, notification.event, notification.epoch);
                    }
                    return true;
                });

            // Register for activity-state transitions of any target tracked on the
            // peer's locality (this includes the peer sentinel itself).
            // activity_notification carries an active/inactive transition, not a
            // lifecycle event, so it has no natural mapping onto the shadow's
            // publish_event()-driven terminal-latch state; it intentionally remains
            // a no-op for this task and is left for a later task to decide
            // whether/how activity transitions should feed into dispatch decisions.
            activity_observer = hpx::supervision::register_activity_observer(
                hpx::launch::sync, peer_locality,
                [](hpx::supervision::activity_notification const&) {
                    return true;
                });
        }
        catch (...)
        {
            // The activity-observer registration failed after the lifecycle
            // observer was registered successfully; unregister it again so it
            // is not leaked on the peer's locality.
            if (lifecycle_observer)
            {
                hpx::supervision::unregister_observer(
                    hpx::launch::sync, peer_locality, lifecycle_observer);
            }

            {
                std::unique_lock l(mtx_);
                peers_.erase(peer_sentinel);
                cond_.notify_all(HPX_MOVE(l));
            }

            std::rethrow_exception(std::current_exception());
        }

        return std::make_pair(lifecycle_observer, activity_observer);
    }

    std::pair<hpx::id_type, bool> registry::reserve_ownership(
        hpx::id_type const& peer_sentinel)
    {
        std::unique_lock l(mtx_);
        for (;;)
        {
            // Reserve ownership of peer_sentinel before performing any of
            // the remote registration calls below, so that concurrent
            // to register duplicate observers. If an entry already exists,
            // this either means it is fully joined (ready == true, in which
            // case its shadow is returned immediately) or that another call
            // is still in the process of joining it.
            auto const [it, inserted] =
                peers_.try_emplace(peer_sentinel, peer_entry{});
            if (inserted)
            {
                break;
            }

            if (it->second.ready)
            {
                return {it->second.shadow, true};
            }

            // Another call already reserved this peer; wait for it to
            // either publish the joined entry or release the reservation
            // (on failure), then retry.
            cond_.wait(l);
        }

        return {};
    }

    hpx::id_type registry::join(
        hpx::id_type const& peer_sentinel, hpx::id_type const& peer_locality)
    {
        // Reserve ownership of peer_sentinel before performing any of
        // the remote registration calls.
        auto [shadow, reserved] = reserve_ownership(peer_sentinel);
        if (reserved)
        {
            return shadow;
        }

        shadow = make_shadow_target();

        // Seed the shadow with `started`, giving it a well-defined starting
        // point in hpx::supervision's lifecycle state machine (see
        // hpx::supervision::is_valid_transition()) before any terminal
        // notification for the peer arrives. Without this, mirroring a terminal
        // event directly onto a shadow that has never recorded any event
        // (`event::unknown`) would be rejected as an invalid transition, since
        // `unknown` may only transition to `started`. This activation is local
        // to the shadow's own event history and independent of whatever event
        // history the peer itself went through.
        hpx::supervision::publish_event(
            shadow, hpx::supervision::event::started, 0);

        // Register for lifecycle-event notifications published for the peer's
        // sentinel.
        auto const& [lifecycle_observer, activity_observer] =
            register_observers(peer_sentinel, peer_locality, shadow);

        {
            std::unique_lock l(mtx_);

            // The placeholder entry was already inserted by reserve_ownership()
            // above, and only this call can ever fill it in, so look it up
            // rather than try_emplace()-ing again.
            peer_entry& entry = peers_.at(peer_sentinel);
            HPX_ASSERT(!entry.ready);

            entry.shadow = shadow;
            entry.lifecycle_observer = lifecycle_observer;
            entry.activity_observer = activity_observer;
            entry.ready = true;

            cond_.notify_all(HPX_MOVE(l));
        }

        return shadow;
    }
}    // namespace hpx::supervision::server

#include <hpx/config/warnings_suffix.hpp>
