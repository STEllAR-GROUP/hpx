//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/async_combinators.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/naming_base.hpp>
#include <hpx/modules/runtime_distributed.hpp>
#include <hpx/modules/synchronization.hpp>

#include <hpx/supervision_dispatch/discovery.hpp>
#include <hpx/supervision_dispatch/dispatch_api.hpp>
#include <hpx/supervision_dispatch/registry.hpp>
#include <hpx/supervision_dispatch/sentinel.hpp>
#include <hpx/supervision_dispatch/testing.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>

namespace hpx::supervision {

    namespace {

        // Defaults to hpx::supervision::default_discovery_timeout; overridable
        // only for tests, via
        // testing::set_failure_detection_poll_timeout_for_testing. Relaxed
        // atomic is sufficient: this is a coarse test knob, not something
        // failure_detection_loop() needs strict ordering against other state
        // for.
        std::atomic<std::int64_t> failure_detection_poll_timeout_ms{
            hpx::supervision::default_discovery_timeout.count()};

        std::chrono::milliseconds current_failure_detection_poll_timeout()
        {
            return std::chrono::milliseconds(
                failure_detection_poll_timeout_ms.load(
                    std::memory_order_relaxed));
        }

        // The four states of the supervision-dispatch lifecycle. Transitions
        // are always driven by a successful compare_exchange on
        // dispatch_state::state, so exactly one caller ever "wins" each
        // transition and is responsible for the side effects that go with it.
        enum class dispatcher_lifecycle_state : std::uint8_t
        {
            uninitialized = 0,
            initializing = 1,
            active = 2,
            finalizing = 3
        };

        // File-local singleton holding all mutable state for init()/
        // finalize()/is_initialized(). A single instance is shared by every
        // caller on this locality via get_dispatch_state() below.
        struct dispatch_state
        {
            // Current lifecycle state. Read (via is_initialized()) without
            // locking mtx_; all writes happen only while transitioning under a
            // winning compare_exchange, matched by the corresponding side
            // effects being fully applied (init) or fully undone
            // (finalize/rollback) before the state becomes externally visible
            // as active/uninitialized again.
            std::atomic<dispatcher_lifecycle_state> state_{
                dispatcher_lifecycle_state::uninitialized};

            // Guards sentinel_/registry_ and in_flight_init_ together, since
            // both are read-modify-write as a unit whenever a lifecycle
            // transition is in progress (a losing caller reading
            // in_flight_init_ must never observe a partially-constructed
            // sentinel_/registry_ pair, and vice versa).
            hpx::spinlock mtx_;

            // Monotonically increasing epoch, incremented once per successful
            // init() call, and used as the epoch argument to the
            // completed-event publish_event() call.
            std::atomic<std::uint64_t> epoch_{0};

            // The locally owned sentinel/registry component pair. Only non-null
            // while state is initializing/active; reset to nullptr (destroying
            // the underlying components via client_'s default
            // manage_lifetime=true) once finalize() unregisters their names.
            hpx::supervision::sentinel sentinel_;
            hpx::supervision::registry registry_;

            // Shared handle to the in-flight init() call's outcome, published
            // by the transition winner immediately after it wins the
            // uninitialized -> initializing compare_exchange, and cleared
            // (reset) once that call settles (whether it succeeds, throws, or
            // rolls back). Concurrent callers observing state == initializing
            // attach to this shared_future instead of starting a second,
            // redundant init sequence.
            hpx::shared_future<void> in_flight_init_;

            // Owned/guarded the same way sentinel_/registry_ already are.
            // failure_detection_task_ holds the running poll loop so finalize()
            // can join it before tearing down registry_/sentinel_.
            hpx::future<void> failure_detection_task_;

            // Cooperative stop flag. Checked at loop-top and per-peer so a call
            // to finalize() can interrupt an in-flight poll cycle promptly
            // instead of waiting out the full await_terminal timeout.
            std::atomic<bool> stop_failure_detection_{false};
        };

        // Meyer's singleton: constructed on first use, destroyed at program
        // exit, and shared by every init()/finalize()/
        // is_initialized() call on this locality.
        dispatch_state& get_dispatch_state() noexcept
        {
            static dispatch_state state;
            return state;
        }

        // Runs the actual initialization sequence. Only ever invoked by the
        // caller that won the uninitialized -> initializing transition.
        void run_init_sequence(
            hpx::chrono::steady_duration const& discovery_timeout)
        {
            dispatch_state& ds = get_dispatch_state();

            // Steps 1-2: create the local sentinel/registry pair and publish
            // event::started before either name is registered, so no peer can
            // ever discover and join a not-yet-started sentinel.
            auto local_sentinel = hpx::supervision::sentinel(hpx::find_here());
            auto local_registry = hpx::supervision::registry(hpx::find_here());

            try
            {
                // New, strictly increasing epoch for this cycle's event::start
                // publication.
                std::uint64_t const new_epoch = ++ds.epoch_;
                local_sentinel.start(hpx::launch::sync, new_epoch);

                // Step 3: register both names.
                local_sentinel.register_name(hpx::launch::sync);
                local_registry.register_name(hpx::launch::sync);

                // Step 4: single reactive discover_and_join() pass.
                discover_and_join(local_registry, discovery_timeout);

                // Success: install the pair for finalize() and is_initialized()
                // to observe.
                {
                    std::scoped_lock<hpx::spinlock> l(ds.mtx_);
                    ds.sentinel_ = HPX_MOVE(local_sentinel);
                    ds.registry_ = HPX_MOVE(local_registry);
                }
            }
            catch (...)
            {
                // Step 5: unregister both names, best-effort.
                hpx::error_code ec1(hpx::throwmode::lightweight);
                local_registry.unregister_name(hpx::launch::sync, ec1);
                hpx::error_code ec2(hpx::throwmode::lightweight);
                local_sentinel.unregister_name(hpx::launch::sync, ec2);

                std::rethrow_exception(std::current_exception());
            }
        }

        bool needs_stopping(dispatch_state const& ds) noexcept
        {
            return ds.stop_failure_detection_.load(std::memory_order_acquire) ||
                !is_initialized();
        }

        // Active-probing complement to the reactive eviction path (evict_peer/
        // cleanup_peer). While that path fences a peer when its lifecycle
        // observer receives a terminal callback, this loop catches the case
        // where a peer goes silent without ever delivering one (e.g. a hard
        // locality crash where no callback fires).
        //
        // Lifecycle:
        //   - Started as the last step of run_init_sequence()'s success path,
        //     after discover_and_join() returns, so it only ever iterates over
        //     peers already joined and never races initialization.
        //   - Stopped from finalize(): stop_failure_detection_ is flipped and
        //     the owning hpx::future<void> is .get()'d before sentinel_/
        //     registry_ are torn down, since the loop dereferences the
        //     registry's shadows while running.
        //
        // Per sweep:
        //   - Snapshots currently-joined, ready, non-evicting peers via
        //     registry_->snapshot_peers() (no new registry state was added for
        //     this: the accessor already existed for this purpose).
        //   - Dispatches await_terminal(...) concurrently (async, not
        //     sequential sync calls) for every peer in the sweep, so an
        //     unresponsive peer doesn't block the detection of others and
        //     doesn't multiply the sweep's total duration by peer count.
        //   - Each dispatch is read against a freshly queried epoch
        //     (query_state), since peer_snapshot itself carries no epoch.
        //
        // Fencing semantics:
        //   - hpx::error::future_cancelled on a peer's await_terminal future
        //     means a genuine timeout: no terminal event arrived under the
        //     epoch we waited on. In that case we bump and publish
        //     event::failed against *our own local copy* of that peer's shadow
        //     only. We never contact the (possibly dead) peer's own
        //     sentinel/registry - the remote re-check inside
        //     invoke_fenced_action is what is authoritative on the target
        //     locality; this loop only needs the local shadow to reflect
        //     failure so future check_admission()/dispatch_work() calls against
        //     it observe the fence.
        //   - hpx::error::stale_state means the epoch was already advanced
        //     (typically the reactive eviction path already fired) - no action
        //     needed.
        //   - Any other outcome is a genuinely delivered terminal event -
        //     nothing to do; the reactive path already handles it.
        //   - check_admission() itself is never called from this loop; it
        //     remains exclusively the caller-side gate inside
        //     dispatch_work.hpp.
        //
        // Cancellation/shutdown behavior:
        //   - The outer sweep loop and the per-peer dispatch loop both check
        //     stop_failure_detection_ and is_initialized() before doing any
        //     more work, so a call to finalize() is observed promptly rather
        //     than only between full sweeps.
        //   - Sweep completion is polled in short poll_interval slices via
        //     wait_all_for rather than a single blocking wait_all, so
        //     finalize() is bounded by ~poll_interval, not by
        //     default_discovery_timeout, even if a sweep has unresponsive peers
        //     in flight.
        //   - If the stop signal fires mid-sweep, outstanding per-peer
        //     continuations are abandoned (not cancelled): they hold their own
        //     reference-counted future shared state independent of the `checks`
        //     vector, so they run to completion safely after this function
        //     returns. This is intentionally allowed: each continuation only
        //     touches a captured-by-value shadow/epoch and calls publish_event
        //     against that local shadow, never reaching into dispatch_state's
        //     registry_/sentinel_, so it cannot race finalize()'s teardown.
        //     Consequently, finalize() returning promptly does not imply all
        //     background poll activity has ceased - only that this loop itself
        //     has stopped issuing new sweeps.
        void failure_detection_loop()
        {
            dispatch_state const& ds = get_dispatch_state();

            // Outer loop: keep polling as long as we haven't been asked to stop
            // and the dispatcher is still in the "active" state.
            // is_initialized() guards against racing finalize() that's already
            // in progress.
            while (!needs_stopping(ds))
            {
                // Snapshot of currently-joined, ready, non-evicting peers. Uses
                // the sync overload so this call blocks inline rather than
                // returning a future we'd have to chain off of.
                auto const peers =
                    ds.registry_.snapshot_peers(hpx::launch::sync);

                std::vector<hpx::future<void>> checks;
                checks.reserve(peers.size());

                for (auto const& peer : peers)
                {
                    // peer_snapshot carries no epoch of its own
                    auto const state = hpx::supervision::query_state(
                        hpx::launch::sync, hpx::find_here(), peer.shadow);

                    auto cont = [peer, epoch = state.epoch](auto&& f) {
                        try
                        {
                            f.get();    // success: real terminal event delivered
                        }
                        catch (hpx::exception const& e)
                        {
                            if (e.get_error() == hpx::error::future_cancelled)
                            {
                                // Genuine timeout: fence locally only, never
                                // touch the peer's own sentinel/registry.
                                hpx::supervision::publish_event(
                                    hpx::launch::sync, hpx::find_here(),
                                    peer.shadow, event::failed, epoch + 1);
                            }
                            // stale_state: epoch already advanced (reactive
                            // eviction already fired) - nothing to do.
                        }
                    };

                    auto const poll_timeout =
                        current_failure_detection_poll_timeout();
                    checks.push_back(await_terminal(hpx::find_here(),
                        peer.shadow, state.epoch, poll_timeout)
                            .then(hpx::launch::sync, HPX_MOVE(cont)));
                }

                // Poll the batch in short slices so the stop flag is rechecked
                // every poll_interval instead of only after the full sweep
                // resolves. This is what makes finalize() exit promptly without
                // a second stop_promise_/wait_any mechanism.
                constexpr std::chrono::milliseconds poll_interval{200};
                while (!needs_stopping(ds) &&
                    hpx::wait_all_for(poll_interval, checks) ==
                        hpx::future_status::timeout)
                {
                    // still waiting on this sweep; recheck stop flag
                }

                // If stop fired mid-sweep, finalize() isn't blocked on the
                // stragglers: each continuation only touches a *copy* of
                // peer.shadow (captured by value) and publishes against that
                // local shadow - it never reaches into ds.registry_/ds.sentinel_,
                // so letting any still-running continuations finish in the
                // background is safe.
            }
        }
    }    // namespace

    namespace testing {

        std::vector<server::registry::peer_snapshot> local_snapshot_peers()
        {
            dispatch_state const& ds = get_dispatch_state();

            // Same guard failure_detection_loop() itself uses before touching
            // registry_ - returns empty if not currently active.
            if (needs_stopping(ds))
                return {};
            return ds.registry_.snapshot_peers(hpx::launch::sync);
        }

        void set_failure_detection_poll_timeout_for_testing(
            hpx::chrono::steady_duration const timeout)
        {
            failure_detection_poll_timeout_ms.store(
                timeout.value().count(), std::memory_order_relaxed);
        }
    }    // namespace testing

    ////////////////////////////////////////////////////////////////////////////
    hpx::shared_future<void> init(
        hpx::chrono::steady_duration const& discovery_timeout)
    {
        dispatch_state& ds = get_dispatch_state();

        auto expected = dispatcher_lifecycle_state::uninitialized;
        if (ds.state_.compare_exchange_strong(
                expected, dispatcher_lifecycle_state::initializing))
        {
            // Winner: publish in_flight_init_ before running the sequence so
            // losers observing "initializing" have something to attach to.
            hpx::future<void> f =
                hpx::async(&run_init_sequence, discovery_timeout);

            hpx::shared_future<void> sf =
                f.then(hpx::launch::sync, [&ds](hpx::future<void>&& result) {
                    if (result.has_value())
                    {
                        std::scoped_lock<hpx::spinlock> l(ds.mtx_);
                        ds.in_flight_init_ = shared_future<void>();
                        ds.state_.store(dispatcher_lifecycle_state::active);
                    }
                    else
                    {
                        std::scoped_lock<hpx::spinlock> l(ds.mtx_);
                        ds.sentinel_ = hpx::supervision::sentinel();
                        ds.registry_ = hpx::supervision::registry();

                        ds.in_flight_init_ = shared_future<void>();
                        ds.state_.store(
                            dispatcher_lifecycle_state::uninitialized);

                        // Reset the stop flag in case a prior init/finalize
                        // cycle left it set (dispatch_api can presumably be
                        // re-initialized after finalize()).
                        ds.stop_failure_detection_.store(
                            false, std::memory_order_release);

                        // Start the poll loop as the last step of the success
                        // path - after discover_and_join() has returned, so it
                        // only ever iterates over peers that are already joined
                        // and never races initialization.
                        ds.failure_detection_task_ =
                            hpx::async(&failure_detection_loop);
                    }
                    result.get();    // rethrow exception, if any
                });

            {
                std::scoped_lock<hpx::spinlock> l(ds.mtx_);
                ds.in_flight_init_ = sf;
            }

            return sf;
        }

        switch (expected)
        {
        case dispatcher_lifecycle_state::active:
            return hpx::make_ready_future();

        case dispatcher_lifecycle_state::finalizing:
            return hpx::make_exceptional_future<void>(HPX_GET_EXCEPTION(
                hpx::error::invalid_status, "hpx::supervision::init",
                "cannot initialize while a concurrent "
                "finalize() is in progress"));

        case dispatcher_lifecycle_state::initializing:
        default:
            // The picked 30s timeout is long enough that a normal CAS-publish
            // handoff will never legitimately hit it, but also short enough
            // that a genuinely stuck/deadlocked winner still eventually
            // surfaces an error instead of hanging forever
            auto const timeout =
                hpx::chrono::steady_duration(std::chrono::seconds(30))
                    .from_now();

            // Attach to the in-flight init. NOTE: narrow, documented spin
            // window between the winner's CAS and its in_flight_init_ publish.
            while (std::chrono::steady_clock::now() < timeout)
            {
                {
                    std::scoped_lock<hpx::spinlock> l(ds.mtx_);
                    if (ds.in_flight_init_.valid())
                    {
                        return ds.in_flight_init_;
                    }
                }

                auto const current = ds.state_.load(std::memory_order_acquire);
                if (current == dispatcher_lifecycle_state::active)
                {
                    return hpx::make_ready_future();
                }
                if (current != dispatcher_lifecycle_state::initializing)
                {
                    break;
                }
                hpx::this_thread::yield();
            }

            return hpx::make_exceptional_future<void>(HPX_GET_EXCEPTION(
                hpx::error::invalid_status, "hpx::supervision::init",
                "timed out attaching to a concurrent "
                "init() call; retry"));
        }
    }

    void init(hpx::launch::sync_policy,
        hpx::chrono::steady_duration const& discovery_timeout)
    {
        init(discovery_timeout).get();
    }

    void finalize()
    {
        dispatch_state& ds = get_dispatch_state();

        if (auto expected = dispatcher_lifecycle_state::active;
            !ds.state_.compare_exchange_strong(
                expected, dispatcher_lifecycle_state::finalizing))
        {
            // Documented no-op: uninitialized, still initializing, or already
            // finalizing all resolve immediately with zero side effects - no
            // event published, no name touched, no component destroyed.
            return;
        }

        // Signal the loop to stop as early as possible in finalize(), before
        // any teardown of the state the loop still references.
        ds.stop_failure_detection_.store(true, std::memory_order_release);
        if (ds.failure_detection_task_.valid())
        {
            // Join the task before releasing sentinel_/registry_ ownership -
            // the loop dereferences registry_'s shadows, so this must complete
            // first to avoid a dangling reference during extraction/publish.
            ds.failure_detection_task_.get();
        }

        // Winner: extract the owned pair under the lock so no other thread can
        // observe a half-torn-down sentinel_/registry_.
        hpx::supervision::sentinel local_sentinel;
        hpx::supervision::registry local_registry;

        {
            std::scoped_lock<hpx::spinlock> l(ds.mtx_);
            local_sentinel = HPX_MOVE(ds.sentinel_);
            local_registry = HPX_MOVE(ds.registry_);

            // Drop any settled handle from the completed init cycle so a later
            // cycle's attaching caller can never observe it.
            ds.in_flight_init_ = hpx::shared_future<void>();
        }

        HPX_ASSERT(local_sentinel);    // active implies sentinel_ was set

        // Publish event::completed synchronously so observers see it before
        // this function returns, using the same publish_event(sync_policy,
        // locality, target, event, epoch) shape sentinel::start() uses for
        // event::started.

        // First make sure the current state has advanced to running as otherwise
        // the publish_event call below will fail.
        hpx::id_type const here = hpx::find_here();
        hpx::id_type const sentinel_id = local_sentinel.get_id();
        auto const epoch = ds.epoch_.load(std::memory_order_acquire);

        std::exception_ptr publish_failure;
        try
        {
            auto const state = hpx::supervision::query_state(
                hpx::launch::sync, here, sentinel_id);
            if (state.last_event == hpx::supervision::event::started)
            {
                hpx::supervision::publish_event(hpx::launch::sync, here,
                    sentinel_id, hpx::supervision::event::running, epoch);
            }
            hpx::supervision::publish_event(hpx::launch::sync, here,
                sentinel_id, hpx::supervision::event::completed, epoch);
        }
        catch (...)
        {
            // Best-effort: still unregister/re-arm below even if event
            // publication failed, so a transient failure never permanently
            // strands the lifecycle at `finalizing`.
            publish_failure = std::current_exception();
        }

        // Unregister both names before releasing ownership, so no new peer
        // can discover either component mid-teardown.
        hpx::error_code ec1(hpx::throwmode::lightweight);
        local_sentinel.unregister_name(hpx::launch::sync, ec1);
        hpx::error_code ec2(hpx::throwmode::lightweight);
        local_registry.unregister_name(hpx::launch::sync, ec2);

        // Let the clients go out of scope here: client_'s default
        // manage_lifetime=true means each destructor releases this locality's
        // ownership of the underlying component.
        local_sentinel = hpx::supervision::sentinel();
        local_registry = hpx::supervision::registry();

        // Re-arm: a subsequent init() call can now start a fresh
        // sentinel/registry pair from scratch.
        ds.state_.store(dispatcher_lifecycle_state::uninitialized,
            std::memory_order_release);

        if (publish_failure)
        {
            std::rethrow_exception(publish_failure);
        }
    }

    bool is_initialized() noexcept
    {
        dispatch_state const& ds = get_dispatch_state();
        return ds.state_.load(std::memory_order_acquire) ==
            dispatcher_lifecycle_state::active;
    }
}    // namespace hpx::supervision
