//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Acceptance check: failure_detection_loop() (started by init(), stopped by
// finalize(), see dispatch_api.cpp) is the active-probing complement to the
// reactive eviction path already covered by registry_snapshot.cpp. Run with two
// localities so one side can be driven "unresponsive" (joined, then simply
// never publishes another event) while the other side's poller is the thing
// under test.
//
// Locality 0 is the "observer": it runs every assertion via dispatch_work()/
// check_admission() against its local view of locality 1's shadow. Locality 1
// is the "peer": for most scenarios it stays live (publishing nothing further
// is *not* required to be alive -- only the fencing scenario deliberately
// withholds all further activity), and it self-paces via barrier::synchronize()
// so the observer can drive each scenario to completion before the peer moves
// on to teardown.

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/hpx_init.hpp>
#include <hpx/modules/actions.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/runtime_distributed.hpp>
#include <hpx/modules/supervision.hpp>
#include <hpx/modules/testing.hpp>

#include <hpx/supervision_dispatch/dispatch_api.hpp>
#include <hpx/supervision_dispatch/dispatch_work.hpp>

// Proposed, not yet present: exposes the two test-only hooks described
// above. If these land under a different header/namespace, only the
// #include and call sites below need updating.
#include <hpx/supervision_dispatch/testing.hpp>

#include <chrono>
#include <cstdint>
#include <optional>
#include <vector>

// Probe action used purely to exercise dispatch_work()'s admission check; its
// body is never expected to matter, only whether it dispatches at all.
int probe()
{
    return 1;
}
HPX_PLAIN_ACTION(probe, probe_action);

namespace {

    // Shrinks failure_detection_loop()'s await_terminal() bound well below
    // the real default_discovery_timeout (60s) so tests are deterministic
    // and fast. Must be set before init() starts the loop.
    constexpr std::chrono::milliseconds test_poll_timeout{500};

    // A generous bound the *unresponsive* peer sleeps for, long enough to
    // comfortably outlast every scenario the observer runs against it.
    constexpr std::chrono::seconds peer_idle_bound{30};

    // Polls query_state()+check_admission() in short increments (rather than
    // sleeping a hardcoded number of poll cycles) until `target` reports
    // fenced under `epoch`, or `bound` elapses. Returns whether fencing was
    // observed within `bound`.
    bool wait_until_fenced(hpx::id_type const& target,
        std::uint64_t const epoch, std::chrono::milliseconds const bound)
    {
        auto const deadline = std::chrono::steady_clock::now() + bound;
        while (std::chrono::steady_clock::now() < deadline)
        {
            if (hpx::supervision::check_admission(target, epoch) ==
                hpx::supervision::dispatch_outcome::rejected_fenced)
            {
                return true;
            }
            hpx::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        return hpx::supervision::check_admission(target, epoch) ==
            hpx::supervision::dispatch_outcome::rejected_fenced;
    }

    // Finds the joined peer's shadow id for `peer_locality` in this locality's
    // own registry, or nullopt if not (yet) joined.
    std::optional<hpx::id_type> find_shadow_for(
        hpx::id_type const& peer_locality)
    {
        for (auto const& peer :
            hpx::supervision::testing::local_snapshot_peers())
        {
            if (peer.peer_locality == peer_locality)
            {
                return peer.shadow;
            }
        }
        return std::nullopt;
    }

}    // namespace

// ============================================================================
// Test Cases: Failure Detection
// ============================================================================

// Scenario 1: detection causes fencing.
//
// The peer locality joins and then goes silent -- no completed/failed event
// ever arrives, simulating a hard crash where no lifecycle callback fires.
// After one poll cycle (bounded by test_poll_timeout, not the real 60s
// default), the observer's local shadow for that peer must report fenced, and a
// subsequent dispatch_work() against it must fail with
// hpx::error::target_fenced -- proving the *local* shadow was fenced by the
// detector, without the observer ever contacting the peer's own
// sentinel/registry.
void test_detection_causes_fencing(hpx::id_type const& peer_locality)
{
    auto const shadow = find_shadow_for(peer_locality);
    HPX_TEST(shadow.has_value());
    if (!shadow)
    {
        return;
    }

    // The epoch to assert fencing against is whatever the shadow was
    // joined/started under; query it directly rather than assuming 0, since
    // discover_and_join() seeds shadows with event::started at the peer's own
    // epoch.
    auto const state = hpx::supervision::query_state(*shadow);
    std::uint64_t const epoch = state.epoch;

    // Give the poller a few sweep cycles' worth of time to notice the silence
    // and fence locally. Bound generously relative to test_poll_timeout so this
    // isn't flaky under scheduler jitter, but far short of the real
    // default_discovery_timeout.
    bool const fenced =
        wait_until_fenced(*shadow, epoch, std::chrono::seconds(5));
    HPX_TEST(fenced);

    hpx::future<int> f =
        hpx::supervision::dispatch_work<probe_action>(*shadow, epoch);

    bool caught = false;
    try
    {
        f.get();
    }
    catch (hpx::exception const& e)
    {
        caught = true;
        HPX_TEST(hpx::get_error(e) == hpx::error::target_fenced);
    }
    HPX_TEST(caught);
}

// Scenario 2: no false positives.
//
// A live, responsive peer must never be spuriously fenced across multiple poll
// cycles. Dispatches repeatedly, sleeping past at least one full poll cycle
// between each, and asserts every dispatch keeps succeeding.
void test_no_false_positives(hpx::id_type const& peer_locality)
{
    auto const shadow = find_shadow_for(peer_locality);
    HPX_TEST(shadow.has_value());
    if (!shadow)
    {
        return;
    }

    auto const state = hpx::supervision::query_state(*shadow);
    std::uint64_t const epoch = state.epoch;

    constexpr int iterations = 5;
    for (int i = 0; i != iterations; ++i)
    {
        HPX_TEST(hpx::supervision::check_admission(*shadow, epoch) ==
            hpx::supervision::dispatch_outcome::admitted);

        hpx::future<int> f =
            hpx::supervision::dispatch_work<probe_action>(*shadow, epoch);
        HPX_TEST_NO_THROW(f.get());

        // Sleep past at least one full test_poll_timeout sweep so this
        // actually exercises repeated poll cycles, not just a single one.
        hpx::this_thread::sleep_for(test_poll_timeout * 2);
    }
}

// Scenario 3: clean shutdown.
//
// finalize() must return promptly (bounded by the loop's internal poll_interval
// slicing, not by the full await_terminal timeout) even while a sweep is in
// flight against an unresponsive peer. Deliberately uses the *unshortened*
// default_discovery_timeout for this scenario's own poll bound -- shortening it
// here would hide a broken implementation that still blocks finalize() on the
// full wait.
void test_clean_shutdown_during_in_flight_sweep(
    hpx::id_type const& observer_marker)
{
    (void) observer_marker;

    // Re-init with the real, unshortened timeout so this scenario actually
    // exercises the "long wait in flight" case finalize() must not block on.
    hpx::supervision::testing::set_failure_detection_poll_timeout_for_testing(
        hpx::supervision::default_discovery_timeout);

    HPX_TEST_NO_THROW(hpx::supervision::init(hpx::launch::sync));

    // Let a sweep actually start against the (still-silent) peer before
    // tearing down, so finalize() races a real in-flight await_terminal.
    hpx::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto const start = std::chrono::steady_clock::now();
    hpx::supervision::finalize();
    auto const elapsed = std::chrono::steady_clock::now() - start;

    // Bounded by poll_interval slicing (~200ms), not by
    // default_discovery_timeout (60s). 2s gives ample margin for scheduler
    // jitter while still clearly failing if finalize() blocked on the full
    // timeout.
    HPX_TEST(elapsed < std::chrono::seconds(2));

    // Restore the short timeout for any tests still to run.
    hpx::supervision::testing::set_failure_detection_poll_timeout_for_testing(
        test_poll_timeout);
}

// Scenario 4: idempotency of the fence.
//
// Once fenced by detection, repeated dispatches must keep failing with the same
// target_fenced error, and the shadow's epoch must not churn between calls --
// the detector must not re-fence (re-bump the epoch) on every subsequent sweep
// once a shadow is already fenced.
void test_fence_is_idempotent(hpx::id_type const& peer_locality)
{
    auto const shadow = find_shadow_for(peer_locality);
    HPX_TEST(shadow.has_value());
    if (!shadow)
    {
        return;
    }

    auto const initial_state = hpx::supervision::query_state(*shadow);
    std::uint64_t const epoch = initial_state.epoch;

    HPX_TEST(wait_until_fenced(*shadow, epoch, std::chrono::seconds(5)));

    std::uint64_t const fenced_epoch =
        hpx::supervision::query_state(*shadow).epoch;

    constexpr int repeats = 3;
    for (int i = 0; i != repeats; ++i)
    {
        bool caught = false;
        try
        {
            hpx::supervision::dispatch_work<probe_action>(*shadow, epoch).get();
        }
        catch (hpx::exception const& e)
        {
            caught = true;
            HPX_TEST(hpx::get_error(e) == hpx::error::target_fenced);
        }
        HPX_TEST(caught);

        // No epoch churn: a second/third sweep observing an already-fenced
        // shadow must not bump it again.
        HPX_TEST_EQ(hpx::supervision::query_state(*shadow).epoch, fenced_epoch);

        hpx::this_thread::sleep_for(test_poll_timeout);
    }
}

// ============================================================================
// Main Test Entry Point (SPMD across 2 localities)
// ============================================================================

int hpx_main()
{
    std::vector<hpx::id_type> const remote_localities =
        hpx::find_remote_localities();
    HPX_TEST(!remote_localities.empty());
    if (remote_localities.empty())
    {
        return hpx::finalize();
    }

    hpx::id_type const& peer_locality = remote_localities.front();

    bool const is_observer = (hpx::get_locality_id() == 0);

    // Shrink the poll timeout before init() starts the background loop, on
    // both sides -- the peer side needs it too for scenarios 2/4's repeated
    // sweep timing to be meaningful.
    hpx::supervision::testing::set_failure_detection_poll_timeout_for_testing(
        test_poll_timeout);

    HPX_TEST_NO_THROW(hpx::supervision::init(hpx::launch::sync));
    hpx::distributed::barrier::synchronize();

    if (is_observer)
    {
        // --- Scenario 2 first: peer is still fully live at this point. ---
        test_no_false_positives(peer_locality);
    }
    hpx::distributed::barrier::synchronize();

    if (!is_observer)
    {
        // The peer now goes silent for the remainder of the test: no more
        // events are published from here on, simulating a hard crash with
        // no terminal callback ever firing.
        hpx::this_thread::sleep_for(peer_idle_bound);
    }
    else
    {
        // --- Scenario 1: peer is now silent; wait for detection. ---
        test_detection_causes_fencing(peer_locality);

        // --- Scenario 4: fence must be sticky, no epoch churn. ---
        test_fence_is_idempotent(peer_locality);
    }
    hpx::distributed::barrier::synchronize();

    if (is_observer)
    {
        hpx::supervision::finalize();
    }
    hpx::distributed::barrier::synchronize();

    // --- Scenario 3: clean shutdown while a sweep may still be in flight. Run
    // on the observer only, against its own re-armed lifecycle -- the peer side
    // has already gone (and stays) silent, so this exercises finalize() racing
    // a real unresponsive-peer sweep end-to-end.
    if (is_observer)
    {
        test_clean_shutdown_during_in_flight_sweep(peer_locality);
    }
    else
    {
        hpx::supervision::finalize();
    }

    return hpx::finalize();
}

int main(int const argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}

#else

int main(int, char*[])
{
    return 0;
}

#endif
