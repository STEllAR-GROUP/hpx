//  Copyright (c) 2026 Rohan Pattanayak
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Regression check for #7825: finalize() called while a failure detection
// sweep is still waiting on its peers, immediately followed by runtime
// shutdown, must exit cleanly. The sweep's await_terminal() waiters (bounded
// by the default 60s poll timeout, against a peer that stays alive and so
// never publishes a terminal event) are still pending when the runtime shuts
// down. They used to be broken only by ~supervision_manager, after the runtime
// had stopped, and their hpx::launch::async continuations then aborted the
// process trying to register a thread.
//
// Deliberately isolated from failure_detection.cpp, whose scenarios depend on
// each other's state (and on a shortened poll timeout), while this needs the
// default poll timeout and shuts down right after finalize().

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/hpx_init.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/runtime_distributed.hpp>
#include <hpx/modules/supervision.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/threading.hpp>

#include <hpx/supervision_dispatch.hpp>

#include <chrono>
#include <vector>

// ============================================================================
// Test Cases: finalize() with a sweep in flight
// ============================================================================

// The detection loop must have a sweep in flight, i.e. await_terminal()
// waiters pending against the (live) peer, when finalize() is called, and
// finalize() must still return promptly.
void test_finalize_with_pending_sweep(
    hpx::supervision::registry const& local_registry)
{
    // Both localities init() concurrently, so the peer may not have been
    // registered yet when this locality ran its own discovery; join again
    // until a sweep against it is in flight.
    auto const deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    bool sweep_started = false;
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (hpx::supervision::testing::
                failure_detection_sweep_in_flight_for_testing())
        {
            sweep_started = true;
            break;
        }
        hpx::supervision::discover_and_join(
            local_registry, std::chrono::milliseconds(500));
        hpx::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    HPX_TEST(sweep_started);

    auto const start = std::chrono::steady_clock::now();
    hpx::supervision::finalize();
    HPX_TEST(
        std::chrono::steady_clock::now() - start < std::chrono::seconds(2));
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

    bool const is_observer = (hpx::get_locality_id() == 0);

    // Keep the default poll timeout (60s): the waiters must outlive the rest
    // of this test, so that only shutdown can resolve them.
    hpx::supervision::registry local_registry;
    HPX_TEST_NO_THROW(
        local_registry = hpx::supervision::init(hpx::launch::sync));
    hpx::distributed::barrier::synchronize();

    if (is_observer)
    {
        test_finalize_with_pending_sweep(local_registry);
    }

    // The peer stays initialized (and alive) until the observer is done, then
    // both shut down right away, as in #7825 (finalize() is a no-op on the
    // observer, which already finalized above).
    hpx::distributed::barrier::synchronize();
    hpx::supervision::finalize();

    return hpx::finalize();
}

int main(int const argc, char* argv[])
{
    int const result = hpx::init(argc, argv);

    // non-console localities may see the runtime being shutdown before they
    // call finalize, causing an error to be reported that can be ignored.
    HPX_TEST(result == 0 || result == -1);

    return hpx::util::report_errors();
}

#else

int main(int, char*[])
{
    return 0;
}

#endif
