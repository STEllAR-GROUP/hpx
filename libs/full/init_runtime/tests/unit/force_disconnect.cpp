//  Copyright (c) 2026 The STE||AR-Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test verifies the guard clauses of hpx::force_disconnect (calling from a
// non-HPX thread, from a non-console locality, or trying to disconnect the
// console locality itself all fail with hpx::error::invalid_status, both
// through the error_code overload and by throwing), as well as the happy path,
// where disconnecting a valid remote locality succeeds and actually removes it
// from AGAS/the connection caches while leaving the remaining localities
// unaffected. It further verifies that a second, idempotent, force_disconnect
// call on the already-removed locality does not fail or hang, and that
// hpx::finalize() still completes promptly afterwards.

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/init.hpp>
#include <hpx/modules/actions.hpp>
#include <hpx/modules/agas.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/modules/components_base.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/naming_base.hpp>
#include <hpx/modules/runtime_distributed.hpp>
#include <hpx/modules/runtime_local.hpp>
#include <hpx/modules/testing.hpp>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// Calling hpx::force_disconnect from a thread that is not an HPX-thread should
// fail with hpx::error::invalid_status, regardless of the given locality, since
// this check is performed before any locality validation.
void test_non_hpx_thread_guard()
{
    hpx::id_type const target = hpx::find_here();

    // ec-based overload
    {
        int retval = 0;
        hpx::error_code ec(hpx::throwmode::lightweight);

        std::thread t([&]() { retval = hpx::force_disconnect(target, ec); });
        t.join();

        HPX_TEST_EQ(retval, -1);
        HPX_TEST(ec);
        HPX_TEST_EQ(ec.value(), static_cast<int>(hpx::error::invalid_status));
    }

    // throwing overload
    {
        bool caught_exception = false;
        hpx::error thrown_error = hpx::error::success;

        std::thread t([&]() {
            try
            {
                hpx::force_disconnect(target);
                HPX_TEST(false);
            }
            catch (hpx::exception const& e)
            {
                caught_exception = true;
                thrown_error = e.get_error();
            }
        });
        t.join();

        HPX_TEST(caught_exception);
        HPX_TEST_EQ(thrown_error, hpx::error::invalid_status);
    }
}

// Non-console localities are not allowed to call hpx::force_disconnect; the
// console locality is the only one permitted to disconnect other localities.
void test_calling_from_non_console_locality(std::uint32_t const locality_id)
{
    if (locality_id != 0)
    {
        hpx::id_type const target = hpx::naming::get_id_from_locality_id(0);

        // error_code overload
        {
            hpx::error_code ec(hpx::throwmode::lightweight);
            int const result = hpx::force_disconnect(target, ec);
            HPX_TEST_EQ(result, -1);
            HPX_TEST(ec);
            HPX_TEST_EQ(
                ec.value(), static_cast<int>(hpx::error::invalid_status));
            std::string const msg = ec.get_message();
            HPX_TEST(msg.find("should be called on the console") !=
                std::string::npos);
        }

        // throwing overload
        {
            bool caught_exception = false;
            try
            {
                hpx::force_disconnect(target);
                HPX_TEST(false);
            }
            catch (hpx::exception const& e)
            {
                caught_exception = true;
                HPX_TEST(e.get_error() == hpx::error::invalid_status);
                std::string const msg = e.what();
                HPX_TEST(msg.find("should be called on the console") !=
                    std::string::npos);
            }
            HPX_TEST(caught_exception);
        }
    }
}

// Non-console localities are not allowed to call hpx::force_disconnect
// regardless of which locality they target - including another non-console
// locality (as opposed to targeting console itself, which is covered by
// test_calling_from_non_console_locality above). This exercises the same
// guard from the untested branch where target != console.
void test_non_console_targets_non_console(std::uint32_t const locality_id)
{
    if (locality_id == 0)
    {
        return;
    }

    hpx::id_type const here = hpx::find_here();
    std::vector<hpx::id_type> const localities = hpx::find_all_localities();

    hpx::id_type target;
    for (hpx::id_type const& id : localities)
    {
        if (id != here && id != hpx::naming::get_id_from_locality_id(0))
        {
            target = id;
            break;
        }
    }
    HPX_TEST(target);
    if (!target)
    {
        return;
    }

    // error_code overload
    {
        hpx::error_code ec(hpx::throwmode::lightweight);
        int const result = hpx::force_disconnect(target, ec);
        HPX_TEST_EQ(result, -1);
        HPX_TEST(ec);
        HPX_TEST_EQ(ec.value(), static_cast<int>(hpx::error::invalid_status));
        std::string const msg = ec.get_message();
        HPX_TEST(
            msg.find("should be called on the console") != std::string::npos);
    }

    // throwing overload
    {
        bool caught_exception = false;
        try
        {
            hpx::force_disconnect(target);
            HPX_TEST(false);
        }
        catch (hpx::exception const& e)
        {
            caught_exception = true;
            HPX_TEST(e.get_error() == hpx::error::invalid_status);
            std::string const msg = e.what();
            HPX_TEST(msg.find("should be called on the console") !=
                std::string::npos);
        }
        HPX_TEST(caught_exception);
    }
}

// The console locality is not allowed to disconnect itself; calling
// hpx::force_disconnect(hpx::find_here(), ...) from console must fail.
void test_console_cannot_disconnect_itself(std::uint32_t const locality_id)
{
    if (locality_id != 0)
    {
        return;
    }

    hpx::id_type const here = hpx::find_here();

    // error_code overload
    {
        hpx::error_code ec(hpx::throwmode::lightweight);
        int const result = hpx::force_disconnect(here, ec);
        HPX_TEST_EQ(result, -1);
        HPX_TEST(ec);
        HPX_TEST_EQ(ec.value(), static_cast<int>(hpx::error::invalid_status));
        std::string const msg = ec.get_message();
        HPX_TEST(
            msg.find("cannot be used to disconnect the ") != std::string::npos);
    }

    // throwing overload
    {
        bool caught_exception = false;
        try
        {
            hpx::force_disconnect(here);
            HPX_TEST(false);
        }
        catch (hpx::exception const& e)
        {
            caught_exception = true;
            HPX_TEST(e.get_error() == hpx::error::invalid_status);
            std::string const msg = e.what();
            HPX_TEST(msg.find("cannot be used to disconnect the ") !=
                std::string::npos);
        }
        HPX_TEST(caught_exception);
    }
}

// A trivial action used to probe whether a given locality is still reachable,
// i.e. still registered with AGAS and present in the connection caches.
std::uint32_t ping_locality()
{
    return hpx::get_locality_id();
}
HPX_PLAIN_ACTION(ping_locality, ping_locality_action)

// Calling hpx::force_disconnect on a valid, non-console, non-self locality must
// succeed and actually remove that locality from AGAS/the connection caches (so
// that further action dispatches to it fail), while leaving all remaining
// localities, including console itself, fully operational.
//
// Returns the id of the disconnected (now removed) locality, together with the
// id of the still-reachable control locality, so that a follow-up
// double-disconnect test (see below) can reuse the same setup instead of
// re-deriving it.
std::pair<hpx::id_type, hpx::id_type> test_force_disconnect_removes_locality(
    std::uint32_t const locality_id)
{
    if (locality_id != 0)
    {
        return {};
    }

    std::vector<hpx::id_type> const localities = hpx::find_all_localities();
    HPX_TEST(localities.size() >= 3);
    if (localities.size() < 3)
    {
        return {};
    }

    hpx::id_type const here = hpx::find_here();
    hpx::id_type target;
    hpx::id_type control;
    for (hpx::id_type const& id : localities)
    {
        if (id == here)
        {
            continue;
        }
        if (!target)
        {
            target = id;
        }
        else if (!control)
        {
            control = id;
        }
    }
    HPX_TEST(target);
    HPX_TEST(control);

    // Disconnect the target locality; this must succeed.
    hpx::error_code ec(hpx::throwmode::lightweight);
    int const result = hpx::force_disconnect(target, ec);
    HPX_TEST_EQ(result, 0);
    HPX_TEST(!ec);

    // Dispatching an action to the disconnected locality must now fail, proving
    // that it was actually removed from AGAS/the connection caches.
    {
        bool caught_exception = false;
        try
        {
            ping_locality_action act;
            act(target);
            HPX_TEST(false);
        }
        catch (hpx::exception const&)
        {
            caught_exception = true;
        }
        HPX_TEST(caught_exception);
    }

    // Confirm removal directly at the AGAS level, not just inferred from a
    // failed action dispatch (which could also be explained by a stale
    // connection-cache entry rather than actual deregistration).
    {
        hpx::error_code ec2(hpx::throwmode::lightweight);
        hpx::naming::address const addr =
            hpx::agas::resolve(hpx::launch::sync, target, ec2);
        HPX_TEST(ec2 || !addr);
    }

    // A third, uninvolved locality must still respond normally.
    {
        ping_locality_action act;
        HPX_TEST_EQ(
            act(control), hpx::naming::get_locality_id_from_id(control));
    }

    // The console locality itself must remain fully operational.
    HPX_TEST(hpx::agas::is_console());

    return std::make_pair(target, control);
}

// Calling hpx::force_disconnect a second time on a locality that has already
// been removed must not crash or hang. The actual implementation
// (runtime_support::remove_locality -> addressing_service::unregister_locality
// -> locality_namespace::free) silently ignores GIDs that are no longer present
// in the partition table, so the second call is expected to succeed (return 0,
// no error/exception raised) rather than fail - i.e. force_disconnect is
// idempotent for an already-disconnected locality.
void test_double_disconnect_is_idempotent(
    std::uint32_t const locality_id, hpx::id_type const& target)
{
    if (locality_id != 0 || !target)
    {
        return;
    }

    // error_code overload
    {
        hpx::error_code ec(hpx::throwmode::lightweight);
        int const result = hpx::force_disconnect(target, ec);
        HPX_TEST_EQ(result, 0);
        HPX_TEST(!ec);
    }

    // throwing overload
    {
        bool caught_exception = false;
        int result = -1;
        try
        {
            result = hpx::force_disconnect(target);
        }
        catch (hpx::exception const&)
        {
            caught_exception = true;
        }
        HPX_TEST(!caught_exception);
        HPX_TEST_EQ(result, 0);
    }
}

// Disconnecting more than one locality in the same run must work: the second
// disconnect is independent of the first and must not be affected by state
// left over from having already removed a different locality (e.g. stale
// cache entries or partition-table bookkeeping for the earlier target).
void test_sequential_disconnects(
    std::uint32_t const locality_id, hpx::id_type const& second_target)
{
    if (locality_id != 0 || !second_target)
    {
        return;
    }

    hpx::error_code ec(hpx::throwmode::lightweight);
    int const result = hpx::force_disconnect(second_target, ec);
    HPX_TEST_EQ(result, 0);
    HPX_TEST(!ec);

    // Confirm this second locality is also actually gone, not just the first.
    bool caught_exception = false;
    try
    {
        ping_locality_action act;
        act(second_target);
        HPX_TEST(false);
    }
    catch (hpx::exception const&)
    {
        caught_exception = true;
    }
    HPX_TEST(caught_exception);
}

// hpx::finalize() is asynchronous: it merely flags all localities to stop
// scheduling new work and returns almost immediately, regardless of
// whether shutdown actually completes promptly (or hangs, e.g. because of
// a just-disconnected locality). The real end-to-end shutdown duration is
// the time between the point where finalize() is invoked and hpx::init()
// actually returning in main(); since hpx_main() and main() do not share a
// live call stack across the shutdown boundary, the starting timestamp is
// stashed in a variable with static storage duration.
std::chrono::steady_clock::time_point finalize_time;

int hpx_main()
{
    std::uint32_t const locality_id = hpx::get_locality_id();
    std::cout << "locality " << locality_id << " reached hpx_main\n";

    test_non_hpx_thread_guard();
    test_calling_from_non_console_locality(locality_id);
    test_non_console_targets_non_console(locality_id);
    test_console_cannot_disconnect_itself(locality_id);

    auto const removed = test_force_disconnect_removes_locality(locality_id);
    test_double_disconnect_is_idempotent(locality_id, removed.first);
    test_sequential_disconnects(locality_id, removed.second);

    finalize_time = std::chrono::steady_clock::now();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(
        hpx::init(argc, argv), 0, "HPX main exited with non-zero status");

    // Measure the actual end-to-end shutdown duration, i.e. the time it took
    // hpx::init() to return after hpx::finalize() was called above. This is a
    // generous but still meaningful bound (far above normal shutdown time, but
    // far below the ctest timeout for this test), meant to catch shutdown
    // hanging or taking excessively long because of the locality that was
    // disconnected above; the ctest timeout remains the ultimate backstop
    // against an actual hang.
    auto duration = std::chrono::steady_clock::now() - finalize_time;
    auto const shutdown_seconds =
        std::chrono::duration_cast<std::chrono::seconds>(duration);

    std::cout << "shutdown after force_disconnect took "
              << shutdown_seconds.count() << " seconds\n";
    HPX_TEST_LT(shutdown_seconds, 30.0);

    return hpx::util::report_errors();
}

#endif
