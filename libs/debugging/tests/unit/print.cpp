//  Copyright (c) 2020 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The purpose of this test is to ensure that when enable_print is false
// that function arguments and other parameters used in the print statements
// are completely elided

#include <hpx/debugging/print.hpp>
#include <hpx/threading_base/print.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/modules/testing.hpp>

#include <string>
#include <chrono>

namespace hpx {
    // this is an enabled debug object that should output messages
    static hpx::debug::enable_print<true>  p_enabled("PRINT  ");
    // this is disabled and we want it to have zero footprint
    static hpx::debug::enable_print<false> p_disabled("PRINT  ");
}   // namespace hpx

int increment(std::atomic<int> &counter) {
    return ++counter;
}

#define AB_LAZY(expr, debug) \
    if (debug.is_enabled()) { return expr; }

int main()
{
    // some counters we will use for checking if anything happens or not
    std::atomic<int> enabled_counter(0);
    std::atomic<int> disabled_counter(0);

    using namespace hpx;
    using namespace hpx::debug;

    // ---------------------------------------------------------
    // Test if normal debug messages trigger argument evaluation
    // use the DP_LAZY macro to prevent evaluation when disabled
    // we expect the counter to increment
    p_enabled.debug("Increment", increment(enabled_counter));
    HPX_TEST_EQ(enabled_counter, 1);

    // we expect the counter to increment as LAZY will be evaluated
    p_enabled.debug("Increment", DP_LAZY(increment(enabled_counter), p_enabled));
    HPX_TEST_EQ(enabled_counter, 2);

    // we do not expect the counter to increment
    if (p_disabled.is_enabled()) {
        p_disabled.debug("Increment", increment(disabled_counter));
    }
    HPX_TEST_EQ(disabled_counter, 0);

    // we do not expect the counter to increment: DP_LAZY will not be evaluated
    p_disabled.debug("Increment", DP_LAZY(increment(disabled_counter), p_disabled));
    HPX_TEST_EQ(disabled_counter, 0);


    // ---------------------------------------------------------
    // Test that scoped log messages behave as expected
    {
        auto s_enabled  = p_enabled.scope("scoped block", DP_LAZY(increment(enabled_counter), p_enabled));
        auto s_disabled = p_disabled.scope("scoped block", DP_LAZY(increment(disabled_counter), p_disabled));
    }
    HPX_TEST_EQ(enabled_counter, 3);
    HPX_TEST_EQ(disabled_counter, 0);


    // ---------------------------------------------------------
    // Test that debug only variables behave as expected
    // create high resolution timers to see if they count
    auto var1 = p_enabled.declare_variable<int>(DP_LAZY(enabled_counter+4, p_enabled));
    (void)var1; // silenced unused var when optimized out

    auto var2 = p_disabled.declare_variable<int>(DP_LAZY(disabled_counter+10, p_disabled));
    (void)var2; // silenced unused var when optimized out

    p_enabled.debug("var 1"
        , hpx::debug::dec<>(DP_LAZY(enabled_counter+=var1, p_enabled)));
    p_disabled.debug("var 2"
        , hpx::debug::dec<>(DP_LAZY(disabled_counter+=var2, p_disabled)));

    HPX_TEST_EQ(enabled_counter, 10);
    HPX_TEST_EQ(disabled_counter, 0);

    // ---------------------------------------------------------
    // Test that timed log messages behave as expected
    static auto t_enabled =
        p_enabled.make_timer(1, debug::str<>("Timed (enabled)"));
    static auto t_disabled =
        p_disabled.make_timer(1, debug::str<>("Timed (disabled)"));

    // run a loop for 2 seconds with a timed print every 1 sec
    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(end - start).count() < 2)
    {
        p_enabled.timed(t_enabled,
            "enabled", debug::dec<3>(DP_LAZY(++enabled_counter, p_enabled)));

        p_disabled.timed(t_disabled,
            "disabled", debug::dec<3>(DP_LAZY(++disabled_counter, p_disabled)));
        end = std::chrono::system_clock::now();
    }
    HPX_TEST_EQ(enabled_counter>10, true);
    HPX_TEST_EQ(disabled_counter, 0);

    std::cout << "enabled  counter " << enabled_counter << std::endl;
    std::cout << "disabled counter " << disabled_counter << std::endl;

    return hpx::util::report_errors();
}

