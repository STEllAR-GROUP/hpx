//  Copyright (c) 2013 Maciej Brodowicz
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/modules/testing.hpp>

#include <cstdint>
#include <iostream>

///////////////////////////////////////////////////////////////////////////////
inline bool close_enough(double m, double ex, double perc)
{
    return 100.0*fabs(m-ex)/ex <= perc;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map&)
{
    const size_t n = 1000000;

    std::uint32_t const prefix = hpx::get_locality_id();
    // use floating point instructions here to avoid measuring runtime side effects
    using hpx::naming::id_type;
    using hpx::performance_counters::get_counter;
    using hpx::performance_counters::performance_counter;
    using hpx::performance_counters::counter_value;

    performance_counter counter(hpx::util::format(
        "/papi{{locality#{}/worker-thread#0}}/PAPI_FP_INS", prefix));

    counter.start(hpx::launch::sync);

    // perform n ops, active counter
    volatile size_t i;
    volatile double a = 0.0, b = 0.0, c = 0.0;
    for (i = 0; i < n; i++) a=b+c;
    (void) a;

    counter_value value1 = counter.get_counter_value(hpx::launch::sync);

    // stop the counter w/o resetting
    counter.stop(hpx::launch::sync);

    // perform n ops (should be uncounted)
    for (i = 0; i < n; i++) a=b+c;
    (void) a;

    // get value and reset, and start again
    counter_value value2 = counter.get_counter_value(hpx::launch::sync, true);
    counter.start(hpx::launch::sync);

    // perform 2*n ops, counted from 0 (or close to it)
    for (i = 0; i < 2*n; i++) a=b+c;
    (void) a;
    counter_value value3 = counter.get_counter_value(hpx::launch::sync);

    // reset counter using reset-only interface
    counter.reset(hpx::launch::sync);

    // perform n ops, counted from 0 (or close to it)
    for (i = 0; i < n; i++) a=b+c;
    (void) a;

    counter_value value4 = counter.get_counter_value(hpx::launch::sync);

    bool pass = status_is_valid(value1.status_) &&
                status_is_valid(value2.status_) &&
                status_is_valid(value3.status_) &&
                status_is_valid(value4.status_);
    if (pass)
    {
        std::uint64_t cnt1 = value1.get_value<std::uint64_t>();
        std::uint64_t cnt2 = value2.get_value<std::uint64_t>();
        std::uint64_t cnt3 = value3.get_value<std::uint64_t>();
        std::uint64_t cnt4 = value4.get_value<std::uint64_t>();

        std::cout << n << " counted fp instructions, result: " << cnt1 << std::endl
                  << n << " uncounted fp instructions, result: "
                  << cnt2-cnt1 << std::endl
                  << 2*n << " fp instructions, count after reset: " << cnt3 << std::endl
          << n << " fp instructions, count after explicit reset: " << cnt4 << std::endl;

        pass = pass && close_enough(cnt1, n, 1.0) &&
           (cnt2 >= cnt1) && close_enough(cnt1, cnt2, 1.0) &&
           close_enough(cnt3, 2.0*cnt1, 1.0) &&
           close_enough(cnt4, cnt1, 1.0);
    }
    HPX_TEST(pass);
    std::cout << (pass? "PASSED": "FAILED") << ".\n";

    hpx::finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    hpx::program_options::options_description cmdline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX.
    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return hpx::init(argc, argv, init_args);
}
