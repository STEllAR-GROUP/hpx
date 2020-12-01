//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Demonstrating #1508: memory and papi counters do not work

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/modules/testing.hpp>

int hpx_main()
{
    using hpx::performance_counters::performance_counter;

    bool counter_created = false;
    bool value_retrieved = false;

    try
    {
        performance_counter memory("/runtime/memory/resident");
        counter_created = true;

        using hpx::performance_counters::counter_value;
        using hpx::performance_counters::status_is_valid;

        counter_value value = memory.get_counter_value(hpx::launch::sync);
        HPX_TEST(status_is_valid(value.status_));

        double val = value.get_value<double>();
        HPX_TEST_LT(0.0, val);

        value_retrieved = true;
    }
    catch (hpx::exception const&)
    {
        HPX_TEST(false);
    }

    HPX_TEST(counter_created);
    HPX_TEST(value_retrieved);

    return hpx::finalize();
}

int main(int argc, char** argv)
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
#endif
