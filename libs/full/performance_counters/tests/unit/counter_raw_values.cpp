//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// The atomic variable 'counter' ensures the thread safety of the counter.
std::atomic<std::int64_t> counter(0);

std::vector<std::int64_t> get_values(bool reset)
{
    std::vector<std::int64_t> result(10);
    std::iota(result.begin(), result.end(), counter.load());

    ++counter;
    if (reset)
        counter.store(0);

    return result;
}

void register_counter_type()
{
    // Call the HPX API function to register the counter type.
    hpx::performance_counters::install_counter_type(
        // counter type name
        "/test/values",
        // function providing counter data
        &get_values,
        // description text
        "returns an array of linearly increasing counter values");
}

int hpx_main()
{
    for (int i = 0; i != 10; ++i)
    {
        hpx::performance_counters::performance_counter c("/test/values");

        // simple counters don't support starting/stopping
        HPX_TEST(!c.start(hpx::launch::sync));

        auto values = c.get_counter_values_array(hpx::launch::sync, false);

        HPX_TEST_EQ(values.count_, static_cast<std::uint64_t>(i + 1));

        std::vector<std::int64_t> expected(10);
        std::iota(expected.begin(), expected.end(), i);
        HPX_TEST(values.values_ == expected);

        std::string name = c.get_name(hpx::launch::sync);
        HPX_TEST_EQ(name, std::string("/test{locality#0/total}/values"));

        // simple counters don't support starting/stopping
        HPX_TEST(!c.stop(hpx::launch::sync));
    }
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    hpx::register_startup_function(&register_counter_type);

    // Initialize and run HPX.
    std::vector<std::string> const cfg = {"hpx.os_threads=1"};
    hpx::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);

    return hpx::util::report_errors();
}
#endif
