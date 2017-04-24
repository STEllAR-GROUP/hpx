//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Demonstrating #1890: Invoking papi counters give segfault

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstdint>

int hpx_main(int argc, char ** argv)
{
#if defined(HPX_HAVE_PAPI)
    using hpx::performance_counters::performance_counter;

    // Create and start the counters
    performance_counter total_cycles(
        "/arithmetics/add@/papi{locality#0/worker-thread#*}/PAPI_TOT_CYC");
    total_cycles.start(hpx::launch::sync);

    performance_counter cycles(
        "/papi{locality#0/worker-thread#0}/PAPI_TOT_CYC");
    cycles.start(hpx::launch::sync);

    std::int64_t val1 = total_cycles.get_value<std::int64_t>(hpx::launch::sync);
    std::int64_t val2 = cycles.get_value<std::int64_t>(hpx::launch::sync);

    HPX_TEST(val1 != 0);
    HPX_TEST(val2 != 0);
#endif

    return hpx::finalize();
}

int main(int argc, char **argv)
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
