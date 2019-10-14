//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/collectives.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

std::size_t iterations = 100;
double startup_end = 0.0;
double shutdown_start = 0.0;

void global_barrier()
{
    hpx::lcos::barrier b("new_global_barrier");

    hpx::util::high_resolution_timer t;
    for (std::size_t i = 0; i != iterations; ++i)
    {
        b.wait();
    }
    double elapsed = t.elapsed();

    if (hpx::get_locality_id() == 0)
    {
        std::cout << "Barrier: " << elapsed / iterations << " (seconds)\n";
    }
}

int hpx_main()
{
    if (hpx::get_locality_id() == 0)
        startup_end = hpx::util::high_resolution_timer::now();
    global_barrier();

    if (hpx::get_locality_id() == 0)
        shutdown_start = hpx::util::high_resolution_timer::now();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg = {
        "hpx.run_hpx_main!=1", "hpx.os_threads!=all"};

    double startup_start = hpx::util::high_resolution_timer::now();
    return hpx::init(argc, argv, cfg);
    double shutdown_end = hpx::util::high_resolution_timer::now();

    if (startup_end > 1e-10)
    {
        double elapsed = startup_end - startup_start;
        std::cout << "Startup: " << elapsed << " (seconds)\n";
    }

    if (shutdown_start > 1e-10)
    {
        double elapsed = shutdown_end - shutdown_start;
        std::cout << "Shutdown: " << elapsed << " (seconds)\n";
    }
}
