//  Copyright (c) 2014 Hartmut Kaiser
//  Copyright (c) 2014 Patricia Grubel
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/util/format.hpp>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

///////////////////////////////////////////////////////////////////////////////
void print_time_results(std::uint32_t num_localities,
    std::uint64_t num_os_threads, std::uint64_t elapsed, std::uint64_t nx,
    std::uint64_t np, std::uint64_t nt, bool header)
{
    if (header)
        std::cout << "Localities,OS_Threads,Execution_Time_sec,"
                     "Points_per_Partition,Partitions,Time_Steps\n"
                  << std::flush;

    std::string const locs_str = hpx::util::format("{},", num_localities);
    std::string const threads_str = hpx::util::format("{},", num_os_threads);
    std::string const nx_str = hpx::util::format("{},", nx);
    std::string const np_str = hpx::util::format("{},", np);
    std::string const nt_str = hpx::util::format("{} ", nt);

    hpx::util::format_to(std::cout,
        "{:-6} {:-6} {:.14g}, {:-21} {:-21} {:-21}\n", locs_str, threads_str,
        elapsed / 1e9, nx_str, np_str, nt_str)
        << std::flush;
}

///////////////////////////////////////////////////////////////////////////////
void print_time_results(std::uint64_t num_os_threads, std::uint64_t elapsed,
    std::uint64_t nx, std::uint64_t np, std::uint64_t nt, bool header)
{
    if (header)
        std::cout << "OS_Threads,Execution_Time_sec,"
                     "Points_per_Partition,Partitions,Time_Steps\n"
                  << std::flush;

    std::string const threads_str = hpx::util::format("{},", num_os_threads);
    std::string const nx_str = hpx::util::format("{},", nx);
    std::string const np_str = hpx::util::format("{},", np);
    std::string const nt_str = hpx::util::format("{} ", nt);

    hpx::util::format_to(std::cout, "{:-21} {:.14g}, {:-21} {:-21} {:-21}\n",
        threads_str, elapsed / 1e9, nx_str, np_str, nt_str)
        << std::flush;
}

void print_time_results(std::uint64_t num_os_threads, std::uint64_t elapsed,
    std::uint64_t nx, std::uint64_t nt, bool header)
{
    if (header)
        std::cout << "OS_Threads,Execution_Time_sec,"
                     "Grid_Points,Time_Steps\n"
                  << std::flush;

    std::string const threads_str = hpx::util::format("{},", num_os_threads);
    std::string const nx_str = hpx::util::format("{},", nx);
    std::string const nt_str = hpx::util::format("{} ", nt);

    hpx::util::format_to(std::cout, "{:-21} {:10.12}, {:-21} {:-21}\n",
        threads_str, elapsed / 1e9, nx_str, nt_str)
        << std::flush;
}
