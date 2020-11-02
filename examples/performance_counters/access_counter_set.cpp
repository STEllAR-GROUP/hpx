//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The purpose of this example is to demonstrate the simplest way to create and
// use a performance counter for HPX.

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/iostream.hpp>

#include <hpx/modules/program_options.hpp>

#include <cstddef>
#include <iomanip>
#include <string>
#include <vector>

int hpx_main(hpx::program_options::variables_map& vm)
{
    // extract the counter name (pattern) from command line
    std::string name;
    if (vm.count("counter"))
    {
        name = vm["counter"].as<std::string>();
    }
    else
    {
        name = "/threads{locality#*/worker-thread#*}/count/cumulative";
    }

    // create the performance counter set from the given pattern (name)
    {
        using namespace hpx::performance_counters;
        performance_counter_set set(name);

        // retrieve the counter information for all attached counters
        std::vector<counter_info> infos = set.get_counter_infos();

        // retrieve the current counter values
        std::vector<double> values = set.get_values<double>(hpx::launch::sync);

        // print the values for all 'raw' (non-histogram) counters
        for (std::size_t i = 0, j = 0; i != infos.size(); ++i)
        {
            if (infos[i].type_ != counter_raw &&
                infos[i].type_ != counter_monotonically_increasing &&
                infos[i].type_ != counter_aggregating &&
                infos[i].type_ != counter_elapsed_time &&
                infos[i].type_ != counter_average_count &&
                infos[i].type_ != counter_average_timer)
            {
                continue;
            }

            if (infos[i].unit_of_measure_.empty())
            {
                hpx::cout << infos[i].fullname_ << ":" << values[j]
                          << std::endl;
            }
            else
            {
                hpx::cout << infos[i].fullname_ << ":" << values[j] << "["
                          << infos[i].unit_of_measure_ << "]" << std::endl;
            }
            ++j;
        }
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Define application-specific command-line options.
    hpx::program_options::options_description cmdline(
        "usage: access_counter_set [options]");

    cmdline.add_options()("counter", hpx::program_options::value<std::string>(),
        "name (pattern) representing the of the performance counter(s) to "
        "query");

    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;

    // Initialize and run HPX.
    return hpx::init(argc, argv, init_args);
}

#endif
