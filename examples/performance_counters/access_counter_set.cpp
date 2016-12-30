//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The purpose of this example is to demonstrate the simplest way to create and
// use a performance counter for HPX.

#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/performance_counters.hpp>

#include <boost/program_options.hpp>

#include <cstddef>
#include <iomanip>
#include <string>
#include <vector>

int hpx_main(boost::program_options::variables_map& vm)
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
    using namespace hpx::performance_counters;
    performance_counter_set set(name);

    // retrieve the counter information for all attached counters
    std::vector<counter_info> infos = set.get_counter_infos();

    // retrieve the current counter values
    std::vector<double> values = set.get_values<double>(hpx::launch::sync);

    // print the values for all 'raw' (non-histogram) counters
    for (std::size_t i = 0, j = 0; i != infos.size(); ++i)
    {
        if (infos[i].type_ != counter_raw)
            continue;

        if (infos[i].unit_of_measure_.empty())
        {
            hpx::cout << infos[i].fullname_ << ":" << values[j] << std::endl;
        }
        else
        {
            hpx::cout
                << infos[i].fullname_ << ":" << values[j] << "["
                << infos[i].unit_of_measure_ << "]" << std::endl;
        }
        ++j;
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Define application-specific command-line options.
    boost::program_options::options_description opts(
        "usage: access_counter_set [options]");

    opts.add_options()
        ("counter", boost::program_options::value<std::string>(),
         "name (pattern) representing the of the performance counter(s) to query")
        ;

    // Initialize and run HPX.
    return hpx::init(opts, argc, argv);
}

