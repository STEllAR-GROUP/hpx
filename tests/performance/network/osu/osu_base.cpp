//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Bidirectional network bandwidth test

#include <hpx/hpx_init.hpp>

#include <cstddef>

void print_header();
void run_benchmark(boost::program_options::variables_map & vm);

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map & vm)
{
    print_header();
    run_benchmark(vm);
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    boost::program_options::options_description
        desc("Usage: " HPX_APPLICATION_STRING " [options]");

    desc.add_options()
        ("window-size",
         boost::program_options::value<std::size_t>()->default_value(1),
         "Number of messages to send in parallel")
        ("loop",
         boost::program_options::value<std::size_t>()->default_value(100),
         "Number of loops")
        ("min-size",
         boost::program_options::value<std::size_t>()->default_value(1),
         "Minimum size of message to send")
        ("max-size",
         boost::program_options::value<std::size_t>()->default_value((1<<22)),
         "Maximum size of message to send");

    return hpx::init(desc, argc, argv);
}
