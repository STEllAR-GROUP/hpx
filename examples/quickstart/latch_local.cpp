//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Demonstrate the use of hpx::lcos::local::latch

#include <hpx/hpx_init.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>

///////////////////////////////////////////////////////////////////////////////
std::ptrdiff_t num_threads = 16;

///////////////////////////////////////////////////////////////////////////////
void wait_for_latch(hpx::lcos::local::latch& l)
{
    l.count_down_and_wait();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    num_threads = vm["num-threads"].as<std::ptrdiff_t>();

    hpx::lcos::local::latch l(num_threads+1);

    std::vector<hpx::future<void> > results;
    for (std::ptrdiff_t i = 0; i != num_threads; ++i)
        results.push_back(hpx::async(&wait_for_latch, std::ref(l)));

    // Wait for all threads to reach this point.
    l.count_down_and_wait();

    hpx::wait_all(results);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using boost::program_options::options_description;
    using boost::program_options::value;

    // Configure application-specific options
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "num-threads,n", value<std::ptrdiff_t>()->default_value(16),
          "number of threads to synchronize at a local latch (default: 16)")
        ;

    return hpx::init(desc_commandline, argc, argv);
}
