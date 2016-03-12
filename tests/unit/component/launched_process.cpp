//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <boost/program_options.hpp>

int hpx_main(boost::program_options::variables_map& vm)
{
    // extract command line arguments
    int exit_code = 0;
    if (vm.count("exit_code") != 0)
        exit_code = vm["exit_code"].as<int>();

    hpx::finalize();
    return exit_code;
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: launched_process_test [options]");

    desc_commandline.add_options()
        ("exit_code,e", value<int>(), "the value to return to the OS")
        ;

    return hpx::init(desc_commandline, argc, argv);
}

