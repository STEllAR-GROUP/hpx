//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>

#include <boost/program_options.hpp>

#include <string>
#include <vector>

int hpx_main(boost::program_options::variables_map& vm)
{
    // extract value of application specific command line option
    int test = vm["test"].as<int>();
    hpx::cout
        << "value for command line option --test: "
        << test << "\n";

    // extract all positional command line argument
    if (vm.count("hpx:positional"))
    {
        std::vector<std::string> positional =
            vm["hpx:positional"].as<std::vector<std::string> >();
        hpx::cout << "positional command line options:\n";
        for (std::string const& arg : positional)
            hpx::cout << arg << "\n";
    }
    else
    {
        hpx::cout << "no positional command line options\n";
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Configure application-specific options.
    boost::program_options::options_description desc_commandline;

    desc_commandline.add_options()
        ("test",
         boost::program_options::value<int>()->default_value(42),
         "additional, application-specific option")
    ;

    return hpx::init(desc_commandline, argc, argv);
}
