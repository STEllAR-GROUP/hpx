//  Copyright (c) 2011 Bryce Lelbach and Daniel Kogler 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <vector>

#include <boost/format.hpp>

#include <hpx/hpx_init.hpp>

#include <examples/barnes_hut/barnes_hut/bhcontroller.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;

using hpx::components::bhcontroller;

using hpx::naming::id_type;

using hpx::applier::get_applier;


///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    int test;
    std::string inputFile;

    inputFile = vm["input"].as<std::string>();
    {
        id_type prefix = get_applier().get_runtime_support_gid();

        bhcontroller dat;

        dat.create(prefix);
        test = dat.construct(inputFile);
        if(test == 0){
            dat.run_simulation();
        }
        dat.free();

//        std::cout << (boost::format("total error   : %1%\n"
//                                    "average error : %2%\n")
//                     % r % (r / size));  
    }

    finalize();
    std::cout<<"OH YEAH!\n";
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "input,i"
        , value<std::string>()->default_value("5file") 
        , "file name of the input file");

    // Initialize and run HPX.
    return init(desc_commandline, argc, argv);
}

