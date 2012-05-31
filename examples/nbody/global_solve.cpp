//  Copyright (c) 2007-2011 Hartmut Kaiser, Richard D Guidry Jr.
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//  Copyright (c)      2012 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <string>

#include "nbody/global_solve.hpp"

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    {
        int rval;
        //create a global_solve component
        examples::global_solve solver;
        solver.create(hpx::find_here());

        rval = solver.init(vm["input-file"].as<std::string>());

        if(rval == 0){
            printf("in main before run\n");
            solver.run(vm["histories"].as<int>(), 
                vm["body-chunk"].as<int>(), vm["time-chunk"].as<int>());
            solver.report(vm["output-directory"].as<std::string>());
        }
        else std::cerr<<"ERROR: input file failed to open\n";
    }

    // Initiate shutdown of the runtime systems on all localities.
    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "input-file,i",
           boost::program_options::value<std::string>()
                ->default_value(std::string("test0")),
                "name of input data file")
        ( "output-directory,o",
           boost::program_options::value<std::string>()
                ->default_value(std::string("output")),
                "name of directory where results will be stored")
        ( "histories,H",
           boost::program_options::value<int>()
                ->default_value(10),
                "number of histories generated")
        ( "body-chunk,B",
           boost::program_options::value<int>()
                ->default_value(10),
          "the maximum number of bodies a future can perform calculations for")
        ( "time-chunk,T",
           boost::program_options::value<int>()
                ->default_value(500),
                "the maximum number of time-steps computed per future")
        ;

    // Initialize and run HPX.
    return hpx::init(desc_commandline, argc, argv);
}

