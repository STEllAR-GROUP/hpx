//  Copyright (c) 2007-2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/util/parse_command_line.hpp>

#include <boost/program_options.hpp>

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();    // create entry point for component factory

///////////////////////////////////////////////////////////////////////////////
namespace startup_shutdown
{
    // This function will be registered as a startup function for HPX below.
    //
    // That means it will be executed in a HPX-thread before hpx_main, but
    // after the runtime has been initialized and started.
    void startup()
    {
        using boost::program_options::options_description;
        using boost::program_options::variables_map;

        options_description desc_commandline("startup_shutdown_component");
        desc_commandline.add_options()
            ("additional",
             "this is an additional option for the startup_shutdown_compnent")
            ;

        // Retrieve command line using the Boost.ProgramOptions library.
        variables_map vm;
        if (!hpx::util::retrieve_commandline_arguments(desc_commandline, vm)) {
            HPX_THROW_EXCEPTION(hpx::not_implemented,
                "startup_shutdown::startup",
                "Failed to handle command line options");
        }

        if (vm.count("additional")) {
            std::cout << "Found additional option on command line!"
                      << std::endl;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Register the startup function which will be called as a HPX-thread during
// runtime startup.
//
// Note that this macro can be used not more than once in one module. Either
// of the 2 parameters for the macro below can be zero (0), which means no
// function will be called. We leave the shutdown function out as we don't need
// it for the purpose of this example.
HPX_REGISTER_STARTUP_SHUTDOWN_MODULE(::startup_shutdown::startup, (void(*)())0);

