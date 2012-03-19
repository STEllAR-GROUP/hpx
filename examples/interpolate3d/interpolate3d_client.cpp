//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>

#include "interpolate3d/interpolate3d.hpp"

using boost::program_options::variables_map;
using boost::program_options::options_description;

///////////////////////////////////////////////////////////////////////////////
inline void
eval(char const* expr, interpolate3d::interpolate3d& gauss, double value_x,
    double value_y, double value_z)
{
    std::cout << expr << gauss.interpolate(value_x, value_y, value_z)
              << " (expected: "
              << std::exp(-value_x*value_x - value_y*value_y - value_z*value_z)
              << ")" << std::endl;
}

int hpx_main(variables_map& vm)
{
    std::string datafilename("gauss.h5");
    int num_localities = 27;

    {
        // create the distributed interpolation object on num_localities
        interpolate3d::interpolate3d gauss;
        gauss.create(datafilename, "/interpolate3d_client/gauss", num_localities);

        // use it to calculate some values
        eval("gauss(0, 0, 0) == ", gauss, 0, 0, 0);
        eval("gauss(1, 0, 0) == ", gauss, 1, 0, 0);
        eval("gauss(0, 1, 0) == ", gauss, 0, 1, 0);
        eval("gauss(0, 0, 1) == ", gauss, 0, 0, 1);
        eval("gauss(1, 0, 1) == ", gauss, 1, 0, 1);
        eval("gauss(1, 1, 0) == ", gauss, 1, 1, 0);
        eval("gauss(0, 1, 1) == ", gauss, 0, 1, 1);
        eval("gauss(-1,  0,  0) == ", gauss, -1,  0,  0);
        eval("gauss( 0, -1,  0) == ", gauss,  0, -1,  0);
        eval("gauss( 0,  0, -1) == ", gauss,  0,  0, -1);
        eval("gauss(-1,  0, -1) == ", gauss, -1,  0, -1);
        eval("gauss(-1, -1,  0) == ", gauss, -1, -1,  0);
        eval("gauss( 0, -1, -1) == ", gauss,  0, -1, -1);

        std::cout << std::endl << std::endl;

        // create a second client instance connected to the already existing
        // interpolation object
        interpolate3d::interpolate3d gauss_connected;
        gauss_connected.connect("/interpolate3d_client/gauss");

        // use it to calculate some values
        eval("gauss(0, 0, 0) == ", gauss_connected, 0, 0, 0);
        eval("gauss(1, 0, 0) == ", gauss_connected, 1, 0, 0);
        eval("gauss(0, 1, 0) == ", gauss_connected, 0, 1, 0);
        eval("gauss(0, 0, 1) == ", gauss_connected, 0, 0, 1);
        eval("gauss(1, 0, 1) == ", gauss_connected, 1, 0, 1);
        eval("gauss(1, 1, 0) == ", gauss_connected, 1, 1, 0);
        eval("gauss(0, 1, 1) == ", gauss_connected, 0, 1, 1);
        eval("gauss(-1,  0,  0) == ", gauss_connected, -1,  0,  0);
        eval("gauss( 0, -1,  0) == ", gauss_connected,  0, -1,  0);
        eval("gauss( 0,  0, -1) == ", gauss_connected,  0,  0, -1);
        eval("gauss(-1,  0, -1) == ", gauss_connected, -1,  0, -1);
        eval("gauss(-1, -1,  0) == ", gauss_connected, -1, -1,  0);
        eval("gauss( 0, -1, -1) == ", gauss_connected,  0, -1, -1);
    }

    hpx::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}

