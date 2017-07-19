//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>

#include <iostream>
#include <string>

#include "interpolate1d/interpolate1d.hpp"

using boost::program_options::variables_map;
using boost::program_options::options_description;

double const pi = 4*std::atan(1.);

///////////////////////////////////////////////////////////////////////////////
inline void
eval(char const* expr, interpolate1d::interpolate1d& sine, double value)
{
    std::cout << expr << sine.interpolate(value)
              << " (expected: " << std::sin(value) << ")"
              << std::endl;
}

int hpx_main()
{
    std::string datafilename("sine.h5");
    int num_localities = 7;

    {
        // create the distributed interpolation object on num_localities
        interpolate1d::interpolate1d sine(datafilename, num_localities);

        // use it to calculate some values
        eval("sin(0) == ", sine, 0);
        eval("sin(pi/3) == ", sine, pi/3);
        eval("sin(pi/2) == " , sine, pi/2);
        eval("sin(2*pi/3) == " , sine, 2*pi/3);
        eval("sin(pi) == " , sine, pi);
        eval("sin(4*pi/3) == " , sine, 4*pi/3);
        eval("sin(3*pi/2) == " , sine, 3*pi/2);
        eval("sin(5*pi/3) == " , sine, 5*pi/3);
        eval("sin(2*pi) == " , sine, 2*pi);
    }

    hpx::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);       // Initialize and run HPX
}

