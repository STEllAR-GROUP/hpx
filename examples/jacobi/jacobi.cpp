
//  Copyright (c) 2012 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/timing.hpp>

#include <hpx/iostream.hpp>

#include "jacobi_component/grid.hpp"
#include "jacobi_component/solver.hpp"

#include <cstddef>
#include <string>
#include <vector>

using hpx::program_options::variables_map;
using hpx::program_options::options_description;
using hpx::program_options::value;

using hpx::chrono::high_resolution_timer;

using hpx::cout;
using hpx::flush;

int hpx_main(variables_map & vm)
{
    {
        std::size_t nx = vm["nx"].as<std::size_t>();
        std::size_t ny = vm["ny"].as<std::size_t>();
        std::size_t max_iterations = vm["max_iterations"].as<std::size_t>();
        std::size_t line_block = vm["line_block"].as<std::size_t>();

        jacobi::grid u(nx, ny, 1.0);

        jacobi::solver solver(u, nx, line_block);

        solver.run(max_iterations);
    }

    return hpx::finalize();
}

int main(int argc, char **argv)
{
    options_description
        desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        (
            "output"
          , value<std::string>()
          , "Output results to file"
        )
        (
            "nx"
          , value<std::size_t>()->default_value(10)
          , "Number of elements in x direction (columns)"
        )
        (
            "ny"
          , value<std::size_t>()->default_value(10)
          , "Number of elements in y direction (rows)"
        )
        (
            "max_iterations"
          , value<std::size_t>()->default_value(10)
          , "Maximum number of iterations"
        )
        (
            "line_block"
          , value<std::size_t>()->default_value(10)
          , "Number of line elements to block the iteration"
        )
        ;

    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(argc, argv, init_args);
}
#endif
