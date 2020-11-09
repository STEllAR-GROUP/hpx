
//  Copyright (c) 2011-2013 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#if !defined(JACOBI_SMP_NO_HPX)
#include <hpx/hpx_init.hpp>
#endif

#include <hpx/modules/program_options.hpp>

#include <cstddef>
#include <iostream>
#include <string>

using hpx::program_options::variables_map;
using hpx::program_options::options_description;
using hpx::program_options::value;
using hpx::program_options::store;
using hpx::program_options::parse_command_line;

#include "jacobi.hpp"

namespace jacobi_smp {

    void jacobi_kernel(
        double * dst
      , const double * src
      , std::size_t n
    )
    {
#ifdef HPX_INTEL_VERSION
#pragma vector always
#pragma unroll(4)
#endif
        for(std::size_t x = 1; x < n-1; ++x)
        {
            dst[x]
                // x-n might underflow an unsigned type. Casting to a signed type has the
                // wanted effect
                = (src[std::ptrdiff_t(x - n)] + src[x + n] + src[x] +
                      src[x - 1] + src[x + 1]) *
                0.2;
        }
    }
}

int hpx_main(variables_map &vm)
{
    {
        std::size_t n           = vm["n"].as<std::size_t>();
        std::size_t iterations  = vm["iterations"].as<std::size_t>();
        std::size_t block_size  = vm["block-size"].as<std::size_t>();

        std::string output_filename;
        if(vm.count("output-filename"))
        {
            output_filename = vm["output-filename"].as<std::string>();
        }

        jacobi_smp::jacobi(n, iterations, block_size, output_filename);
    }

#if defined(JACOBI_SMP_NO_HPX)
    return 0;
#else
    return hpx::finalize();
#endif
}

int main(int argc, char **argv)
{
    options_description
        desc_cmd("usage: " HPX_APPLICATION_STRING " [options]");

    desc_cmd.add_options()
    (
        "n", value<std::size_t>()->default_value(16)
      , "Will run on grid with dimensions (n x n)"
    )
    (
        "iterations", value<std::size_t>()->default_value(1000)
      , "Number of iterations"
    )
    (
        "block-size", value<std::size_t>()->default_value(256)
      , "Block size of the different chunks to calculate in parallel"
    )
    (
        "output-filename", value<std::string>()
      , "Filename of the result (if empty no result is written)"
    );

#if defined(JACOBI_SMP_NO_HPX)
    variables_map vm;
    desc_cmd.add_options()("help", "This help message");
    store(parse_command_line(argc, argv, desc_cmd), vm);
    if(vm.count("help"))
    {
        std::cout << desc_cmd;
        return 1;
    }
    return hpx_main(vm);
#else
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_cmd;

    return hpx::init(argc, argv, init_args);
#endif

}
