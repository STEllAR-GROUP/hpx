//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "grid.hpp"

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#ifdef BRIGHT_FUTURE_NO_HPX
#include <iostream>
#ifdef BRIGHT_FUTURE_WITH_MPI
#include <mpi.h>
#endif
#else
#include <hpx/include/iostreams.hpp>
#endif

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::util::high_resolution_timer;

#ifdef BRIGHT_FUTURE_NO_HPX
using std::cout;
using std::flush;
#else
using hpx::init;
using hpx::finalize;

using hpx::cout;
using hpx::flush;
#endif

typedef bright_future::grid<double> grid_type;
typedef grid_type::size_type size_type;

extern void gs(
    /*
    bright_future::grid<double> & u
  , bright_future::grid<double> const & rhs
  */
    size_type n_x
  , size_type n_y
  , double hx
  , double hy
  , double k
  , double relaxation
  , unsigned max_iterations
  , unsigned iteration_block
  , unsigned block_size
  , std::size_t cache_block
  , std::string const & output
);

int hpx_main(variables_map & vm)
{
    {

        size_type width = vm["width"].as<size_type>();
        size_type height = vm["height"].as<size_type>();

        double hx = vm["hx"].as<double>();
        double hy = vm["hy"].as<double>();

        size_type n_x = static_cast<size_type>(width/hx + 0.5) + 1;
        size_type n_y = static_cast<size_type>(height/hy + 0.5) + 1;

        unsigned max_iterations  = vm["max_iterations"].as<unsigned>();
        unsigned iteration_block = vm["iteration_block"].as<unsigned>();
        unsigned block_size      = vm["block_size"].as<unsigned>();
        std::size_t cache_block     = vm["cache_block"].as<std::size_t>();

        double k = 6.283185307179586232;
        double relaxation = 1.0;

        std::string output;
        if(vm.count("output"))
        {
            output = vm["output"].as<std::string>();
        }

        gs(n_x, n_y, hx, hy, k, relaxation, max_iterations, iteration_block, block_size, cache_block, output);

#ifndef BRIGHT_FUTURE_NO_HPX
        finalize();
#endif
    }

#ifdef BRIGHT_FUTURE_WITH_MPI
    MPI_Finalize();
#endif

    return 0;
}

int main(int argc, char **argv)
{

#ifdef BRIGHT_FUTURE_WITH_MPI
    MPI_Init(&argc, &argv);
#endif

    options_description
        desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        (
            "output"
          , value<std::string>()
          , "Output results to file"
        )
        (
            "width"
          , value<size_type>()->default_value(2)
          , "The width of the domain"
        )
        (
            "height"
          , value<size_type>()->default_value(1)
          , "The height of the domain"
        )
        (
            "hx"
          , value<double>()->default_value(0.2)
          , "Stepsize in x direction"
        )
        (
            "hy"
          , value<double>()->default_value(0.1)
          , "Stepsize in y direction"
        )
        (
            "max_iterations"
          , value<unsigned>()->default_value(10000)
          , "Maximum number of iterations"
        )
        (
            "iteration_block"
          , value<unsigned>()->default_value(100)
          , "Calculate residuum after each iteration block"
        )
        (
            "block_size"
          , value<unsigned>()->default_value(256)
          , "How to block the iteration"
        )
        (
            "cache_block"
          , value<std::size_t>()->default_value(128)
          , "How to block the iteration"
        )
        ;

#ifdef BRIGHT_FUTURE_NO_HPX
    variables_map vm;

    desc_commandline.add_options()
        (
            "help", "This help message"
        )
        ;

    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc_commandline), vm);

    if(vm.count("help"))
    {
        cout << desc_commandline;
        return 0;
    }
    return hpx_main(vm);
#else
    return init(desc_commandline, argc, argv);
#endif

}
