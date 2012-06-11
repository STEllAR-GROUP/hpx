//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "grid.hpp"
#include <cmath>

#include <hpx/hpx.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <iostream>
#include <fstream>

using std::cout;
using std::flush;

using hpx::util::high_resolution_timer;

using bright_future::range_type;
using bright_future::jacobi_kernel_simple;

typedef bright_future::grid<double> grid_type;

void gs(
    std::size_t n_x
  , std::size_t n_y
  , double //hx_
  , double //hy_
  , double //k_
  , double //relaxation_
  , unsigned max_iterations
  , unsigned //iteration_block
  , unsigned block_size
  , std::size_t cache_block
  , std::string const & output
)
{
    std::vector<grid_type> u(2, grid_type(n_x, n_y, block_size, 1));

    high_resolution_timer t;
    std::size_t old = 0;
    std::size_t new_ = 1;
    t.restart();
    for(std::size_t iter = 0; iter < max_iterations; ++iter)
    {
#pragma omp parallel for shared(u) schedule(static)
        for(std::size_t y_block = 1; y_block < n_y-1; y_block += block_size)
        {
            std::size_t y_end = (std::min)(y_block + block_size, n_y-1);
            for(std::size_t x_block = 1; x_block < n_x-1; x_block += block_size)
            {
                std::size_t x_end = (std::min)(x_block + block_size, n_x-1);
                jacobi_kernel_simple(
                    u
                  , range_type(x_block, x_end)
                  , range_type(y_block, y_end)
                  , old, new_
                  , cache_block
                );
            }
        }
        std::swap(old, new_);
    }
    double time_elapsed = t.elapsed();
    //cout << n_x-1 << "x" << n_y-1 << " " << ((((n_x-2)*(n_y-2) * max_iterations)/1e6)/time_elapsed) << " MLUPS/s\n" << flush;
    cout << n_x-1 << "x" << n_y-1 << " " << ((double((n_x-1)*(n_y-1) * max_iterations)/1e6)/time_elapsed) << " MLUPS/s\n" << flush;

    if(!output.empty())
    {
        std::ofstream file(output.c_str());
        for(std::size_t x = 0; x < n_x; ++x)
        {
            for(std::size_t y = 0; y < n_y; ++y)
            {
                file << x << " " << y << " " << 2 * u[old](x, y) << "\n";
            }
            file << "\n";
        }
    }
}
