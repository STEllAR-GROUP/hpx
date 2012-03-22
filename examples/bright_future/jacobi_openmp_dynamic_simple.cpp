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
typedef grid_type::size_type size_type;

void gs(
    size_type n_x
  , size_type n_y
  , double hx_
  , double hy_
  , double k_
  , double relaxation_
  , unsigned max_iterations
  , unsigned iteration_block
  , unsigned block_size
  , std::size_t cache_block
  , std::string const & output
)
{
    std::vector<grid_type> u(2, grid_type(n_x, n_y, block_size, 1));

    high_resolution_timer t;
    size_type old = 0;
    size_type new_ = 1;
    t.restart();
    for(std::size_t iter = 0; iter < max_iterations; ++iter)
    {
#pragma omp parallel for shared(u) schedule(dynamic)
        for(size_type y_block = 1; y_block < n_y-1; y_block += block_size)
        {
            size_type y_end = (std::min)(y_block + block_size, n_y-1);
            for(size_type x_block = 1; x_block < n_x-1; x_block += block_size)
            {
                size_type x_end = (std::min)(x_block + block_size, n_x-1);
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
    cout << n_x-1 << "x" << n_y-1 << " " << ((((n_x-1)*(n_y-1) * max_iterations)/1e6)/time_elapsed) << " MLUPS/s\n" << flush;

    if(!output.empty())
    {
        std::ofstream file(output.c_str());
        for(size_type x = 0; x < n_x; ++x)
        {
            for(size_type y = 0; y < n_y; ++y)
            {
                file << x << " " << y << " " << 2 * u[old](x, y) << "\n";
            }
            file << "\n";
        }
    }
}
