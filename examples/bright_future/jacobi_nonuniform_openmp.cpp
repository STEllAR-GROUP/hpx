//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include "sparse_matrix.hpp"

#include <iostream>
#include <fstream>

using std::cout;
using std::flush;
using hpx::util::high_resolution_timer;

void solve(
    bright_future::crs_matrix<double> const & A
  , std::vector<double> & x
  , std::vector<double> const & b_
  , std::size_t block_size
  , std::size_t max_iterations
)
{
    std::vector<std::vector<double> > u(2, x);

    high_resolution_timer t;
    std::size_t old = 0;
    std::size_t new_ = 1;
    std::cout << "running " << max_iterations << " iterations\n";
    t.restart();
    for(std::size_t iter = 0; iter < max_iterations; ++iter)
    {
#pragma omp parallel for shared(u) schedule(static)
        for(std::size_t i = 0; i < x.size(); i += block_size)
        {
            bright_future::jacobi_kernel_nonuniform(
                A
              , u
              , b_
              , std::pair<std::size_t, std::size_t>(i, (std::min)(i + block_size, x.size()))
              , old
              , new_
            );
        }
        std::swap(old, new_);
    }
    double time_elapsed = t.elapsed();
    cout << x.size() << " "
         << ((double(x.size() * max_iterations)/1e6)/time_elapsed) << " MLUPS/s\n" << flush;

    x = u[old];
}
