//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "grid.hpp"
#include <cmath>

#include <hpx/hpx.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <iostream>

using std::cout;
using std::flush;

using hpx::util::high_resolution_timer;

using bright_future::update;
using bright_future::update_residuum;

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
    grid_type rhs(n_x, n_y);
    grid_type u(n_x, n_y);

    double relaxation;
    double k;
    double hx;
    double hy;
    double div;// = (2.0/(hx*hx) + 2.0/(hy*hy) + (k*k));
    double hx_sq;// = hx * hx;
    double hy_sq;// = hy * hy;
    size_type y = 0;
    size_type x = 0;

    // set our initial values, setting the top boundary to be a dirichlet
    // boundary condition
    unsigned iter;

    high_resolution_timer t;
    {
        hx = hx_;
        hy = hy_;
        k = k_;
        relaxation = relaxation_;
        div = (2.0/(hx*hx) + 2.0/(hy*hy) + (k*k));
        hx_sq = hx * hx;
        hy_sq = hy * hy;
        y = 0;
        x = 0;
#pragma omp parallel for shared(u, rhs, n_x, n_y, hx, hy) private(x, y)  schedule(static)
        for(y = 0; y < n_y; ++y)
        {
            for(x = (y%2) + 1; x < n_x; x += 2)
            {
                u(x, y) = y == (n_y - 1) ? sin((x * hx) * 6.283) * sinh(6.283) : 0.0;
                rhs(x, y) = 39.478 * sin((x * hx) * 6.283) * sinh((y * hy) * 6.283);
            }
        }
#pragma omp parallel for shared(u, rhs, n_x, n_y, hx, hy) private(x, y)  schedule(static)
        for(y = 0; y < n_y; ++y)
        {
            for(x = ((y+1)%2) + 1; x < n_x; x += 2)
            {
                u(x, y) = y == (n_y - 1) ? sin((x * hx) * 6.283) * sinh(6.283) : 0.0;
                rhs(x, y) = 39.478 * sin((x * hx) * 6.283) * sinh((y * hy) * 6.283);
            }
        }

        t.restart();
        for(iter = 0; iter < max_iterations; ++iter)//iter += iteration_block)
        {
            // split up iterations so we don't need to check the residual every iteration
            //for(unsigned jter = 0; jter < iteration_block; ++jter)
            {
                // update the "red" points
#pragma omp parallel for shared(u, rhs, n_x, n_y, hx_sq, hy_sq, div, relaxation) private(x, y)  schedule(static)
                for(y = 1; y < n_y-1; ++y)
                {
                    for(x = (y%2) + 1; x < n_x - 1; x += 2)
                    {
                        u(x, y) = update(u, rhs, x, y, hx_sq, hy_sq, div, relaxation);
                    }
                }
                // update the "black" points
#pragma omp parallel for shared(u, rhs, n_x, n_y, hx_sq, hy_sq, div, relaxation) private(x, y)  schedule(static)
                for(y = 1; y < n_y-1; ++y)
                {
                    for(x = ((y+1)%2) + 1; x < n_x - 1; x += 2)
                    {
                        u(x, y) = update(u, rhs, x, y, hx_sq, hy_sq, div, relaxation);
                    }
                }
            }
            /*
            // check if we converged yet
            grid_type residuum(u.x(), u.y());
            size_type y, x;
#pragma omp parallel for private(x, y)
            for(y = 1; y < n_y-1; ++y)
            {
                for(x = 1; x < n_x-1; ++x)
                {
                    residuum(x, y) = update_residuum(u, rhs, x, y, hx_sq, hy_sq, k);
                }
            }
            double r = 0.0;
#pragma omp parallel for reduction(+:r)
            for(unsigned i = 0; i < residuum.size(); ++i)
            {
                r = r + residuum[i] * residuum[i];
            }

            if(std::sqrt(r) <= 1e-10)
            {
                break;
            }
            */
        }
    }
    double time_elapsed = t.elapsed();
    cout << ((((n_x-2)*(n_y-2) * max_iterations)/1e6)/time_elapsed) << " MLUP/S\n" << flush;

    if(!output.empty())
    {
        std::ofstream file(output.c_str());
        for(size_type x = 0; x < n_x; ++x)
        {
            for(size_type y = 0; y < n_y; ++y)
            {
                file << x * hx << " " << y * hy << " " << u(x, y) << "\n";
            }
            file << "\n";
        }
    }
}
