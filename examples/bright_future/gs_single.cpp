//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "grid.hpp"
#include <cmath>

#include <hpx/hpx.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <iostream>
#include <boost/phoenix.hpp>
#include <boost/phoenix/stl/cmath.hpp>

using std::cout;
using std::flush;

using hpx::util::high_resolution_timer;

using bright_future::update;
using bright_future::update_residuum;

typedef bright_future::grid<double> grid_type;
typedef grid_type::size_type size_type;

namespace bright_future
{
    template <typename T>
    inline void init_u(grid<T> & u, unsigned n_y, T hx, unsigned x_start = 0)
    {
        using boost::phoenix::placeholders::arg1;
        using boost::phoenix::placeholders::arg2;
        using boost::phoenix::local_names::_a;
        using boost::phoenix::local_names::_x;
        using boost::phoenix::local_names::_y;
        using boost::phoenix::sin;

        u.init(
            let(
                _x = (arg1 + x_start) * hx  // actual x coordinate
            )
            [
                if_else(
                    arg2 == n_y - 1
                  , sin(_x * 6.283) * sinh(6.283)
                  , 0.0
                )
            ]
        );
    }

    template <typename T>
    inline void init_rhs(grid<T> & rhs, T hx, T hy, unsigned x_start = 0, unsigned y_start = 0)
    {
        using boost::phoenix::placeholders::arg1;
        using boost::phoenix::placeholders::arg2;
        using boost::phoenix::local_names::_a;
        using boost::phoenix::local_names::_x;
        using boost::phoenix::local_names::_y;
        using boost::phoenix::sin;

        rhs.init(
            let(
                _x = (arg1 + x_start) * hx  // actual x coordinate
              , _y = (arg2 + y_start) * hy  // actual y coordinate
            )
            [
                39.478 * sin(_x * 6.283) * sinh(_y * 6.283)
            ]
        );
    }

}

void gs(
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
  , std::string const & output
)
{
    grid_type rhs(n_x, n_y);
    grid_type u(n_x, n_y);

    double div = (2.0/(hx*hx) + 2.0/(hy*hy) + (k*k));
    double hx_sq = hx * hx;
    double hy_sq = hy * hy;

    // set our initial values, setting the top boundary to be a dirichlet
    // boundary condition
    bright_future::init_u(u, n_y, hx);
    bright_future::init_rhs(rhs, hx, hy);

    high_resolution_timer t;

    for(unsigned iter = 0; iter < max_iterations; ++iter)//iter += iteration_block)
    {
        // split up iterations so we don't need to check the residual every iteration
        //for(unsigned jter = 0; jter < iteration_block; ++jter)
        {
            // update the "red" points
            size_type y_block, x_block, y, x;
#pragma omp parallel for private(x_block, y_block, x, y)
            for(x_block = 1; x_block < n_x - 1; x_block += block_size)
            {
                for(y_block = 1; y_block < n_y - 1; y_block += block_size)
                {
                    for(y = y_block; y < std::min(y_block + block_size, n_y-1); ++y)
                    {
                        for(x = (y%2) + 1; x < std::min(x_block + block_size, n_x - 1); x += 2)
                        {
                            u(x, y) = update(u, rhs, x, y, hx_sq, hy_sq, div, relaxation);
                        }
                    }
                }
            }

            // update the "black" points
#pragma omp parallel for private(x_block, y_block, x, y)
            for(x_block = 1; x_block < n_x - 1; x_block += block_size)
            {
                for(y_block = 1; y_block < n_y - 1; y_block += block_size)
                {
                    for(y = y_block; y < std::min(y_block + block_size, n_y-1); ++y)
                    {
                        for(x = ((y+1)%2) + 1; x < std::min(x_block + block_size, n_x - 1); x += 2)
                        {
                            u(x, y) = update(u, rhs, x, y, hx_sq, hy_sq, div, relaxation);
                        }
                    }
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
    double time_elapsed = t.elapsed();
    cout << (n_x*n_y) << " " << time_elapsed << "\n" << flush;

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
