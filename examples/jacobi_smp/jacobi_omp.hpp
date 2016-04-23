
//  Copyright (c) 2011-2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/util/high_resolution_timer.hpp>

#include <memory>
#include <string>
#include <vector>

#include "jacobi.hpp"

namespace jacobi_smp {
    void jacobi(
        std::size_t n
      , std::size_t iterations, std::size_t block_size
      , std::string output_filename)
    {
        typedef std::vector<double> vector;

        std::shared_ptr<vector> grid_new(new vector(n * n, 1));
        std::shared_ptr<vector> grid_old(new vector(n * n, 1));

        hpx::util::high_resolution_timer t;
        for(std::size_t i = 0; i < iterations; ++i)
        {
            // MSVC is unhappy if the OMP loop variable is unsigned
#pragma omp parallel for schedule(JACOBI_SMP_OMP_SCHEDULE)
            for(boost::int64_t y = 1; y < boost::int64_t(n-1); ++y)
            {
                      double * dst = &(*grid_new)[y * n];
                const double * src = &(*grid_new)[y * n];
                jacobi_kernel(
                    dst
                  , src
                  , n
                );
            }
            std::swap(grid_new, grid_old);
        }

        report_timing(n, iterations, t.elapsed());
        output_grid(output_filename, *grid_old, n);
   }
}
