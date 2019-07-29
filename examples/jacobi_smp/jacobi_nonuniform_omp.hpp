
//  Copyright (c) 2011-2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/timing/high_resolution_timer.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "jacobi_nonuniform.hpp"

namespace jacobi_smp {
    void jacobi(
        crs_matrix<double> const & A
      , std::vector<double> const & b
      , std::size_t iterations
      , std::size_t block_size
    )
    {
        typedef std::vector<double> vector_type;

        std::shared_ptr<vector_type> dst(new vector_type(b));
        std::shared_ptr<vector_type> src(new vector_type(b));

        hpx::util::high_resolution_timer t;
        for(std::size_t i = 0; i < iterations; ++i)
        {
            // MSVC is unhappy if the OMP loop variable is unsigned
#pragma omp parallel for schedule(JACOBI_SMP_OMP_SCHEDULE)
            for(std::int64_t row = 0; row < std::int64_t(b.size());  ++row)
            {
                jacobi_kernel_nonuniform(
                          A
                        , *dst
                        , *src
                        , b
                        , row
                        );
            }
            std::swap(dst, src);
        }

        double time_elapsed = t.elapsed();
        std::cout << dst->size() << " "
            << ((double(dst->size() * iterations)/1e6)/time_elapsed) << " MLUPS/s\n"
            << std::flush;
    }
}
