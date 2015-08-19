
//  Copyright (c) 2011-2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "jacobi_nonuniform.hpp"

#include <utility>
#include <hpx/hpx_fwd.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/high_resolution_timer.hpp>

namespace jacobi_smp {

    void jacobi_kernel_wrap(
        range const & r,
        crs_matrix<double> const & A,
        std::vector<double> & x_dst,
        std::vector<double> const & x_src, std::vector<double> const & b)
    {
        for(std::size_t row = r.begin(); row < r.end(); ++row)
        {
            jacobi_kernel_nonuniform(A, x_dst, x_src, b, row);
        }
    }

    void jacobi(
        crs_matrix<double> const & A
      , std::vector<double> const & b
      , std::size_t iterations
      , std::size_t block_size
    )
    {
        typedef std::vector<double> vector_type;

        boost::shared_ptr<vector_type> dst(new vector_type(b));
        boost::shared_ptr<vector_type> src(new vector_type(b));

        std::vector<range> block_ranges;
        // pre-computing ranges for the different blocks
        for(std::size_t i = 0; i < dst->size(); i += block_size)
        {
            block_ranges.push_back(
                range(i, std::min<std::size_t>(dst->size(), i + block_size)));
        }

        // pre-computing dependencies
        std::vector<std::vector<std::size_t> > dependencies(block_ranges.size());
        for(std::size_t b = 0; b < block_ranges.size(); ++b)
        {
            for(std::size_t i = block_ranges[b].begin(); i < block_ranges[b].end(); ++i)
            {
                std::size_t begin = A.row_begin(i);
                std::size_t end = A.row_end(i);

                for(std::size_t ii = begin; ii < end; ++ii)
                {
                    std::size_t idx = A.indices[ii];
                    for(std::size_t j = 0; j < block_ranges.size(); ++j)
                    {
                        if(block_ranges[j].begin() <= idx && idx < block_ranges[j].end())
                        {
                            if(std::find(dependencies[b].begin(),
                                dependencies[b].end(), j) == dependencies[b].end())
                            {
                                dependencies[b].push_back(j);
                            }
                            break;
                        }
                    }
                }
            }
        }

        typedef std::vector<hpx::shared_future<void> > future_vector;
        boost::shared_ptr<future_vector> deps_dst
            (new future_vector(dependencies.size(), hpx::make_ready_future()));
        boost::shared_ptr<future_vector> deps_src
            (new future_vector(dependencies.size(), hpx::make_ready_future()));

        hpx::util::high_resolution_timer t;
        for(std::size_t iter = 0; iter < iterations; ++iter)
        {
            for(std::size_t block = 0; block < block_ranges.size(); ++block)
            {
                std::vector<std::size_t> const & deps(dependencies[block]);
                std::vector<hpx::shared_future<void> > trigger;
                trigger.reserve(deps.size());
                for (std::size_t dep : deps)
                {
                    trigger.push_back((*deps_src)[dep]);
                }

                (*deps_dst)[block]
                    = hpx::when_all(std::move(trigger)).then(
                        hpx::launch::async,
                        hpx::util::bind(
                            jacobi_kernel_wrap
                          , block_ranges[block]
                          , boost::cref(A)
                          , boost::ref(*dst)
                          , boost::cref(*src)
                          , boost::cref(b)
                        )
                    );
            }
            std::swap(dst, src);
            std::swap(deps_dst, deps_src);
        }

        hpx::wait_all(*deps_dst);
        hpx::wait_all(*deps_src);

        double time_elapsed = t.elapsed();
        std::cout << dst->size() << " "
            << ((double(dst->size() * iterations)/1e6)/time_elapsed) << " MLUPS/s\n"
            << std::flush;
    }
}
