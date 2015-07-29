
//  Copyright (c) 2011-2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef JACOBI_SMP_JACOBI_HPP
#define JACOBI_SMP_JACOBI_HPP

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

namespace jacobi_smp {
    void jacobi(
        std::size_t n
      , std::size_t iterations, std::size_t block_size
      , std::string output_filename);

    struct range
    {
        range() : begin_(0), end_(0) {}
        range(std::size_t begin, std::size_t end) : begin_(begin), end_(end) {}

        std::size_t begin() const { return begin_; }
        std::size_t end() const { return end_; }

        std::size_t begin_;
        std::size_t end_;
    };

    void jacobi_kernel(
        double * dst
      , const double * src
      , std::size_t n
    );

    inline void report_timing(std::size_t n, std::size_t iterations, double elapsed)
    {
        std::cout
            << n << "x" << n << " "
            << elapsed << " s "
            << ((double((n-2)*(n-2) * iterations)/1e6)/elapsed) << " MLUPS/s\n";
    }

    inline void output_grid(std::string filename,
        std::vector<double> const & grid, std::size_t n)
    {
        if(!filename.empty())
        {
            std::ofstream file(filename.c_str());
            for(std::size_t x = 0; x < n; ++x)
            {
                for(std::size_t y = 0; y < n; ++y)
                {
                    file << x << " " << y << " " << 2 * grid[x + y * n] << "\n";
                }
                file << "\n";
            }
            file << std::flush;
        }
    }
}

#endif
