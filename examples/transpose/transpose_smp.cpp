//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/parallel_numeric.hpp>

#include <boost/range/irange.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#define COL_SHIFT 1000.00           // Constant to shift column index
#define ROW_SHIFT 0.001             // Constant to shift row index

bool verbose = false;

double test_results(std::uint64_t order, std::vector<double> const & trans);

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    std::uint64_t order = vm["matrix_size"].as<std::uint64_t>();
    std::uint64_t iterations = vm["iterations"].as<std::uint64_t>();
    std::uint64_t tile_size = order;

    if(vm.count("tile_size"))
        tile_size = vm["tile_size"].as<std::uint64_t>();

    verbose = vm.count("verbose") ? true : false;

    std::uint64_t bytes =
        static_cast<std::uint64_t>(2.0 * sizeof(double) * order * order);

    std::vector<double> A(order * order);
    std::vector<double> B(order * order);

    std::cout
        << "Serial Matrix transpose: B = A^T\n"
        << "Matrix order          = " << order << "\n";
    if(tile_size < order)
        std::cout << "Tile size             = " << tile_size << "\n";
    else
        std::cout << "Untiled\n";
    std::cout
        << "Number of iterations  = " << iterations << "\n";

    using hpx::parallel::for_each;
    using hpx::parallel::execution::par;

    const std::uint64_t start = 0;

    // Fill the original matrix, set transpose to known garbage value.
    auto range = boost::irange(start, order);
    // parallel for
    for_each(par, boost::begin(range), boost::end(range),
        [&](std::uint64_t i)
        {
            for(std::uint64_t j = 0; j < order; ++j)
            {
                A[i * order + j] = COL_SHIFT * j + ROW_SHIFT * i;
                B[i * order + j] = -1.0;
            }
        }
    );

    double errsq = 0.0;
    double avgtime = 0.0;
    double maxtime = 0.0;
    double mintime = 366.0 * 24.0*3600.0; // set the minimum time to a large value;
                                          // one leap year should be enough
    for(std::uint64_t iter = 0; iter < iterations; ++iter)
    {
        hpx::util::high_resolution_timer t;
        if(tile_size < order)
        {
            auto range = boost::irange(start, order+tile_size, tile_size);
            // parallel for
            for_each(par, boost::begin(range), boost::end(range),
                [&](std::uint64_t i)
                {
                    for(std::uint64_t j = 0; j < order; j += tile_size)
                    {
                        std::uint64_t i_max = (std::min)(order, i + tile_size);
                        std::uint64_t j_max = (std::min)(order, j + tile_size);

                        for(std::uint64_t it = i; it < i_max; ++it)
                        {
                            for(std::uint64_t jt = j; jt < j_max; ++jt)
                            {
                                B[it + order * jt] = A[jt + order * it];
                            }
                        }
                    }
                }
            );
        }
        else
        {
            auto range = boost::irange(start, order);
            // parallel for
            for_each(par, boost::begin(range), boost::end(range),
                [&](std::uint64_t i)
                {
                    for(std::uint64_t j = 0; j < order; ++j)
                    {
                        B[i + order * j] = A[j + order * i];
                    }
                }
            );
        }

        double elapsed = t.elapsed();

        if(iter > 0 || iterations == 1) // Skip the first iteration
        {
            avgtime = avgtime + elapsed;
            maxtime = (std::max)(maxtime, elapsed);
            mintime = (std::min)(mintime, elapsed);
        }

        errsq += test_results(order, B);
    } // end of iter loop

    // Analyze and output results

    double epsilon = 1.e-8;
    if(errsq < epsilon)
    {
        std::cout << "Solution validates\n";
        avgtime = avgtime/static_cast<double>(
            (std::max)(iterations-1, static_cast<std::uint64_t>(1)));
        std::cout
          << "Rate (MB/s): " << 1.e-6 * bytes/mintime << ", "
          << "Avg time (s): " << avgtime << ", "
          << "Min time (s): " << mintime << ", "
          << "Max time (s): " << maxtime << "\n";

        if(verbose)
            std::cout << "Squared errors: " << errsq << "\n";
    }
    else
    {
        std::cout
          << "ERROR: Aggregate squared error " << errsq
          << " exceeds threshold " << epsilon << "\n";
        hpx::terminate();
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace boost::program_options;

    options_description desc_commandline;
    desc_commandline.add_options()
        ("matrix_size", value<std::uint64_t>()->default_value(1024),
         "Matrix Size")
        ("iterations", value<std::uint64_t>()->default_value(10),
         "# iterations")
        ("tile_size", value<std::uint64_t>(),
         "Number of tiles to divide the individual matrix blocks for improved "
         "cache and TLB performance")
        ( "verbose", "Verbose output")
    ;

    return hpx::init(desc_commandline, argc, argv);
}

double test_results(std::uint64_t order, std::vector<double> const & trans)
{

    using hpx::parallel::transform_reduce;
    using hpx::parallel::execution::par;

    const std::uint64_t start = 0;

    auto range = boost::irange(start, order);
    // parallel reduce
    double errsq =
        transform_reduce(par, boost::begin(range), boost::end(range),
            [&](std::uint64_t i) -> double
            {
                double errsq = 0.0;
                for(std::uint64_t j = 0; j < order; ++j)
                {
                    double diff = trans[i * order + j] -
                        (COL_SHIFT*i + ROW_SHIFT * j);
                    errsq += diff * diff;
                }
                return errsq;
            },
            0.0,
            [](double lhs, double rhs) { return lhs + rhs; }
        );

    if(verbose)
        std::cout << " Squared sum of differences: " << errsq << "\n";

    return errsq;
}
