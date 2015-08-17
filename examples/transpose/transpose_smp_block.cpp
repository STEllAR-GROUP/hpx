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
#include <vector>

#define COL_SHIFT 1000.00           // Constant to shift column index
#define ROW_SHIFT 0.001             // Constant to shift row index

bool verbose = false;

typedef std::vector<double> block;
typedef double* sub_block;

void transpose(sub_block A, sub_block B, boost::uint64_t block_order,
    boost::uint64_t tile_size);
double test_results(boost::uint64_t order, boost::uint64_t block_order,
    std::vector<block> const & trans);

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    boost::uint64_t order = vm["matrix_size"].as<boost::uint64_t>();
    boost::uint64_t iterations = vm["iterations"].as<boost::uint64_t>();
    boost::uint64_t num_blocks = vm["num_blocks"].as<boost::uint64_t>();
    boost::uint64_t tile_size = order;

    if(vm.count("tile_size"))
        tile_size = vm["tile_size"].as<boost::uint64_t>();

    verbose = vm.count("verbose") ? true : false;

    boost::uint64_t bytes =
        static_cast<boost::uint64_t>(2.0 * sizeof(double) * order * order);

    boost::uint64_t block_order = order / num_blocks;
    boost::uint64_t col_block_size = order * block_order;

    std::vector<block> A(num_blocks, block(col_block_size));
    std::vector<block> B(num_blocks, block(col_block_size));

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
    using hpx::parallel::par;
    using hpx::parallel::task;

    const boost::uint64_t start = 0;

    // Fill the original matrix, set transpose to known garbage value.
    auto range = boost::irange(start, num_blocks);
    for_each(par, boost::begin(range), boost::end(range),
        [&](boost::uint64_t b)
        {
            for(boost::uint64_t i = 0; i < order; ++i)
            {
                for(boost::uint64_t j = 0; j < block_order; ++j)
                {
                    double col_val = COL_SHIFT * (b*block_order + j);

                    A[b][i * block_order + j] = col_val + ROW_SHIFT * i;
                    B[b][i * block_order + j] = -1.0;
                }
            }
        }
    );

    double errsq = 0.0;
    double avgtime = 0.0;
    double maxtime = 0.0;
    double mintime = 366.0 * 24.0*3600.0; // set the minimum time to a large value;
                                         // one leap year should be enough
    for(boost::uint64_t iter = 0; iter < iterations; ++iter)
    {
        hpx::util::high_resolution_timer t;

        auto range = boost::irange(start, num_blocks);

        std::vector<hpx::shared_future<void> > transpose_futures;
        transpose_futures.resize(num_blocks);

        for_each(par, boost::begin(range), boost::end(range),
            [&](boost::uint64_t b)
            {
                transpose_futures[b] =
                    for_each(par(task), boost::begin(range), boost::end(range),
                        [&, b](boost::uint64_t phase)
                        {
                            const boost::uint64_t block_size = block_order * block_order;
                            const boost::uint64_t from_block = phase;
                            const boost::uint64_t from_phase = b;
                            const boost::uint64_t A_offset = from_phase * block_size;
                            const boost::uint64_t B_offset = phase * block_size;

                            transpose(&A[from_block][A_offset], &B[b][B_offset],
                                block_order, tile_size);
                        }
                    ).share();
            }
        );

        hpx::wait_all(transpose_futures);

        double elapsed = t.elapsed();

        if(iter > 0 || iterations == 1) // Skip the first iteration
        {
            avgtime = avgtime + elapsed;
            maxtime = (std::max)(maxtime, elapsed);
            mintime = (std::min)(mintime, elapsed);
        }

        errsq += test_results(order, block_order, B);
    } // end of iter loop

    // Analyze and output results

    double epsilon = 1.e-8;
    if(errsq < epsilon)
    {
        std::cout << "Solution validates\n";
        avgtime = avgtime/static_cast<double>(
            (std::max)(iterations-1, static_cast<boost::uint64_t>(1)));
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
        ("matrix_size", value<boost::uint64_t>()->default_value(1024),
         "Matrix Size")
        ("iterations", value<boost::uint64_t>()->default_value(10),
         "# iterations")
        ("tile_size", value<boost::uint64_t>(),
         "Number of tiles to divide the individual matrix blocks for improved "
         "cache and TLB performance")
        ("num_blocks", value<boost::uint64_t>()->default_value(256),
         "Number of blocks to divide the individual matrix blocks for improved "
         "cache and TLB performance")
        ( "verbose", "Verbose output")
    ;

    return hpx::init(desc_commandline, argc, argv);
}

void transpose(sub_block A, sub_block B, boost::uint64_t block_order,
    boost::uint64_t tile_size)
{
    if(tile_size < block_order)
    {
        for(boost::uint64_t i = 0; i < block_order; i += tile_size)
        {
            for(boost::uint64_t j = 0; j < block_order; j += tile_size)
            {
                boost::uint64_t i_max = (std::min)(block_order, i + tile_size);
                boost::uint64_t j_max = (std::min)(block_order, j + tile_size);

                for(boost::uint64_t it = i; it < i_max; ++it)
                {
                    for(boost::uint64_t jt = j; jt < j_max; ++jt)
                    {
                        B[it + block_order * jt] = A[jt + block_order * it];
                    }
                }
            }
        }
    }
    else
    {
        for(boost::uint64_t i = 0; i < block_order; ++i)
        {
            for(boost::uint64_t j = 0; j < block_order; ++j)
            {
                B[i + block_order * j] = A[j + block_order * i];
            }
        }
    }
}

double test_results(boost::uint64_t order, boost::uint64_t block_order,
    std::vector<block> const & trans)
{
    using hpx::parallel::for_each;
    using hpx::parallel::par;

    const boost::uint64_t start = 0;
    const boost::uint64_t end = trans.size();

    // Fill the original matrix, set transpose to known garbage value.
    auto range = boost::irange(start, end);
    double errsq =
        transform_reduce(par, boost::begin(range), boost::end(range),
            [&](boost::uint64_t b) -> double
            {
                double errsq = 0.0;
                for(boost::uint64_t i = 0; i < order; ++i)
                {
                    double col_val = COL_SHIFT * i;
                    for(boost::uint64_t j = 0; j < block_order; ++j)
                    {
                        double diff = trans[b][i * block_order + j] -
                            (col_val + ROW_SHIFT * (b * block_order + j));
                        errsq += diff * diff;
                    }
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
