// Copyright (c) 2019 Weile Wei
// Copyright (c) 2019 Maxwell Reeser
// Copyright (c) 2019 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////
/// This is an example of distributed matrix transposition using distributed_object.
///
/// A distributed object is a single logical object partitioned across
/// a set of localities. (A locality is a single node in a cluster or a
/// NUMA domain in a SMP machine.) Each locality constructs an instance of
/// distributed_object<T>, where a value of type T represents the value of this
/// this locality's instance value. Once distributed_object<T> is constrcuted, it
/// has a universal name which can be used on any locality in the given
/// localities to locate the resident instance.

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/algorithms/transform_reduce.hpp>

#include <boost/range/irange.hpp>

#include <hpx/include/async.hpp>
#include <hpx/include/dataflow.hpp>
#include <hpx/lcos/barrier.hpp>
#include <hpx/lcos/distributed_object.hpp>
#include <hpx/lcos/when_all.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

bool verbose = false;    // command line argument

typedef double* sub_block;

#define COL_SHIFT 1000.00    // Constant to shift column index
#define ROW_SHIFT 0.01       // Constant to shift row index

// Register type for template components
REGISTER_DISTRIBUTED_OBJECT_PART(double);
using myVectorDouble = std::vector<double>;
REGISTER_DISTRIBUTED_OBJECT_PART(myVectorDouble);

using hpx::lcos::distributed_object;

///////////////////////////////////////////////////////////////////////////////
// transpose matrix when the target matrix is in a remote node
void transpose(hpx::future<std::vector<double>> Af, std::uint64_t A_offset,
    distributed_object<std::vector<double>>& B_temp, std::uint64_t B_offset,
    std::uint64_t block_size, std::uint64_t block_order,
    std::uint64_t tile_size);

///////////////////////////////////////////////////////////////////////////////
// transpose matrix when the target and destination matrix are in a same node
void transpose_local(distributed_object<std::vector<double>>& A_temp,
    std::uint64_t A_offset, distributed_object<std::vector<double>>& B_temp,
    std::uint64_t B_offset, std::uint64_t block_size, std::uint64_t block_order,
    std::uint64_t tile_size);

double test_results(std::uint64_t order, std::uint64_t block_order,
    std::vector<distributed_object<std::vector<double>>>& trans,
    std::uint64_t blocks_start, std::uint64_t blocks_end);

///////////////////////////////////////////////////////////////////////////////
void run_matrix_transposition(boost::program_options::variables_map& vm)
{
    hpx::id_type here = hpx::find_here();
    bool root = here == hpx::find_root_locality();

    std::uint64_t num_localities = hpx::get_num_localities().get();

    std::uint64_t order = vm["matrix_size"].as<std::uint64_t>();
    std::uint64_t iterations = vm["iterations"].as<std::uint64_t>();
    std::uint64_t num_local_blocks = vm["num_blocks"].as<std::uint64_t>();
    std::uint64_t tile_size = order;

    if (vm.count("tile_size"))
        tile_size = vm["tile_size"].as<std::uint64_t>();

    verbose = vm.count("verbose") > 0 ? true : false;

    std::uint64_t bytes =
        static_cast<std::uint64_t>(2.0 * sizeof(double) * order * order);

    std::uint64_t num_blocks = num_localities * num_local_blocks;

    std::uint64_t block_order = order / num_blocks;
    std::uint64_t col_block_size = order * block_order;

    std::uint64_t id = hpx::get_locality_id();

    std::vector<distributed_object<std::vector<double>>> A(num_blocks);
    std::vector<distributed_object<std::vector<double>>> B(num_blocks);

    std::uint64_t blocks_start = id * num_local_blocks;
    std::uint64_t blocks_end = (id + 1) * num_local_blocks;

    // First allocate and create our local blocks
    for (std::uint64_t b = 0; b != num_local_blocks; ++b)
    {
        std::uint64_t block_idx = b + blocks_start;
        A[block_idx] = distributed_object<std::vector<double>>(
            "A", std::vector<double>(col_block_size));
        B[block_idx] = distributed_object<std::vector<double>>(
            "B", std::vector<double>(col_block_size));
    }

    using hpx::parallel::for_each;
    using hpx::parallel::execution::par;

    // Fill the original matrix, set transpose to known garbage value.
    auto range = boost::irange(blocks_start, blocks_end);
    hpx::parallel::for_each(
        par, std::begin(range), std::end(range), [&](std::uint64_t b) {
            for (std::uint64_t i = 0; i != order; ++i)
            {
                for (std::uint64_t j = 0; j != block_order; ++j)
                {
                    double col_val = COL_SHIFT * (b * block_order + j);
                    (*A[b])[i * block_order + j] = col_val + ROW_SHIFT * i;
                    (*B[b])[i * block_order + j] = -1.0;
                }
            }
        });

    // wait all matrix to be initialized
    hpx::lcos::barrier b("wait_for_init", hpx::find_all_localities().size(),
        hpx::get_locality_id());
    b.wait();

    if (root)
    {
        std::cout << "Serial Matrix transpose: B = A^T\n"
                  << "Matrix order           = " << order << "\n"
                  << "Matrix local columns   = " << block_order << "\n"
                  << "Total number of blocks = " << num_blocks << "\n"
                  << "Number of localities   = " << num_localities << "\n";
        if (tile_size < order)
            std::cout << "Tile size             = " << tile_size << "\n";
        else
            std::cout << "Untiled\n";
        std::cout << "Number of iterations  = " << iterations << "\n";
    }

    double errsq = 0.0;
    double avgtime = 0.0;
    double maxtime = 0.0;
    double mintime =
        366.0 * 24.0 * 3600.0;    // set the minimum time to a large value;
                                  // one leap year should be enough

    // start of iter loop
    for (std::uint64_t iter = 0; iter < iterations; ++iter)
    {
        // starts matrix transposition
        std::vector<hpx::future<void>> block_futures;
        block_futures.resize(num_local_blocks);
        hpx::util::high_resolution_timer t;
        for_each(par, std::begin(range), std::end(range), [&](std::uint64_t b) {
            std::vector<hpx::future<void>> phase_futures;
            phase_futures.reserve(num_blocks);

            auto phase_range =
                boost::irange(static_cast<std::uint64_t>(0), num_blocks);
            for (std::uint64_t phase : phase_range)
            {
                const std::uint64_t block_size = block_order * block_order;
                const std::uint64_t from_block = phase;
                const std::uint64_t from_phase = b;
                const std::uint64_t A_offset = from_phase * block_size;
                const std::uint64_t B_offset = phase * block_size;
                const std::uint64_t from_locality = from_block % num_localities;
                // Perform matrix transposition locally
                if (blocks_start <= phase && phase < blocks_end)
                {
                    phase_futures.push_back(hpx::async(&transpose_local,
                        std::ref(A[from_block]), A_offset, std::ref(B[b]),
                        B_offset, block_size, block_order, tile_size));
                }
                // fetch remote matrix and then transpose the matrix
                else
                {
                    phase_futures.push_back(hpx::dataflow(&transpose,
                        A[b].fetch(from_locality), A_offset, std::ref(B[b]),
                        B_offset, block_size, block_order, tile_size));
                }
            }

            block_futures[b - blocks_start] = hpx::when_all(phase_futures);
        });

        hpx::wait_all(block_futures);

        double elapsed = t.elapsed();

        if (iter > 0 || iterations == 1)    // Skip the first iteration
        {
            avgtime = avgtime + elapsed;
            maxtime = (std::max)(maxtime, elapsed);
            mintime = (std::min)(mintime, elapsed);
        }

        if (root)
            errsq +=
                test_results(order, block_order, B, blocks_start, blocks_end);
    }    // end of iter loop

    double epsilon = 1.e-8;
    if (root)
    {
        if (errsq < epsilon)
        {
            std::cout << "Solution validates\n";
            avgtime = avgtime /
                static_cast<double>(
                    (std::max)(iterations - 1, static_cast<std::uint64_t>(1)));
            std::cout << "Rate (MB/s): " << 1.e-6 * bytes / mintime << ", "
                      << "Avg time (s): " << avgtime << ", "
                      << "Min time (s): " << mintime << ", "
                      << "Max time (s): " << maxtime << "\n";

            if (verbose)
                std::cout << "Squared errors: " << errsq << "\n";
        }
        else
        {
            std::cout << "ERROR: Aggregate squared error " << errsq
                      << " exceeds threshold " << epsilon << "\n";
            hpx::terminate();
        }
    }
}

void transpose(hpx::future<std::vector<double>> Af, std::uint64_t A_offset,
    distributed_object<std::vector<double>>& B_temp, std::uint64_t B_offset,
    std::uint64_t block_size, std::uint64_t block_order,
    std::uint64_t tile_size)
{
    std::vector<double> A_temp = Af.get();
    const sub_block A(&(A_temp[A_offset]));
    sub_block B(&((*B_temp)[B_offset]));

    if (tile_size < block_order)
    {
        for (std::uint64_t i = 0; i < block_order; i += tile_size)
        {
            for (std::uint64_t j = 0; j < block_order; j += tile_size)
            {
                std::uint64_t max_i = (std::min)(block_order, i + tile_size);
                std::uint64_t max_j = (std::min)(block_order, j + tile_size);

                for (std::uint64_t it = i; it != max_i; ++it)
                {
                    for (std::uint64_t jt = j; jt != max_j; ++jt)
                    {
                        B[it + block_order * jt] = A[jt + block_order * it];
                    }
                }
            }
        }
    }
    else
    {
        for (std::uint64_t i = 0; i != block_order; ++i)
        {
            for (std::uint64_t j = 0; j != block_order; ++j)
            {
                B[i + block_order * j] = A[j + block_order * i];
            }
        }
    }
}

void transpose_local(distributed_object<std::vector<double>>& A_temp,
    std::uint64_t A_offset, distributed_object<std::vector<double>>& B_temp,
    std::uint64_t B_offset, std::uint64_t block_size, std::uint64_t block_order,
    std::uint64_t tile_size)
{
    const sub_block A(&((*A_temp)[A_offset]));
    sub_block B(&((*B_temp)[B_offset]));

    if (tile_size < block_order)
    {
        for (std::uint64_t i = 0; i < block_order; i += tile_size)
        {
            for (std::uint64_t j = 0; j < block_order; j += tile_size)
            {
                std::uint64_t max_i = (std::min)(block_order, i + tile_size);
                std::uint64_t max_j = (std::min)(block_order, j + tile_size);

                for (std::uint64_t it = i; it != max_i; ++it)
                {
                    for (std::uint64_t jt = j; jt != max_j; ++jt)
                    {
                        B[it + block_order * jt] = A[jt + block_order * it];
                    }
                }
            }
        }
    }
    else
    {
        for (std::uint64_t i = 0; i != block_order; ++i)
        {
            for (std::uint64_t j = 0; j != block_order; ++j)
            {
                B[i + block_order * j] = A[j + block_order * i];
            }
        }
    }
}

double test_results(std::uint64_t order, std::uint64_t block_order,
    std::vector<distributed_object<std::vector<double>>>& trans,
    std::uint64_t blocks_start, std::uint64_t blocks_end)
{
    using hpx::parallel::transform_reduce;
    using hpx::parallel::execution::par;

    // Fill the original matrix, set transpose to known garbage value.
    auto range = boost::irange(blocks_start, blocks_end);
    double errsq = transform_reduce(
        par, std::begin(range), std::end(range), 0.0,
        [](double lhs, double rhs) { return lhs + rhs; },
        [&](std::uint64_t b) -> double {
            sub_block trans_block = &((*(trans[b]))[0]);
            double errsq = 0.0;
            for (std::uint64_t i = 0; i < order; ++i)
            {
                double col_val = COL_SHIFT * i;
                for (std::uint64_t j = 0; j < block_order; ++j)
                {
                    double diff = trans_block[i * block_order + j] -
                        (col_val + ROW_SHIFT * (b * block_order + j));
                    errsq += diff * diff;
                }
            }
            return errsq;
        });
    if (verbose)
    {
        std::cout << " Squared sum of differences: " << errsq << "\n";
    }

    return errsq;
}

int hpx_main(boost::program_options::variables_map& vm)
{
    run_matrix_transposition(vm);
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace boost::program_options;

    options_description desc_commandline;
    desc_commandline.add_options()("matrix_size",
        value<std::uint64_t>()->default_value(1024),
        "Matrix Size")("iterations", value<std::uint64_t>()->default_value(1),
        "# iterations")("tile_size", value<std::uint64_t>(),
        "Number of tiles to divide the individual matrix blocks for improved "
        "cache and TLB performance")("num_blocks",
        value<std::uint64_t>()->default_value(1),
        "Number of blocks to divide the individual matrix blocks for "
        "improved cache and TLB performance")("verbose", "Verbose output");

    // Initialize and run HPX, this example requires to run hpx_main on all
    // localities
    std::vector<std::string> const cfg = {"hpx.run_hpx_main!=1"};
    return hpx::init(desc_commandline, argc, argv, cfg);
}
