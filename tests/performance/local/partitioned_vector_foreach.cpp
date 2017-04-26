//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/partitioned_vector.hpp>
#include <hpx/include/parallel_for_each.hpp>

#include <boost/format.hpp>
#include <boost/range/functions.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "worker_timed.hpp"

///////////////////////////////////////////////////////////////////////////////
// Define the vector types to be used.
HPX_REGISTER_PARTITIONED_VECTOR(int);

///////////////////////////////////////////////////////////////////////////////
int delay = 1000;
int test_count = 100;
int chunk_size = 0;
int num_overlapping_loops = 0;

///////////////////////////////////////////////////////////////////////////////
template <typename Vector>
struct wait_op
{
    typedef typename Vector::value_type value_type;

    void operator()(value_type) const
    {
        worker_timed(delay);
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename Policy, typename Vector>
std::uint64_t foreach_vector(Policy && policy, Vector const& v)
{
    typedef typename Vector::value_type value_type;

    std::uint64_t start = hpx::util::high_resolution_clock::now();

    for (int i = 0; i != test_count; ++i)
    {
        hpx::parallel::for_each(
            std::forward<Policy>(policy),
            boost::begin(v), boost::end(v), wait_op<Vector>()
        );
    }

    return (hpx::util::high_resolution_clock::now() - start) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    std::size_t vector_size = vm["vector_size"].as<std::size_t>();
//     bool csvoutput = vm.count("csv_output") != 0;
    delay = vm["work_delay"].as<int>();
    test_count = vm["test_count"].as<int>();
    chunk_size = vm["chunk_size"].as<int>();

    // verify that input is within domain of program
    if (test_count == 0 || test_count < 0) {
        hpx::cout << "test_count cannot be zero or negative...\n" << hpx::flush;
    }
    else if (delay < 0) {
        hpx::cout << "delay cannot be a negative number...\n" << hpx::flush;
    }
    else {
        // create executor parameters object
        hpx::parallel::static_chunk_size cs(chunk_size);

        // retrieve reference time
        std::vector<int> ref(vector_size);
        std::uint64_t seq_ref = foreach_vector(
            hpx::parallel::execution::seq, ref);
        std::uint64_t par_ref = foreach_vector(
            hpx::parallel::execution::par.with(cs), ref); //-V106

        // sequential hpx::partitioned_vector iteration
        {
            hpx::partitioned_vector<int> v(vector_size);

            hpx::cout << "hpx::partitioned_vector<int>(execution::seq): "
                << foreach_vector(hpx::parallel::execution::seq, v) /
                        double(seq_ref)
                << "\n";
            hpx::cout << "hpx::partitioned_vector<int>(execution::par): "
                << foreach_vector(hpx::parallel::execution::par.with(cs), v) /
                        double(par_ref) //-V106
                << "\n";
        }

        {
            hpx::partitioned_vector<int> v(vector_size, hpx::container_layout(2));

            hpx::cout << "hpx::partitioned_vector<int>(execution::seq, "
                        "container_layout(2)): "
                << foreach_vector(hpx::parallel::execution::seq, v) /
                        double(seq_ref)
                << "\n";
            hpx::cout << "hpx::partitioned_vector<int>(execution::par, "
                        "container_layout(2)): "
                << foreach_vector(hpx::parallel::execution::par.with(cs), v) /
                        double(par_ref) //-V106
                << "\n";
        }

        {
            hpx::partitioned_vector<int> v(vector_size, hpx::container_layout(10));

            hpx::cout << "hpx::partitioned_vector<int>(execution::seq, "
                            "container_layout(10)): "
                << foreach_vector(hpx::parallel::execution::seq, v) /
                        double(seq_ref)
                << "\n";
            hpx::cout << "hpx::partitioned_vector<int>(execution::par, "
                            "container_layout(10)): "
                << foreach_vector(hpx::parallel::execution::par.with(cs), v) /
                        double(par_ref) //-V106
                << "\n";
        }
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    //initialize program
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    boost::program_options::options_description cmdline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ("vector_size"
        , boost::program_options::value<std::size_t>()->default_value(1000)
        , "size of vector (default: 1000)")

        ("work_delay"
        , boost::program_options::value<int>()->default_value(1000)
        , "loop delay per element in nanoseconds (default: 1000)")

        ("test_count"
        , boost::program_options::value<int>()->default_value(100)
        , "number of tests to be averaged (defalt: 100)")

        ("chunk_size"
        , boost::program_options::value<int>()->default_value(0)
        , "number of iterations to combine while parallelization (default: 0)")
        ;

    return hpx::init(cmdline, argc, argv, cfg);
}

