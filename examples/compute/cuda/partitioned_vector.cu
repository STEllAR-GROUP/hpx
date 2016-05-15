//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define HPX_APPLICATION_NAME partitioned_vector_cu
#define HPX_APPLICATION_STRING "partitioned_vector_cu"
#define HPX_APPLICATION_EXPORTS

#include <hpx/include/compute.hpp>
#include <hpx/include/partitioned_vector.hpp>

#include <hpx/hpx_init.hpp>

#include <iostream>
#include <numeric>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Define the partitioned vector types to be used.
typedef hpx::compute::cuda::allocator<int> target_allocator;
typedef hpx::compute::vector<int, target_allocator> target_vector;

HPX_REGISTER_PARTITIONED_VECTOR(int, target_vector);


///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    // TODO: add more

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run")
        ;

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}
