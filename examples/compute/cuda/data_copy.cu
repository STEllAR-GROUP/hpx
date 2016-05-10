//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/compute/cuda.hpp>
#include <hpx/compute/vector.hpp>

#include <hpx/parallel/algorithms/copy.hpp>

#include <hpx/hpx_init.hpp>

#include <iostream>
#include <numeric>
#include <string>
#include <vector>

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    // create data vector on host
    int const N = 100;
    std::vector<int> h_A(N);
    std::vector<int> h_B(N);
    std::iota(h_A.begin(), h_A.end(), (std::rand() % 100) + 2);

    hpx::compute::cuda::target target;

    // create data vector on device
    typedef hpx::compute::cuda::allocator<int> allocator_type;
    allocator_type alloc(target);

    hpx::compute::vector<int, allocator_type> d_A(N, alloc);

    hpx::future<void> f = target.get_future();

    f.get();

    // copy data from host to device
    hpx::parallel::copy(
        hpx::parallel::par,
        h_A.begin(), h_A.end(), d_A.begin());

    // copy data from device to host
    hpx::parallel::copy(
        hpx::parallel::par,
        d_A.begin(), d_A.end(), h_B.begin());

    if(std::equal(h_A.begin(), h_A.end(), h_B.begin()))
        std::cout << "Copy succeeded!\n";
    else
        std::cout << "Copy not successful :(\n";

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
