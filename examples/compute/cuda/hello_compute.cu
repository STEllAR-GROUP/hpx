///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/include/compute.hpp>
#include <hpx/include/parallel_copy.hpp>

#include <hpx/hpx_init.hpp>

#include <numeric>
#include <iostream>
#include <string>
#include <vector>

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    int const N = 100;
    std::vector<int> h_A(N);
    std::vector<int> h_B(N);
    std::vector<int> h_C_ref(N);
    std::vector<int> h_C(N);

    std::iota(h_A.begin(), h_A.end(), (std::rand() % 100) + 2);
    std::iota(h_B.begin(), h_B.end(), (std::rand() % 100) + 2);

    std::transform(h_A.begin(), h_A.end(), h_B.begin(), h_C_ref.begin(),
        [](int a, int b) { return a + b; });

    typedef hpx::compute::cuda::allocator<int> allocator_type;

    hpx::compute::cuda::target target;
    allocator_type alloc(target);
    hpx::compute::vector<int, allocator_type> d_A(N, alloc);
    hpx::compute::vector<int, allocator_type> d_B(N, alloc);
    hpx::compute::vector<int, allocator_type> d_C(N, alloc);

    hpx::parallel::copy(
        hpx::parallel::execution::seq,
        h_A.begin(), h_A.end(), d_A.begin());
    hpx::parallel::copy(
        hpx::parallel::execution::seq,
        h_B.begin(), h_B.end(), d_B.begin());

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    hpx::compute::cuda::detail::launch(target, blocksPerGrid, threadsPerBlock,
        [=] __device__ (int* A, int* B, int* C) mutable
        {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if(i < N)
                C[i] = A[i] + B[i];
        }, d_A.device_data(), d_B.device_data(), d_C.device_data());


    hpx::parallel::copy(
        hpx::parallel::execution::seq,
        d_C.begin(), d_C.end(), h_C.begin());

    bool success = true;
    for(int i = 0; i < N; ++i)
    {
        if(h_C[i] != h_C_ref[i] || h_C[i] != h_A[i] + h_B[i])
        {
            std::cout << "Error at " << i << "\n";
            std::cout << h_C[i] << " != " << h_C_ref[i] << "\n";
            std::cout << h_C[i] << " != " << h_A[i] + h_B[i] << "\n";
            success = false;
        }
    }
    if(success)
    {
        std::cout << "Yay!\n";
    }

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
