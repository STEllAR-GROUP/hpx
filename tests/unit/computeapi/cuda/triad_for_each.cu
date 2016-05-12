///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/include/compute.hpp>
#include <hpx/include/parallel_for_loop.hpp>
#include <hpx/include/parallel_copy.hpp>

#include <hpx/hpx_init.hpp>

#include <numeric>
#include <iostream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
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

    std::transform(
        h_A.begin(), h_A.end(), h_B.begin(), h_C_ref.begin(),
        [](int a, int b) { return a + b; });

    // define execution target (here device 0)
    hpx::compute::cuda::target target;

    // allocate data
    typedef hpx::compute::cuda::allocator<int> allocator_type;
    allocator_type alloc(target);

    hpx::compute::vector<int, allocator_type> d_A(N, alloc);
    hpx::compute::vector<int, allocator_type> d_B(N, alloc);
    hpx::compute::vector<int, allocator_type> d_C(N, alloc);

    // copy data to device
    hpx::parallel::copy(
        hpx::parallel::par,
        h_A.begin(), h_A.end(), d_A.begin());
    hpx::parallel::copy(
        hpx::parallel::par,
        h_B.begin(), h_B.end(), d_B.begin());

    // create executor
    typedef hpx::compute::cuda::default_executor executor_type;
    executor_type exec(target);

    hpx::parallel::for_loop_n(
        hpx::parallel::par.on(exec),
        d_A.data(), d_A.size(),
        hpx::parallel::induction(d_B.data()),
        hpx::parallel::induction(d_C.data()),
        [] __device__ (int* A, int* B, int* C)
        {
            *C = *A + *B;
        });

    hpx::parallel::copy(
        hpx::parallel::par,
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
