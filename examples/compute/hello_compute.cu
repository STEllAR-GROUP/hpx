///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/compute/cuda/target.hpp>
#include <hpx/compute/cuda/detail/launch.hpp>
#include <hpx/compute/vector.hpp>

#include <hpx/hpx_init.hpp>

#include <numeric>
#include <iostream>
#include <vector>

HPX_HOST_DEVICE
void cuda_main(unsigned int seed)
{
    hpx::compute::cuda::target target;

    int N = 100;
    hpx::compute::vector<int> h_A(N);
    hpx::compute::vector<int> h_B(N);
    hpx::compute::vector<int> h_C_ref(N);
    hpx::compute::vector<int> h_C(N);

    std::iota(h_A.begin(), h_A.end(), (std::rand() % 100) + 2);
    std::iota(h_B.begin(), h_B.end(), (std::rand() % 100) + 2);

    std::transform(h_A.begin(), h_A.end(), h_B.begin(), h_C_ref.begin(),
        [](int a, int b) { return a + b; });

    {
        hpx::compute::cuda::detail::scoped_active_target active(target);
        int *d_A = 0;
        int *d_B = 0;
        int *d_C = 0;

        cudaMalloc(&d_A, N * sizeof(int));
        cudaMalloc(&d_B, N * sizeof(int));
        cudaMalloc(&d_C, N * sizeof(int));

        cudaMemcpy(d_A, h_A.data(), N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), N * sizeof(int), cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        hpx::compute::cuda::detail::launch(target, blocksPerGrid, threadsPerBlock,
            [](int * A, int *B, int *C, int N) -> void
            {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                if(i < N)
                    C[i] = A[i] + B[i];
            },
            d_A, d_B, d_C, N);

        cudaMemcpy(h_C.data(), d_C, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(active.stream());
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

    }

    for(int i = 0; i < N; ++i)
    {
        if(h_C[i] != h_C_ref[i] || h_C[i] != h_A[i] + h_B[i])
        {
            std::cout << "Error at " << i << "\n";
        }
    }
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    cuda_main(seed);

    return hpx::finalize();
}

#include <string>

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
