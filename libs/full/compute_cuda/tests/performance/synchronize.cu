//  Copyright (c) 2017 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/compute.hpp>

#include <cstddef>
#include <iostream>

__global__ void dummy() {}

int hpx_main(hpx::program_options::variables_map& vm)
{
    std::size_t iterations = vm["iterations"].as<std::size_t>();

    // Get the cuda targets we want to run on
    hpx::cuda::experimental::target target;

    // Create the executor
    hpx::cuda::experimental::default_executor executor(target);

    {
        auto cuda_stream = target.native_handle().get_stream();
        hpx::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            dummy<<<1, 1, 0, cuda_stream>>>();
            cudaStreamSynchronize(cuda_stream);
        }
        double elapsed = timer.elapsed();
        std::cout << "native + synchronize:                                "
                  << elapsed << '\n';
    }

    {
        hpx::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            executor.sync_execute([] HPX_DEVICE() {});
        }
        double elapsed = timer.elapsed();
        std::cout << "executor.execute([](){}):                            "
                  << elapsed << '\n';
    }
    {
        hpx::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            executor.post([] HPX_DEVICE() {});
            target.synchronize();
        }
        double elapsed = timer.elapsed();
        std::cout << "executor.post([](){}) + synchronize:        " << elapsed
                  << '\n';
    }
    {
        hpx::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            executor.post([] HPX_DEVICE() {});
            target.get_future_with_callback().get();
        }
        double elapsed = timer.elapsed();
        std::cout
            << "executor.post([](){}) + get_future_with_callback().get(): "
            << elapsed << '\n';
    }
    {
        hpx::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            executor.async_execute([] HPX_DEVICE() {}).get();
        }
        double elapsed = timer.elapsed();
        std::cout << "executor.async_execute([](){}).get():                "
                  << elapsed << '\n';
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace hpx::program_options;

    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");
    cmdline.add_options()("iterations",
        hpx::program_options::value<std::size_t>()->default_value(1024),
        "number of iterations (default: 1024)");
    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return hpx::init(argc, argv, init_args);
}
