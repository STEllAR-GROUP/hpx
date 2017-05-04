//  Copyright (c) 2017 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/compute.hpp>

#include <cstddef>
#include <iostream>

__global__ void dummy()
{}

int hpx_main(boost::program_options::variables_map& vm)
{
    std::size_t iterations = vm["iterations"].as<std::size_t>();

    // Get the cuda targets we want to run on
    hpx::compute::cuda::target target;

    // Create the executor
    hpx::compute::cuda::default_executor executor(target);

    {
        auto cuda_stream = target.native_handle().get_stream();
        hpx::util::high_resolution_timer timer;
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
        hpx::util::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            executor.execute([] HPX_DEVICE (){});
        }
        double elapsed = timer.elapsed();
        std::cout << "executor.execute([](){}):                            "
            << elapsed << '\n';
    }
    {
        hpx::util::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            executor.apply_execute([] HPX_DEVICE (){});
            target.synchronize();
        }
        double elapsed = timer.elapsed();
        std::cout << "executor.apply_execute([](){}) + synchronize:        "
            << elapsed << '\n';
    }
    {
        hpx::util::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            executor.apply_execute([] HPX_DEVICE (){});
            target.get_future().get();
        }
        double elapsed = timer.elapsed();
        std::cout << "executor.apply_execute([](){}) + get_future().get(): "
            << elapsed << '\n';
    }
    {
        hpx::util::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            executor.async_execute([] HPX_DEVICE (){}).get();
        }
        double elapsed = timer.elapsed();
        std::cout << "executor.async_execute([](){}).get():                "
            << elapsed << '\n';
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace boost::program_options;

    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");
    cmdline.add_options()
        (   "iterations",
            boost::program_options::value<std::size_t>()->default_value(1024),
            "number of iterations (default: 1024)")
        ;
    return hpx::init(cmdline, argc, argv);
}
