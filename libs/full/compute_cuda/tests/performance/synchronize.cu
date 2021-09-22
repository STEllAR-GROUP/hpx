//  Copyright (c) 2017 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/chrono.hpp>
#include <hpx/local/execution.hpp>
#include <hpx/local/init.hpp>
#include <hpx/modules/async_cuda.hpp>
#include <hpx/modules/compute_cuda.hpp>

#include <cstddef>
#include <iostream>

__global__ void dummy() {}

int hpx_main(hpx::program_options::variables_map& vm)
{
    std::size_t const iterations = vm["iterations"].as<std::size_t>();
    std::size_t const batch_size = 10;
    std::size_t const batch_iterations = iterations / batch_size;
    std::size_t const non_batch_iterations = iterations % batch_size;

    // Get the cuda targets we want to run on
    hpx::cuda::experimental::target target;

    // Create the executor
    hpx::cuda::experimental::default_executor executor(target);

    // Warmup
    {
        auto cuda_stream = target.native_handle().get_stream();
        hpx::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            dummy<<<1, 1, 0, cuda_stream>>>();
            hpx::cuda::experimental::check_cuda_error(
                cudaStreamSynchronize(cuda_stream));
        }
        double elapsed = timer.elapsed();
        std::cout
            << "native + synchronize (warmup):                                 "
            << elapsed << '\n';
    }

    {
        auto cuda_stream = target.native_handle().get_stream();
        hpx::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            dummy<<<1, 1, 0, cuda_stream>>>();
            hpx::cuda::experimental::check_cuda_error(
                cudaStreamSynchronize(cuda_stream));
        }
        double elapsed = timer.elapsed();
        std::cout
            << "native + synchronize:                                          "
            << elapsed << '\n';
    }

    {
        auto cuda_stream = target.native_handle().get_stream();

        hpx::chrono::high_resolution_timer timer;

        for (std::size_t i = 0; i < batch_iterations; ++i)
        {
            for (std::size_t b = 0; b < batch_size; ++b)
            {
                dummy<<<1, 1, 0, cuda_stream>>>();
            }
            hpx::cuda::experimental::check_cuda_error(
                cudaStreamSynchronize(cuda_stream));
        }

        for (std::size_t i = 0; i < non_batch_iterations; ++i)
        {
            dummy<<<1, 1, 0, cuda_stream>>>();
        }
        hpx::cuda::experimental::check_cuda_error(
            cudaStreamSynchronize(cuda_stream));

        double elapsed = timer.elapsed();
        std::cout
            << "native + synchronize batched:                                  "
            << elapsed << '\n';
    }

    {
        hpx::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            executor.sync_execute([] HPX_DEVICE() {});
        }
        double elapsed = timer.elapsed();
        std::cout
            << "executor.sync_execute([](){}):                                 "
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
        std::cout
            << "executor.post([](){}) + synchronize:                           "
            << elapsed << '\n';
    }

    {
        hpx::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i < batch_iterations; ++i)
        {
            for (std::size_t b = 0; b < batch_size; ++b)
            {
                executor.post([] HPX_DEVICE() {});
            }
            target.synchronize();
        }

        for (std::size_t i = 0; i < non_batch_iterations; ++i)
        {
            executor.post([] HPX_DEVICE() {});
        }
        target.synchronize();

        double elapsed = timer.elapsed();
        std::cout
            << "executor.post([](){}) + synchronize batched:                   "
            << elapsed << '\n';
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
            << "executor.post([](){}) + get_future() callback:                 "
            << elapsed << '\n';
    }

    {
        hpx::chrono::high_resolution_timer timer;

        for (std::size_t i = 0; i < batch_iterations; ++i)
        {
            for (std::size_t b = 0; b < batch_size; ++b)
            {
                executor.post([] HPX_DEVICE() {});
            }
            target.get_future_with_callback().get();
        }

        for (std::size_t i = 0; i < non_batch_iterations; ++i)
        {
            executor.post([] HPX_DEVICE() {});
        }
        target.get_future_with_callback().get();

        double elapsed = timer.elapsed();
        std::cout
            << "executor.post([](){}) + get_future() callback batched:         "
            << elapsed << '\n';
    }

    {
        hpx::cuda::experimental::enable_user_polling poll("default");

        hpx::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            executor.post([] HPX_DEVICE() {});
            target.get_future_with_event().get();
        }
        double elapsed = timer.elapsed();
        std::cout
            << "executor.post([](){}) + get_future() event:                    "
            << elapsed << '\n';
    }

    {
        hpx::cuda::experimental::enable_user_polling poll("default");

        hpx::chrono::high_resolution_timer timer;

        for (std::size_t i = 0; i < batch_iterations; ++i)
        {
            for (std::size_t b = 0; b < batch_size; ++b)
            {
                executor.post([] HPX_DEVICE() {});
            }
            target.get_future_with_event().get();
        }

        for (std::size_t i = 0; i < non_batch_iterations; ++i)
        {
            executor.post([] HPX_DEVICE() {});
        }
        target.get_future_with_event().get();

        double elapsed = timer.elapsed();
        std::cout
            << "executor.post([](){}) + get_future() event batched:            "
            << elapsed << '\n';
    }

    {
        hpx::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            executor.async_execute([] HPX_DEVICE() {}).get();
        }
        double elapsed = timer.elapsed();
        std::cout
            << "executor.async_execute([](){}).get():                          "
            << elapsed << '\n';
    }

    {
        hpx::chrono::high_resolution_timer timer;

        for (std::size_t i = 0; i < batch_iterations; ++i)
        {
            hpx::future<void> f;
            for (std::size_t b = 0; b < batch_size; ++b)
            {
                f = executor.async_execute([] HPX_DEVICE() {});
            }
            f.get();
        }

        hpx::future<void> f;
        for (std::size_t i = 0; i < non_batch_iterations; ++i)
        {
            f = executor.async_execute([] HPX_DEVICE() {});
        }
        f.get();

        double elapsed = timer.elapsed();
        std::cout
            << "executor.async_execute([](){}).get() batched:                  "
            << elapsed << '\n';
    }

    {
        hpx::cuda::experimental::enable_user_polling poll("default");

        namespace ex = hpx::execution::experimental;
        namespace cu = hpx::cuda::experimental;

        auto const f = [](cudaStream_t cuda_stream) {
            dummy<<<1, 1, 0, cuda_stream>>>();
        };

        auto cuda_stream = target.native_handle().get_stream();

        hpx::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            cu::transform_stream(ex::just(), f, cuda_stream) | ex::sync_wait();
        }
        double elapsed = timer.elapsed();
        std::cout
            << "transform_stream:                                              "
            << elapsed << '\n';
    }

    {
        hpx::cuda::experimental::enable_user_polling poll("default");

        namespace ex = hpx::execution::experimental;
        namespace cu = hpx::cuda::experimental;

        auto const f = [](cudaStream_t cuda_stream) {
            dummy<<<1, 1, 0, cuda_stream>>>();
        };

        auto cuda_stream = target.native_handle().get_stream();

        hpx::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i < batch_iterations; ++i)
        {
            // We have to manually unroll this loop, because the type of the
            // sender changes for each additional transform_stream call. The
            // number of unrolled calls must match batch_size above.
            cu::transform_stream(ex::just(), f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) | ex::sync_wait();
        }
        // Do the remainder one-by-one
        for (std::size_t i = 0; i < non_batch_iterations; ++i)
        {
            cu::transform_stream(ex::just(), f, cuda_stream) | ex::sync_wait();
        }
        double elapsed = timer.elapsed();
        std::cout
            << "transform_stream batched:                                      "
            << elapsed << '\n';
    }

    {
        hpx::cuda::experimental::enable_user_polling poll("default");

        namespace ex = hpx::execution::experimental;
        namespace cu = hpx::cuda::experimental;

        auto const f = [](cudaStream_t cuda_stream) {
            dummy<<<1, 1, 0, cuda_stream>>>();
        };

        auto cuda_stream = target.native_handle().get_stream();

        hpx::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i < batch_iterations; ++i)
        {
            // We have to manually unroll this loop, because the type of the
            // sender changes for each additional transform_stream call. The
            // number of unrolled calls must match batch_size above. Here we
            // intentionally insert dummy transform([]{}) calls between the
            // transform_stream calls to force synchronization between the
            // kernel launches.
            cu::transform_stream(ex::just(), f, cuda_stream) |
                ex::transform([] {}) | cu::transform_stream(f, cuda_stream) |
                ex::transform([] {}) | cu::transform_stream(f, cuda_stream) |
                ex::transform([] {}) | cu::transform_stream(f, cuda_stream) |
                ex::transform([] {}) | cu::transform_stream(f, cuda_stream) |
                ex::transform([] {}) | cu::transform_stream(f, cuda_stream) |
                ex::transform([] {}) | cu::transform_stream(f, cuda_stream) |
                ex::transform([] {}) | cu::transform_stream(f, cuda_stream) |
                ex::transform([] {}) | cu::transform_stream(f, cuda_stream) |
                ex::transform([] {}) | cu::transform_stream(f, cuda_stream) |
                ex::sync_wait();
        }
        // Do the remainder one-by-one
        for (std::size_t i = 0; i < non_batch_iterations; ++i)
        {
            cu::transform_stream(ex::just(), f, cuda_stream) | ex::sync_wait();
        }
        double elapsed = timer.elapsed();
        std::cout
            << "transform_stream force synchronize batched:                    "
            << elapsed << '\n';
    }

    {
        hpx::cuda::experimental::enable_user_polling poll("default");

        namespace ex = hpx::execution::experimental;
        namespace cu = hpx::cuda::experimental;

        auto const f = [](cudaStream_t cuda_stream) {
            dummy<<<1, 1, 0, cuda_stream>>>();
        };

        auto cuda_stream = target.native_handle().get_stream();

        hpx::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            cu::transform_stream(ex::just(), f, cuda_stream) |
                ex::on(ex::thread_pool_scheduler{}) | ex::sync_wait();
        }
        double elapsed = timer.elapsed();
        std::cout
            << "transform_stream with on:                                      "
            << elapsed << '\n';
    }

    {
        hpx::cuda::experimental::enable_user_polling poll("default");

        namespace ex = hpx::execution::experimental;
        namespace cu = hpx::cuda::experimental;

        auto const f = [](cudaStream_t cuda_stream) {
            dummy<<<1, 1, 0, cuda_stream>>>();
        };

        auto cuda_stream = target.native_handle().get_stream();

        hpx::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i < batch_iterations; ++i)
        {
            // We have to manually unroll this loop, because the type of the
            // sender changes for each additional transform_stream call. The
            // number of unrolled calls must match batch_size above.
            cu::transform_stream(ex::just(), f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                ex::on(ex::thread_pool_scheduler{}) | ex::sync_wait();
        }
        // Do the remainder one-by-one
        for (std::size_t i = 0; i < non_batch_iterations; ++i)
        {
            cu::transform_stream(ex::just(), f, cuda_stream) |
                ex::on(ex::thread_pool_scheduler{}) | ex::sync_wait();
        }
        double elapsed = timer.elapsed();
        std::cout
            << "transform_stream with on batched:                              "
            << elapsed << '\n';
    }

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    using namespace hpx::program_options;

    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");
    cmdline.add_options()("iterations",
        hpx::program_options::value<std::size_t>()->default_value(1024),
        "number of iterations (default: 1024)");
    hpx::local::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return hpx::local::init(hpx_main, argc, argv, init_args);
}
