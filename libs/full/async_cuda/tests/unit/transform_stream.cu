//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/init.hpp>
#include <hpx/modules/async_cuda.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/executors.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <cstddef>
#include <utility>

__global__ void dummy_kernel() {}

struct dummy
{
    static std::atomic<std::size_t> host_void_calls;
    static std::atomic<std::size_t> stream_void_calls;
    static std::atomic<std::size_t> host_int_calls;
    static std::atomic<std::size_t> stream_int_calls;
    static std::atomic<std::size_t> host_double_calls;
    static std::atomic<std::size_t> stream_double_calls;

    static void reset_counts()
    {
        host_void_calls = 0;
        stream_void_calls = 0;
        host_int_calls = 0;
        stream_int_calls = 0;
        host_double_calls = 0;
        stream_double_calls = 0;
    }

    void operator()() const
    {
        ++host_void_calls;
    }

    void operator()(cudaStream_t stream) const
    {
        ++stream_void_calls;
        dummy_kernel<<<1, 1, 0, stream>>>();
    }

    double operator()(int x) const
    {
        ++host_int_calls;
        return x + 1;
    }

    double operator()(int x, cudaStream_t stream) const
    {
        ++stream_int_calls;
        dummy_kernel<<<1, 1, 0, stream>>>();
        return x + 1;
    }

    int operator()(double x) const
    {
        ++host_double_calls;
        return x + 1;
    }

    int operator()(double x, cudaStream_t stream) const
    {
        ++stream_double_calls;
        dummy_kernel<<<1, 1, 0, stream>>>();
        return x + 1;
    }
};

std::atomic<std::size_t> dummy::host_void_calls{0};
std::atomic<std::size_t> dummy::stream_void_calls{0};
std::atomic<std::size_t> dummy::host_int_calls{0};
std::atomic<std::size_t> dummy::stream_int_calls{0};
std::atomic<std::size_t> dummy::host_double_calls{0};
std::atomic<std::size_t> dummy::stream_double_calls{0};

__global__ void increment_kernel(int* p)
{
    ++(*p);
}

struct increment
{
    int* operator()(int* p, cudaStream_t stream) const
    {
        increment_kernel<<<1, 1, 0, stream>>>(p);
        return p;
    }
};

struct cuda_memcpy_async
{
    template <typename... Ts>
    auto operator()(Ts&&... ts)
    {
        return cudaMemcpyAsync(std::forward<Ts>(ts)...);
    }
};

int hpx_main()
{
    namespace cu = ::hpx::cuda::experimental;
    namespace ex = ::hpx::execution::experimental;

    cu::enable_user_polling p;

    // Only stream transform
    {
        dummy::reset_counts();
        auto s1 = ex::just();
        auto s2 = cu::transform_stream(std::move(s1), dummy{});
        // NOTE: transform_stream calls triggers the receiver on a plain
        // std::thread. We explicitly change the context back to an hpx::thread.
        ex::sync_wait(ex::on(std::move(s2), ex::thread_pool_scheduler{}));
        HPX_TEST_EQ(dummy::host_void_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::stream_void_calls.load(), std::size_t(1));
        HPX_TEST_EQ(dummy::host_int_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::stream_int_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::host_double_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::stream_double_calls.load(), std::size_t(0));
    }

    {
        dummy::reset_counts();
        auto s1 = ex::just();
        auto s2 = cu::transform_stream(std::move(s1), dummy{});
        auto s3 = cu::transform_stream(std::move(s2), dummy{});
        auto s4 = cu::transform_stream(std::move(s3), dummy{});
        ex::sync_wait(ex::on(std::move(s4), ex::thread_pool_scheduler{}));
        HPX_TEST_EQ(dummy::host_void_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::stream_void_calls.load(), std::size_t(3));
        HPX_TEST_EQ(dummy::host_int_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::stream_int_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::host_double_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::stream_double_calls.load(), std::size_t(0));
    }

    // Mixing stream transform with host scheduler
    {
        dummy::reset_counts();
        auto s1 = ex::just();
        auto s2 = cu::transform_stream(std::move(s1), dummy{});
        auto s3 = ex::on(std::move(s2), ex::thread_pool_scheduler{});
        auto s4 = ex::transform(std::move(s3), dummy{});
        auto s5 = cu::transform_stream(std::move(s4), dummy{});
        ex::sync_wait(ex::on(std::move(s5), ex::thread_pool_scheduler{}));
        HPX_TEST_EQ(dummy::host_void_calls.load(), std::size_t(1));
        HPX_TEST_EQ(dummy::stream_void_calls.load(), std::size_t(2));
        HPX_TEST_EQ(dummy::host_int_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::stream_int_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::host_double_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::stream_double_calls.load(), std::size_t(0));
    }

    {
        dummy::reset_counts();
        auto s1 = ex::schedule(ex::thread_pool_scheduler{});
        auto s2 = ex::transform(std::move(s1), dummy{});
        auto s3 = cu::transform_stream(std::move(s2), dummy{});
        auto s4 = ex::on(std::move(s3), ex::thread_pool_scheduler{});
        auto s5 = ex::transform(std::move(s4), dummy{});
        ex::sync_wait(std::move(s5));
        HPX_TEST_EQ(dummy::host_void_calls.load(), std::size_t(2));
        HPX_TEST_EQ(dummy::stream_void_calls.load(), std::size_t(1));
        HPX_TEST_EQ(dummy::host_int_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::stream_int_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::host_double_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::stream_double_calls.load(), std::size_t(0));
    }

    // Only stream transform with non-void values
    {
        dummy::reset_counts();
        auto s1 = ex::just(1);
        auto s2 = cu::transform_stream(std::move(s1), dummy{});
        HPX_TEST_EQ(
            ex::sync_wait(ex::on(std::move(s2), ex::thread_pool_scheduler{})),
            2.0);
        HPX_TEST_EQ(dummy::host_void_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::stream_void_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::host_int_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::stream_int_calls.load(), std::size_t(1));
        HPX_TEST_EQ(dummy::host_double_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::stream_double_calls.load(), std::size_t(0));
    }

    {
        dummy::reset_counts();
        auto s1 = ex::just(1);
        auto s2 = cu::transform_stream(std::move(s1), dummy{});
        auto s3 = cu::transform_stream(std::move(s2), dummy{});
        auto s4 = cu::transform_stream(std::move(s3), dummy{});
        HPX_TEST_EQ(
            ex::sync_wait(ex::on(std::move(s4), ex::thread_pool_scheduler{})),
            4.0);
        HPX_TEST_EQ(dummy::host_void_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::stream_void_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::host_int_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::stream_int_calls.load(), std::size_t(2));
        HPX_TEST_EQ(dummy::host_double_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::stream_double_calls.load(), std::size_t(1));
    }

    // Mixing stream transform with host scheduler with non-void values
    {
        dummy::reset_counts();
        auto s1 = ex::just(1);
        auto s2 = cu::transform_stream(std::move(s1), dummy{});
        auto s3 = ex::on(std::move(s2), ex::thread_pool_scheduler{});
        auto s4 = ex::transform(std::move(s3), dummy{});
        auto s5 = cu::transform_stream(std::move(s4), dummy{});
        HPX_TEST_EQ(
            ex::sync_wait(ex::on(std::move(s5), ex::thread_pool_scheduler{})),
            4.0);
        HPX_TEST_EQ(dummy::host_void_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::stream_void_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::host_int_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::stream_int_calls.load(), std::size_t(2));
        HPX_TEST_EQ(dummy::host_double_calls.load(), std::size_t(1));
        HPX_TEST_EQ(dummy::stream_double_calls.load(), std::size_t(0));
    }

    {
        dummy::reset_counts();
        auto s1 = ex::just(1);
        auto s2 = ex::on(std::move(s1), ex::thread_pool_scheduler{});
        auto s3 = ex::transform(std::move(s2), dummy{});
        auto s4 = cu::transform_stream(std::move(s3), dummy{});
        auto s5 = ex::on(std::move(s4), ex::thread_pool_scheduler{});
        auto s6 = ex::transform(std::move(s5), dummy{});
        HPX_TEST_EQ(ex::sync_wait(std::move(s6)), 4.0);
        HPX_TEST_EQ(dummy::host_void_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::stream_void_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::host_int_calls.load(), std::size_t(2));
        HPX_TEST_EQ(dummy::stream_int_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::host_double_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::stream_double_calls.load(), std::size_t(1));
    }

    {
        dummy::reset_counts();
        auto s1 = ex::just_on(ex::thread_pool_scheduler{}, 1);
        auto s2 = ex::transform(std::move(s1), dummy{});
        auto s3 = cu::transform_stream(std::move(s2), dummy{});
        auto s4 = cu::transform_stream(std::move(s3), dummy{});
        auto s5 = ex::on(std::move(s4), ex::thread_pool_scheduler{});
        auto s6 = ex::transform(std::move(s5), dummy{});
        HPX_TEST_EQ(ex::sync_wait(std::move(s6)), 5.0);
        HPX_TEST_EQ(dummy::host_void_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::stream_void_calls.load(), std::size_t(0));
        HPX_TEST_EQ(dummy::host_int_calls.load(), std::size_t(1));
        HPX_TEST_EQ(dummy::stream_int_calls.load(), std::size_t(1));
        HPX_TEST_EQ(dummy::host_double_calls.load(), std::size_t(1));
        HPX_TEST_EQ(dummy::stream_double_calls.load(), std::size_t(1));
    }

    // Chaining multiple stream transforms without intermediate synchronization
    {
        using type = int;
        type p_h = 0;

        type* p;
        cu::check_cuda_error(cudaMalloc((void**) &p, sizeof(type)));

        auto s = ex::just(p, &p_h, sizeof(type), cudaMemcpyHostToDevice) |
            cu::transform_stream(cuda_memcpy_async{}) |
            ex::transform(&cu::check_cuda_error) |
            ex::transform([p] { return p; }) |
            cu::transform_stream(increment{}) |
            cu::transform_stream(increment{}) |
            cu::transform_stream(increment{});
        ex::when_all(ex::just(&p_h), std::move(s), ex::just(sizeof(type)),
            ex::just(cudaMemcpyDeviceToHost)) |
            cu::transform_stream(cuda_memcpy_async{}) |
            ex::transform(&cu::check_cuda_error) |
            ex::transform([&p_h] { HPX_TEST_EQ(p_h, 3); }) |
            ex::on(ex::thread_pool_scheduler{}) | ex::sync_wait();

        cu::check_cuda_error(cudaFree(p));
    }

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
