//  Copyright (c) 2022 Gregor Daiﬂ
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// hpxinspect:noascii

#include <hpx/config.hpp>
#include <hpx/future.hpp>
#include <hpx/hpx_init.hpp>

#if defined(HPX_HAVE_SYCL)
#include <hpx/async_sycl/sycl_executor.hpp>
#include <hpx/async_sycl/sycl_future.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "common/sycl_vector_add_test_utils.hpp"

#include <CL/sycl.hpp>

constexpr size_t vectorsize = 200000000;
constexpr size_t num_bytes = vectorsize * sizeof(float);

int hpx_main(int, char*[])
{
    // Enable polling for the future
    hpx::sycl::experimental::detail::register_polling(
        hpx::resource::get_thread_pool(0));
    std::cout << "SYCL Future polling enabled!\n";
    std::cout << "SYCL language version: " << SYCL_LANGUAGE_VERSION << "\n";

    hpx::sycl::experimental::sycl_executor exec(cl::sycl::default_selector{});
    auto a = cl::sycl::malloc_device<float>(
        vectorsize, exec.get_device(), exec.get_context());
    auto b = cl::sycl::malloc_device<float>(
        vectorsize, exec.get_device(), exec.get_context());
    auto c = cl::sycl::malloc_device<float>(
        vectorsize, exec.get_device(), exec.get_context());
    auto a_host = cl::sycl::malloc_host<float>(vectorsize, exec.get_context());
    auto b_host = cl::sycl::malloc_host<float>(vectorsize, exec.get_context());
    auto c_host = cl::sycl::malloc_host<float>(vectorsize, exec.get_context());

    // Note: Parameter types needs to match exactly, otherwise the correct
    // function won't be found -- in this case we need to cast from float* to void*
    // otherwise the correct memset overload won't be found
    hpx::apply(
        exec, &cl::sycl::queue::memset, static_cast<void*>(c), 0, num_bytes);
    hpx::apply(exec, &cl::sycl::queue::memset, static_cast<void*>(a_host), 0,
        num_bytes);
    hpx::apply(exec, &cl::sycl::queue::memset, static_cast<void*>(b_host), 0,
        num_bytes);
    hpx::apply(exec, &cl::sycl::queue::memset, static_cast<void*>(c_host), 0,
        num_bytes);

    float aj = 1.0;
    float bj = 2.0;
    float cj = 0.0;
    const float scalar = 3.0;
    const auto reset_input_method = [=](cl::sycl::id<1> i) {
        a[i] = aj;
        b[i] = bj;
    };
    hpx::apply(exec, &cl::sycl::queue::parallel_for,
        cl::sycl::range<1>{vectorsize}, reset_input_method);

    const auto copy_step = [=](cl::sycl::id<1> i) { c[i] = a[i]; };
    const auto scale_step = [=](cl::sycl::id<1> i) { b[i] = scalar * c[i]; };
    const auto add_step = [=](cl::sycl::id<1> i) { c[i] = a[i] + b[i]; };
    const auto triad_step = [=](cl::sycl::id<1> i) {
        a[i] = b[i] + scalar * c[i];
    };

    hpx::apply(exec, &cl::sycl::queue::parallel_for,
        cl::sycl::range<1>{vectorsize}, copy_step);
    hpx::apply(exec, &cl::sycl::queue::parallel_for,
        cl::sycl::range<1>{vectorsize}, scale_step);
    hpx::apply(exec, &cl::sycl::queue::parallel_for,
        cl::sycl::range<1>{vectorsize}, add_step);
    hpx::apply(exec, &cl::sycl::queue::parallel_for,
        cl::sycl::range<1>{vectorsize}, triad_step);

    // Note: Parameter types needs to match exactly, otherwise the correct
    // function won't be found -- in this case we need to cast from float* to const float*
    // otherwise the correct copy overload won't be found
    hpx::apply(exec, &cl::sycl::queue::copy, static_cast<const float*>(c),
        c_host, vectorsize);
    hpx::apply(exec, &cl::sycl::queue::copy, static_cast<const float*>(b),
        b_host, vectorsize);
    auto fut = hpx::async(exec, &cl::sycl::queue::copy,
        static_cast<const float*>(a), a_host, vectorsize);

    fut.get();

    cj = aj;
    bj = scalar * cj;
    cj = aj + bj;
    aj = bj + scalar * cj;

    float aSumErr = 0.0;
    float bSumErr = 0.0;
    float cSumErr = 0.0;
    for (std::size_t j = 0; j < vectorsize; j++)
    {
        aSumErr += std::abs(a_host[j] - aj);
        bSumErr += std::abs(b_host[j] - bj);
        cSumErr += std::abs(c_host[j] - cj);
    }
    const float aAvgErr = aSumErr / static_cast<float>(vectorsize);
    const float bAvgErr = bSumErr / static_cast<float>(vectorsize);
    const float cAvgErr = cSumErr / static_cast<float>(vectorsize);
    float epsilon = 1.e-6;
    if (std::abs(aAvgErr / aj) > epsilon)
    {
        std::cerr << "Validation error! Wrong results in array a!" << std::endl;
        std::terminate();
    }
    if (std::abs(bAvgErr / bj) > epsilon)
    {
        std::cerr << "Validation error! Wrong results in array b!" << std::endl;
        std::terminate();
    }
    if (std::abs(cAvgErr / cj) > epsilon)
    {
        std::cerr << "Validation error! Wrong results in array c!" << std::endl;
        std::terminate();
    }
    std::cerr << "Validation passed!" << std::endl;

    // Cleanup
    std::cout << "\nDisabling SYCL future polling.\n";
    hpx::sycl::experimental::detail::unregister_polling(
        hpx::resource::get_thread_pool(0));
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}
#else
#include <iostream>

// Handle none-sycl builds
int main()
{
    std::cerr << "SYCL Support was not given at compile time! " << std::endl;
    std::cerr << "Please check your build configuration!" << std::endl;
    std::cerr << "Exiting..." << std::endl;
    return 1;    // Fail test, as it was meant to test SYCL
}
#endif
