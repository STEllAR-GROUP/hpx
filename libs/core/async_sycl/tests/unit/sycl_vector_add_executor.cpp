//  Copyright (c) 2022 Gregor Daiß
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <algorithm>
#include <exception>
#include <iostream> 
#include <string> 
#include <vector>

#include <hpx/hpx_init.hpp> 
#include <hpx/futures/future.hpp>
#include <hpx/local/future.hpp>
#if defined(HPX_HAVE_SYCL)
#include <hpx/async_sycl/sycl_future.hpp> 
#include <hpx/async_sycl/sycl_executor.hpp> 

#include "common/sycl_vector_add_test_utils.hpp"

#include <CL/sycl.hpp> 

// This vector_size leads to 610MB per buffer (size_t datatypes)
// Entire scenario thus requires about 1830MB device memory when running
constexpr size_t vector_size = 80000000;

/// Test executor async_execute with a vector_add example
void VectorAdd_test1(const std::vector<size_t>& a_vector,
    const std::vector<size_t>& b_vector, std::vector<size_t>& add_parallel)
{
    cl::sycl::range<1> num_items{a_vector.size()};
    {
        bool continuation_triggered = false;
        // buffers from host vectors
        cl::sycl::buffer a_buf(a_vector.data(), num_items);
        cl::sycl::buffer b_buf(b_vector.data(), num_items);
        cl::sycl::buffer add_buf(add_parallel.data(), num_items);

        // Create executor
        hpx::sycl::experimental::sycl_executor exec(
            cl::sycl::default_selector{});
        std::cout << "Running on device: "
                  << exec.get_device().get_info<cl::sycl::info::device::name>()
                  << std::endl;
        // use async_execute
        auto async_normal_fut = exec.async_execute(
            &cl::sycl::queue::submit, [&](cl::sycl::handler& h) {
                cl::sycl::accessor a(a_buf, h, cl::sycl::read_only);
                cl::sycl::accessor b(b_buf, h, cl::sycl::read_only);
                cl::sycl::accessor add(
                    add_buf, h, cl::sycl::write_only, cl::sycl::no_init);
                h.parallel_for(
                    num_items, [=](auto i) { add[i] = a[i] + b[i]; });
            });
        // Add contiuation
        auto continuation_future1 = async_normal_fut.then([&continuation_triggered](auto&& fut) {
            fut.get();
            std::cout << "OKAY: Continuation working!" << std::endl;
            continuation_triggered = true;
            return;
        });
        if (async_normal_fut.is_ready())
        {
            std::cerr
                << "ERROR: Async kernel launch future is immediately ready "
                << "(thus probably not asynchronous at at all)!" << std::endl;
            std::terminate();
        }
        else
        {
            std::cout << "OKAY: Kernel hpx::future is NOT ready immediately "
                         "after launch!"
                      << std::endl;
        }
        continuation_future1.get();
        //  Was the continuation triggered by get as well?
        if (!continuation_triggered)
        {
            std::cerr << "ERROR: Continuation was apparently not triggered, "
                         "despite calling get!"
                      << std::endl;
            std::terminate();
        }

        // NOTE about usage: according to the sycl specification (2020) section
        // 3.9.8, the entire thing will synchronize here, due to the buffers
        // being destroyed!
        //
        // Hence this implictly syncs everything, so we should use get on any
        // futures/continuations beforehand (or simply make sure that the sycl
        // buffers (a_buf, b_buf_ add_buf)
        // have a longer lifetime by moving them to another scope.
    }
}

/// Test executor post and get_future member method with a vector_add example
void VectorAdd_test2(const std::vector<size_t>& a_vector,
    const std::vector<size_t>& b_vector, std::vector<size_t>& add_parallel)
{
    cl::sycl::range<1> num_items{a_vector.size()};
    // buffers from host vectors
    cl::sycl::buffer a_buf(a_vector.data(), num_items);
    cl::sycl::buffer b_buf(b_vector.data(), num_items);
    cl::sycl::buffer add_buf(add_parallel.data(), num_items);

    // Test post and get_future methods
    hpx::sycl::experimental::sycl_executor exec(
        cl::sycl::default_selector{});
    std::cout << "Running on device: "
              << exec.get_device().get_info<cl::sycl::info::device::name>()
              << std::endl;
    // Launch kernel one-way
    exec.post(&cl::sycl::queue::submit, [&](cl::sycl::handler& h) {
        cl::sycl::accessor a(a_buf, h, cl::sycl::read_only);
        cl::sycl::accessor b(b_buf, h, cl::sycl::read_only);
        cl::sycl::accessor add(
            add_buf, h, cl::sycl::write_only, cl::sycl::no_init);
        h.parallel_for(num_items, [=](auto i) { add[i] = a[i] + b[i]; });
    });
    // NOTE: This is NOT the recommended way to get a future for a kernel
    // launch as exec.get_future needs to submit an internal dummy kernel
    // to get an event to create this future!
    //
    // It is instead recommended to use exec.async_execute to get a future!
    // exec.get_future is merely a convinience method in case we get a
    // command_queue from some third-party library and need a future for
    // the current point the queue
    auto my_manual_fut = exec.get_future();
    if (my_manual_fut.is_ready())
    {
        std::cerr << "ERROR: Manual get_future using internal dummy kernel "
                     "is immediately ready "
                  << "(thus probably not asynchronous at at all)!"
                  << std::endl;
        std::terminate();
    }
    else
    {
        std::cout << "OKAY: Manual get_future hpx::future is NOT ready "
                     "immediately "
                     "after launch!"
                  << std::endl;
    }
    my_manual_fut.get();
}

int hpx_main(int, char**)
{
    static_assert(vector_size >= 6, "vector_size unreasonably small");
    // Enable polling for the future
    hpx::sycl::experimental::detail::register_polling(hpx::resource::get_thread_pool(0));
    std::cout << "SYCL Future polling enabled!\n";
    std::cout << "SYCL language version: " << SYCL_LANGUAGE_VERSION << "\n";
    // Input vectors
    std::vector<size_t> a(vector_size), b(vector_size), add_parallel(vector_size);

    // Run tests
    std::cout << "Test async execute:" << std::endl;
    fill_vector_add_input(a, b, add_parallel);
    VectorAdd_test1(a, b, add_parallel);
    check_vector_add_results(a, b, add_parallel); 
    print_vector_results(a, b, add_parallel);
    std::cout << "Test post and get_future() member method:" << std::endl;
    fill_vector_add_input(a, b, add_parallel); // also resets result buffer
    VectorAdd_test1(a, b, add_parallel);
    check_vector_add_results(a, b, add_parallel);
    print_vector_results(a, b, add_parallel);

    // Cleanup
    std::cout << "Disabling SYCL future polling.\n";
    hpx::sycl::experimental::detail::unregister_polling(hpx::resource::get_thread_pool(0));
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}
#endif
