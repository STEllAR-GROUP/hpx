//  Copyright (c) 2024
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>

// Include our thrust policy and algorithms
#include <hpx/async_cuda/thrust/policy.hpp>
#include <hpx/async_cuda/thrust/algorithms.hpp>

#include <vector>
#include <iostream>
#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main()
{
    // Performance test settings
    std::size_t size = 100000;  // Smaller size for faster testing
    int fill_value = 42;

    std::cout << "=== HPX-Thrust Policy Integration Test ===" << std::endl;
    std::cout << "Vector size: " << size << " elements" << std::endl;
    std::cout << "Testing thrust_host_policy and thrust_device_policy" << std::endl;

    // Test 1: CPU Sequential Fill (baseline)
    std::cout << "\n--- Test 1: CPU Sequential Fill (hpx::execution::seq) ---" << std::endl;
    std::vector<int> cpu_vec(size, 0);
    auto start_cpu = std::chrono::high_resolution_clock::now();
    hpx::fill(hpx::execution::seq, cpu_vec.begin(), cpu_vec.end(), fill_value);
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_cpu = stop_cpu - start_cpu;
    std::cout << "CPU execution time: " << duration_cpu.count() << " ms" << std::endl;
    bool success_cpu = (cpu_vec[0] == fill_value) && (cpu_vec[size - 1] == fill_value);
    std::cout << "CPU verification: " << (success_cpu ? "Success" : "Failed") << std::endl;

    // Test 2: Thrust Host Policy Fill
    std::cout << "\n--- Test 2: Thrust Host Policy Fill ---" << std::endl;
    hpx::async_cuda::thrust::thrust_host_policy host_policy{};
    std::vector<int> host_vec(size, 0);
    
    std::cout << "Using thrust_host_policy (CPU multicore via thrust::host)..." << std::endl;
    auto start_host = std::chrono::high_resolution_clock::now();
    hpx::fill(host_policy, host_vec.begin(), host_vec.end(), fill_value);
    auto stop_host = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_host = stop_host - start_host;
    std::cout << "Host policy execution time: " << duration_host.count() << " ms" << std::endl;
    bool success_host = (host_vec[0] == fill_value) && (host_vec[size - 1] == fill_value);
    std::cout << "Host policy verification: " << (success_host ? "Success" : "Failed") << std::endl;

    // Test 3: Thrust Device Policy Fill
    std::cout << "\n--- Test 3: Thrust Device Policy Fill ---" << std::endl;
    hpx::async_cuda::thrust::thrust_device_policy device_policy{};
    thrust::device_vector<int> device_vec(size, 0);
    
    std::cout << "Using thrust_device_policy (GPU via thrust::device)..." << std::endl;
    
    // Warm-up run to account for CUDA initialization overhead
    hpx::fill(device_policy, device_vec.begin(), device_vec.end(), 1);
    
    auto start_device = std::chrono::high_resolution_clock::now();
    hpx::fill(device_policy, device_vec.begin(), device_vec.end(), fill_value);
    auto stop_device = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_device = stop_device - start_device;
    std::cout << "Device policy execution time: " << duration_device.count() << " ms" << std::endl;
    
    // Verify GPU results by copying back to host
    int first_val = device_vec[0];
    int last_val = device_vec[size - 1];
    bool success_device = (first_val == fill_value) && (last_val == fill_value);
    std::cout << "Device policy verification: " << (success_device ? "Success" : "Failed") << std::endl;

    // Test 4: Global Instance Test
    std::cout << "\n--- Test 4: Global Instance Test ---" << std::endl;
    std::cout << "Testing global thrust_host and thrust_device instances..." << std::endl;
    
    // Test global thrust_host instance
    std::vector<int> global_host_vec(1000, 0);
    hpx::fill(hpx::async_cuda::thrust::thrust_host, global_host_vec.begin(), global_host_vec.end(), 99);
    bool global_host_success = (global_host_vec[0] == 99) && (global_host_vec[999] == 99);
    std::cout << "thrust_host global instance: " << (global_host_success ? "Success" : "Failed") << std::endl;
    
    // Test global thrust_device instance
    thrust::device_vector<int> global_device_vec(1000, 0);
    hpx::fill(hpx::async_cuda::thrust::thrust_device, global_device_vec.begin(), global_device_vec.end(), 88);
    int first_device = global_device_vec[0];
    int last_device = global_device_vec[999];
    bool global_device_success = (first_device == 88) && (last_device == 88);
    std::cout << "thrust_device global instance: " << (global_device_success ? "Success" : "Failed") << std::endl;

    // Test 5: Policy .get() method demonstration
    std::cout << "\n--- Test 5: Demonstrating .get() method ---" << std::endl;
    std::cout << "host_policy.get() returns thrust::host: " << std::boolalpha << 
        std::is_same_v<decltype(host_policy.get()), decltype(::thrust::host)> << std::endl;
    std::cout << "device_policy.get() returns thrust::device: " << std::boolalpha << 
        std::is_same_v<decltype(device_policy.get()), decltype(::thrust::device)> << std::endl;

    // Test 6: Inheritance verification
    std::cout << "\n--- Test 6: Inheritance Verification ---" << std::endl;
    std::cout << "thrust_host_policy inherits from thrust_policy: " << std::boolalpha <<
        std::is_base_of_v<hpx::async_cuda::thrust::thrust_policy, hpx::async_cuda::thrust::thrust_host_policy> << std::endl;
    std::cout << "thrust_device_policy inherits from thrust_policy: " << std::boolalpha <<
        std::is_base_of_v<hpx::async_cuda::thrust::thrust_policy, hpx::async_cuda::thrust::thrust_device_policy> << std::endl;

    // Summary
    std::cout << "\n=== Performance Comparison ===" << std::endl;
    std::cout << "CPU Sequential:    " << duration_cpu.count() << " ms" << std::endl;
    std::cout << "Thrust Host:       " << duration_host.count() << " ms" << std::endl;
    std::cout << "Thrust Device:     " << duration_device.count() << " ms" << std::endl;

    bool all_tests_passed = success_cpu && success_host && success_device && 
                           global_host_success && global_device_success;

    std::cout << "\n=== Test Results ===" << std::endl;
    std::cout << "Overall result: " << (all_tests_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;

    return all_tests_passed ? 0 : 1;
} 