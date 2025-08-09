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
#include <numeric>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main()
{
    // Performance test settings
    std::size_t size = 10000;  // Reasonable size for testing
    int fill_value = 42;

    std::cout << "=== HPX-Thrust Universal Algorithm Dispatch Test ===" << std::endl;
    std::cout << "Vector size: " << size << " elements" << std::endl;
    std::cout << "Testing universal tag_invoke with multiple algorithms" << std::endl;

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

    // Test 2: Thrust Host Policy - Multiple Algorithms
    std::cout << "\n--- Test 2: Thrust Host Policy (Multi-Algorithm Test) ---" << std::endl;
    hpx::async_cuda::thrust::thrust_host_policy host_policy{};
    std::vector<int> host_vec(size, 0);
    std::vector<int> host_vec2(size, 1);
    std::vector<int> host_result(size, 0);
    
    std::cout << "🚀 Testing hpx::fill with thrust_host_policy..." << std::endl;
    auto start_host = std::chrono::high_resolution_clock::now();
    hpx::fill(host_policy, host_vec.begin(), host_vec.end(), fill_value);
    auto stop_host = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_host = stop_host - start_host;
    std::cout << "Host fill execution time: " << duration_host.count() << " ms" << std::endl;
    
    std::cout << "🚀 Testing hpx::copy with thrust_host_policy..." << std::endl;
    hpx::copy(host_policy, host_vec.begin(), host_vec.end(), host_result.begin());
    
    std::cout << "🚀 Testing hpx::transform with thrust_host_policy..." << std::endl;
    hpx::transform(host_policy, host_vec.begin(), host_vec.end(), host_vec2.begin(), 
                   host_result.begin(), [](int a, int b) { return a + b; });
    
    bool success_host = (host_vec[0] == fill_value) && (host_vec[size - 1] == fill_value) &&
                       (host_result[0] == fill_value + 1) && (host_result[size - 1] == fill_value + 1);
    std::cout << "Host policy verification: " << (success_host ? "Success" : "Failed") << std::endl;
    
    // Test 3: Thrust Device Policy - Multiple Algorithms
    std::cout << "\n--- Test 3: Thrust Device Policy (GPU Multi-Algorithm Test) ---" << std::endl;
    hpx::async_cuda::thrust::thrust_device_policy device_policy{};
    thrust::device_vector<int> device_vec(size, 0);
    thrust::device_vector<int> device_vec2(size, 2);
    thrust::device_vector<int> device_result(size, 0);
    
    std::cout << "🚀 Testing hpx::fill with thrust_device_policy..." << std::endl;
    
    // Warm-up run to account for CUDA initialization overhead
    hpx::fill(device_policy, device_vec.begin(), device_vec.end(), 1);
    
    auto start_device = std::chrono::high_resolution_clock::now();
    hpx::fill(device_policy, device_vec.begin(), device_vec.end(), fill_value);
    auto stop_device = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_device = stop_device - start_device;
    std::cout << "Device fill execution time: " << duration_device.count() << " ms" << std::endl;
    
    std::cout << "🚀 Testing hpx::copy with thrust_device_policy..." << std::endl;
    hpx::copy(device_policy, device_vec.begin(), device_vec.end(), device_result.begin());
    
    std::cout << "🚀 Testing hpx::transform with thrust_device_policy..." << std::endl;
    hpx::transform(device_policy, device_vec.begin(), device_vec.end(), device_vec2.begin(),
                   device_result.begin(), [] __device__ (int a, int b) { return a * b; });
    
    // Verify GPU results by copying back to host
    int first_val = device_vec[0];
    int last_val = device_vec[size - 1];
    int first_result = device_result[0];
    int last_result = device_result[size - 1];
    bool success_device = (first_val == fill_value) && (last_val == fill_value) &&
                         (first_result == fill_value * 2) && (last_result == fill_value * 2);
    std::cout << "Device policy verification: " << (success_device ? "Success" : "Failed") << std::endl;

    // Test 4: Test NEW algorithms that weren't available before
    std::cout << "\n--- Test 4: Testing Additional Algorithms ---" << std::endl;
    
    // Initialize test data
    thrust::host_vector<int> numbers(10);
    std::iota(numbers.begin(), numbers.end(), 1); // Fill with 1, 2, 3, ..., 10
    thrust::device_vector<int> d_numbers = numbers;
    
    std::cout << "Original data: ";
    for(int i = 0; i < 10; ++i) std::cout << numbers[i] << " ";
    std::cout << std::endl;
    
    std::cout << "🚀 Testing hpx::reverse with thrust_device_policy..." << std::endl;
    hpx::reverse(device_policy, d_numbers.begin(), d_numbers.end());
    
    // Copy back to see result
    thrust::host_vector<int> reversed_result = d_numbers;
    std::cout << "After reverse: ";
    for(int i = 0; i < 10; ++i) std::cout << reversed_result[i] << " ";
    std::cout << std::endl;
    
    std::cout << "🚀 Testing hpx::reduce with thrust_device_policy..." << std::endl;
    int sum = hpx::reduce(device_policy, d_numbers.begin(), d_numbers.end(), 0);
    std::cout << "Sum of reversed array: " << sum << " (expected: 55)" << std::endl;

    // Test 5: Global Instance Test
    std::cout << "\n--- Test 5: Global Instance Test ---" << std::endl;
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

    // Performance Comparison
    std::cout << "\n=== Performance Comparison ===" << std::endl;
    std::cout << "CPU Sequential:    " << duration_cpu.count() << " ms" << std::endl;
    std::cout << "Thrust Host:       " << duration_host.count() << " ms" << std::endl;
    std::cout << "Thrust Device:     " << duration_device.count() << " ms" << std::endl;

    bool all_tests_passed = success_cpu && success_host && success_device && 
                           global_host_success && global_device_success && 
                           (sum == 55); // Check reduce result

    std::cout << "\n=== Universal Algorithm Dispatch Results ===" << std::endl;
    std::cout << "✅ Algorithms tested: fill, copy, transform, reverse, reduce" << std::endl;
    std::cout << "✅ Policies tested: thrust_host_policy, thrust_device_policy" << std::endl;
    std::cout << "✅ Universal tag_invoke: " << (all_tests_passed ? "Working perfectly!" : "Some issues detected") << std::endl;
    std::cout << "Overall result: " << (all_tests_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;

    return all_tests_passed ? 0 : 1;
} 