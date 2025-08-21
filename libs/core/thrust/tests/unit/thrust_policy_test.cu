#include <hpx/config.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/thrust/algorithms.hpp>
#include <hpx/thrust/policy.hpp>
#include <hpx/execution.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/futures.hpp>

#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

#include <hpx/async_cuda/cuda_polling_helper.hpp>
#include <hpx/async_cuda/custom_gpu_api.hpp>
#include <hpx/async_cuda/target.hpp>
#include <hpx/modules/async_cuda.hpp>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int hpx_main(hpx::program_options::variables_map&)
{
    // Enable CUDA event polling on the "default" pool for HPX <-> CUDA future bridging
    hpx::cuda::experimental::enable_user_polling polling_guard("default");

    std::size_t size = 10000;
    int fill_value = 42;

    std::cout << "=== HPX-Thrust Universal Algorithm Dispatch Test ==="
              << std::endl;
    std::cout << "Vector size: " << size << " elements" << std::endl;

    // Test 1: CPU Sequential Fill
    std::cout << "\n--- Test 1: CPU Sequential Fill (hpx::execution::seq) ---"
              << std::endl;
    std::vector<int> cpu_vec(size, 0);
    auto start_cpu = std::chrono::high_resolution_clock::now();
    hpx::fill(hpx::execution::seq, cpu_vec.begin(), cpu_vec.end(), fill_value);
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_cpu =
        stop_cpu - start_cpu;
    std::cout << "CPU execution time: " << duration_cpu.count() << " ms"
              << std::endl;
    bool success_cpu =
        (cpu_vec[0] == fill_value) && (cpu_vec[size - 1] == fill_value);
    std::cout << "CPU verification: " << (success_cpu ? "Success" : "Failed")
              << std::endl;

    // Test 2: Thrust Host Policy
    std::cout << "\n--- Test 2: Thrust Host Policy ---" << std::endl;
    hpx::thrust::thrust_host_policy host_policy{};
    std::vector<int> host_vec(size, 0);
    std::vector<int> host_vec2(size, 1);
    std::vector<int> host_result(size, 0);

    std::cout << "Testing hpx::fill with thrust_host_policy..." << std::endl;
    auto start_host = std::chrono::high_resolution_clock::now();
    hpx::fill(host_policy, host_vec.begin(), host_vec.end(), fill_value);
    auto stop_host = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_host =
        stop_host - start_host;
    std::cout << "Host fill execution time: " << duration_host.count() << " ms"
              << std::endl;

    std::cout << "Testing hpx::copy with thrust_host_policy..." << std::endl;
    hpx::copy(
        host_policy, host_vec.begin(), host_vec.end(), host_result.begin());

    std::cout << " Testing hpx::transform with thrust_host_policy..."
              << std::endl;
    hpx::transform(host_policy, host_vec.begin(), host_vec.end(),
        host_vec2.begin(), host_result.begin(),
        [](int a, int b) { return a + b; });

    bool success_host = (host_vec[0] == fill_value) &&
        (host_vec[size - 1] == fill_value) &&
        (host_result[0] == fill_value + 1) &&
        (host_result[size - 1] == fill_value + 1);
    std::cout << "Host policy verification: "
              << (success_host ? "Success" : "Failed") << std::endl;

    // Test 3: Thrust Device Policy - Multiple Algorithms
    std::cout
        << "\n--- Test 3: Thrust Device Policy (GPU Multi-Algorithm Test) ---"
        << std::endl;
    hpx::thrust::thrust_device_policy device_policy{};
    thrust::device_vector<int> device_vec(size, 0);
    thrust::device_vector<int> device_vec2(size, 2);
    thrust::device_vector<int> device_result(size, 0);

    std::cout << " Testing hpx::fill with thrust_device_policy..." << std::endl;

    // Warm-up run to account for CUDA initialization overhead
    hpx::fill(device_policy, device_vec.begin(), device_vec.end(), 1);

    auto start_device = std::chrono::high_resolution_clock::now();
    hpx::fill(device_policy, device_vec.begin(), device_vec.end(), fill_value);
    auto stop_device = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_device =
        stop_device - start_device;
    std::cout << "Device fill execution time: " << duration_device.count()
              << " ms" << std::endl;

    std::cout << " Testing hpx::copy with thrust_device_policy..." << std::endl;
    hpx::copy(device_policy, device_vec.begin(), device_vec.end(),
        device_result.begin());

    std::cout << " Testing hpx::transform with thrust_device_policy..."
              << std::endl;
    hpx::transform(device_policy, device_vec.begin(), device_vec.end(),
        device_vec2.begin(), device_result.begin(),
        [] __device__(int a, int b) { return a * b; });

    // Verify GPU results by copying back to host
    int first_val = device_vec[0];
    int last_val = device_vec[size - 1];
    int first_result = device_result[0];
    int last_result = device_result[size - 1];
    bool success_device = (first_val == fill_value) &&
        (last_val == fill_value) && (first_result == fill_value * 2) &&
        (last_result == fill_value * 2);
    std::cout << "Device policy verification: "
              << (success_device ? "Success" : "Failed") << std::endl;

    // Test 4: Test NEW algorithms that weren't available before
    std::cout << "\n--- Test 4: Testing Additional Algorithms ---" << std::endl;

    // Initialize test data
    thrust::host_vector<int> numbers(10);
    std::iota(
        numbers.begin(), numbers.end(), 1);    // Fill with 1, 2, 3, ..., 10
    thrust::device_vector<int> d_numbers = numbers;

    std::cout << "Original data: ";
    for (int i = 0; i < 10; ++i)
        std::cout << numbers[i] << " ";
    std::cout << std::endl;

    std::cout << " Testing hpx::reverse with thrust_device_policy..."
              << std::endl;
    hpx::reverse(device_policy, d_numbers.begin(), d_numbers.end());

    // Copy back to see result
    thrust::host_vector<int> reversed_result = d_numbers;
    std::cout << "After reverse: ";
    for (int i = 0; i < 10; ++i)
        std::cout << reversed_result[i] << " ";
    std::cout << std::endl;

    std::cout << " Testing hpx::reduce with thrust_device_policy..."
              << std::endl;
    int sum = hpx::reduce(device_policy, d_numbers.begin(), d_numbers.end(), 0);
    std::cout << "Sum of reversed array: " << sum << " (expected: 55)"
              << std::endl;

    // Test 5: Global Instance Test
    std::cout << "\n--- Test 5: Global Instance Test ---" << std::endl;
    std::cout << "Testing global thrust_host and thrust_device instances..."
              << std::endl;

    // Test global thrust_host instance
    std::vector<int> global_host_vec(1000, 0);
    hpx::fill(hpx::thrust::thrust_host, global_host_vec.begin(),
        global_host_vec.end(), 99);
    bool global_host_success =
        (global_host_vec[0] == 99) && (global_host_vec[999] == 99);
    std::cout << "thrust_host global instance: "
              << (global_host_success ? "Success" : "Failed") << std::endl;

    // Test global thrust_device instance
    thrust::device_vector<int> global_device_vec(1000, 0);
    hpx::fill(hpx::thrust::thrust_device, global_device_vec.begin(),
        global_device_vec.end(), 88);
    int first_device = global_device_vec[0];
    int last_device = global_device_vec[999];
    bool global_device_success = (first_device == 88) && (last_device == 88);
    std::cout << "thrust_device global instance: "
              << (global_device_success ? "Success" : "Failed") << std::endl;

    // Performance Comparison
    std::cout << "\n=== Performance Comparison ===" << std::endl;
    std::cout << "CPU Sequential:    " << duration_cpu.count() << " ms"
              << std::endl;
    std::cout << "Thrust Host:       " << duration_host.count() << " ms"
              << std::endl;
    std::cout << "Thrust Device:     " << duration_device.count() << " ms"
              << std::endl;

    bool all_tests_passed = success_cpu && success_host && success_device &&
        global_host_success && global_device_success &&
        (sum == 55);    // Check reduce result

    // Test 6: Task policy with explicit target (async: fill)
    std::cout << "\n--- Test 6: Task policy (explicit target) async fill ---"
              << std::endl;
    hpx::cuda::experimental::target tgt =
        hpx::cuda::experimental::get_default_target();
    hpx::thrust::thrust_task_policy task_policy{};
    auto p_task = task_policy.on(tgt);
    thrust::device_vector<int> dev_task_vec(size, 0);
    auto f_fill =
        hpx::fill(p_task, dev_task_vec.begin(), dev_task_vec.end(), 7);
    static_assert(std::is_same<decltype(f_fill), hpx::future<void>>::value,
        "async fill should return future<void>");
    f_fill.get();
    bool task_explicit_ok =
        (dev_task_vec[0] == 7) && (dev_task_vec[size - 1] == 7);
    std::cout << "Task explicit target verification: "
              << (task_explicit_ok ? "Success" : "Failed") << std::endl;
    all_tests_passed = all_tests_passed && task_explicit_ok;

    // Test 7: Task policy default-target fallback (async: reverse)
    std::cout << "\n--- Test 7: Task policy (default target) async reverse ---"
              << std::endl;
    thrust::host_vector<int> h_small(10);
    std::iota(h_small.begin(), h_small.end(), 1);
    thrust::device_vector<int> d_small = h_small;
    hpx::thrust::thrust_task_policy
        default_task{};    // no .on(target) -> uses default target
    auto f_rev = hpx::reverse(default_task, d_small.begin(), d_small.end());
    f_rev.get();
    thrust::host_vector<int> h_after = d_small;
    bool task_default_ok = (h_after.front() == 10) && (h_after.back() == 1);
    std::cout << "Task default target verification: "
              << (task_default_ok ? "Success" : "Failed") << std::endl;
    all_tests_passed = all_tests_passed && task_default_ok;

    // Test 8: Same-stream ordering (reuse bound target): fill 1 then fill 2
    std::cout << "\n--- Test 8: Same-stream ordering (fill->fill) ---"
              << std::endl;
    thrust::device_vector<int> order_vec(size, 0);
    auto f1 = hpx::fill(p_task, order_vec.begin(), order_vec.end(), 1);
    auto f2 = hpx::fill(p_task, order_vec.begin(), order_vec.end(), 2);
    hpx::when_all(f1, f2).get();
    bool order_ok = (order_vec[0] == 2) && (order_vec[size - 1] == 2);
    std::cout << "Same-stream ordering verification: "
              << (order_ok ? "Success" : "Failed") << std::endl;
    all_tests_passed = all_tests_passed && order_ok;

    // Test 9: Async algorithms with return values (fill_n)
    std::cout << "\n--- Test 9: Async fill_n with return value ---"
              << std::endl;
    thrust::device_vector<int> fill_n_vec(size, 0);

    // Test async fill_n - should return future<iterator>
    auto f_fill_n = hpx::fill_n(default_task, fill_n_vec.begin(), size / 2, 99);
    static_assert(std::is_same_v<decltype(f_fill_n),
                      hpx::future<thrust::device_vector<int>::iterator>>,
        "async fill_n should return future<iterator>");

    // Get the result iterator when operation completes
    auto result_iter = f_fill_n.get();

    // Verify the iterator points to the correct position
    auto expected_iter = fill_n_vec.begin() + size / 2;
    bool iter_correct = (result_iter == expected_iter);

    // Verify the actual fill operation worked
    bool first_half_filled =
        (fill_n_vec[0] == 99) && (fill_n_vec[size / 2 - 1] == 99);
    bool second_half_untouched =
        (fill_n_vec[size / 2] == 0) && (fill_n_vec[size - 1] == 0);

    bool fill_n_success =
        iter_correct && first_half_filled && second_half_untouched;
    std::cout << "Async fill_n verification: "
              << (fill_n_success ? "Success" : "Failed") << std::endl;
    std::cout << "  - Iterator position: "
              << (iter_correct ? "Correct" : "Wrong") << std::endl;
    std::cout << "  - First half filled: " << (first_half_filled ? "Yes" : "No")
              << std::endl;
    std::cout << "  - Second half untouched: "
              << (second_half_untouched ? "Yes" : "No") << std::endl;

    all_tests_passed = all_tests_passed && fill_n_success;

    std::cout << "\n=== Test Results ===" << std::endl;
    std::cout
        << "Algorithms tested: fill, fill_n, copy, transform, reverse, reduce"
        << std::endl;
    std::cout << "Policies tested: thrust_host_policy, thrust_device_policy"
              << std::endl;
    std::cout << "Universal tag_invoke: "
              << (all_tests_passed ? "Working" : "Some issues detected")
              << std::endl;
    std::cout << "Overall result: "
              << (all_tests_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED")
              << std::endl;

    (void) all_tests_passed;
    return hpx::local::finalize();
}

int main(int argc, char** argv)
{
    hpx::local::init_params init_args;
    auto result = hpx::local::init(hpx_main, argc, argv, init_args);
    return result;
}