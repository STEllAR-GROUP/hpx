..
    Copyright (c) 2025 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _modules_thrust:

======
thrust
======

The thrust module provides integration between |hpx| parallel algorithms 
and NVIDIA Thrust, enabling high-performance GPU computing with the familiar |hpx| 
algorithm interface. Use standard |hpx| algorithms with Thrust execution policies 
to automatically leverage GPU acceleration.

Key Features
============

* **GPU Acceleration**: Automatically use GPU-optimized algorithms with standard |hpx| syntax
* **Three Execution Policies**: Host, device, and asynchronous task-based execution
* **Future Integration**: Non-blocking GPU operations with |hpx| futures
* **Comprehensive Coverage**: Supports 58+ parallel algorithms from Thrust 12.4.3
* **Type Safety**: Compile-time errors for unsupported algorithm combinations

Execution Policies
==================

The module provides three main execution policies:

``hpx::thrust::thrust_host_policy``
   Executes algorithms on CPU using optimized parallel implementations. 
   Use for CPU-bound workloads or when GPU is unavailable.

``hpx::thrust::thrust_device_policy``
   Executes algorithms synchronously on GPU. Use for compute-intensive 
   operations on large datasets where you can wait for completion.

``hpx::thrust::thrust_task_policy``
   Executes algorithms asynchronously on GPU and returns |hpx| futures. 
   Use for overlapping GPU work with CPU computation or chaining operations.

Policy Mappings
===============

The |hpx| thrust policies map directly to Thrust execution policies:

.. code-block:: c++

    // Policy mappings
    hpx::thrust::thrust_host_policy    -> thrust::host
    hpx::thrust::thrust_device_policy  -> thrust::device  
    hpx::thrust::thrust_task_policy    -> thrust::par_nosync

HPX Futures Integration
=======================

The ``thrust_task_policy`` returns standard |hpx| futures that integrate seamlessly 
with all |hpx| future operations:

.. code-block:: c++

    hpx::thrust::thrust_task_policy task{};
    thrust::device_vector<int> vec(1000, 0);
    
    // Returns hpx::future<void>
    auto fill_future = hpx::fill(task, vec.begin(), vec.end(), 42);
    
    // Use with standard HPX future operations
    fill_future.wait();                    // Block until complete
    fill_future.get();                     // Get result (void)
    
    // Chain with .then() continuations  
    auto next = fill_future.then([&](auto&&) {
        return hpx::reduce(task, vec.begin(), vec.end(), 0);
    });
    
    // Combine with when_all/when_any
    auto all_done = hpx::when_all(fill_future, other_future);
    
    // Works with hpx::async and other future sources
    auto mixed = hpx::when_all(
        hpx::fill(task, vec.begin(), vec.end(), 1),
        hpx::async([]() { return compute_cpu_work(); })
    );

Basic Usage
===========

Synchronous Device Execution
-----------------------------

The most common usage pattern is synchronous execution on GPU device:

.. code-block:: c++

    #include <hpx/thrust/policy.hpp>
    #include <hpx/thrust/algorithms.hpp>
    #include <thrust/device_vector.h>
    #include <thrust/host_vector.h>

    // Create execution policy and data
    hpx::thrust::thrust_device_policy device{};
    thrust::device_vector<int> d_vec(1000, 0);
    
    // Fill with value 42
    hpx::fill(device, d_vec.begin(), d_vec.end(), 42);
    
    // Transform elements
    hpx::transform(device, d_vec.begin(), d_vec.end(), d_vec.begin(), 
                   [] __device__ (int x) { return x * 2; });
    
    // Reduce to sum
    int sum = hpx::reduce(device, d_vec.begin(), d_vec.end(), 0);

Host Execution
--------------

For CPU-based execution using optimized parallel algorithms:

.. code-block:: c++

    #include <hpx/thrust/policy.hpp>
    #include <vector>

    hpx::thrust::thrust_host_policy host{};
    std::vector<int> vec(1000);
    
    // Generate sequence
    hpx::generate(host, vec.begin(), vec.end(), 
                  []() { return std::rand() % 100; });
    
    // Sort in parallel on CPU
    hpx::sort(host, vec.begin(), vec.end());

Asynchronous Execution with Futures
------------------------------------

For non-blocking GPU operations and continuation-based programming:

Note: This will only work if there is an immediate return from the thrust algorithm call.

.. code-block:: c++

    #include <hpx/thrust/policy.hpp>
    #include <hpx/async_cuda/cuda_polling_helper.hpp>
    #include <hpx/modules/async_cuda.hpp>

    int hpx_main() {
        // Required for async GPU operations
        hpx::cuda::experimental::enable_user_polling polling_guard("default");
        
        hpx::thrust::thrust_task_policy task{};
        thrust::device_vector<int> d_vec(10000, 1);
        
        // Chain asynchronous operations
        auto fill_future = hpx::fill_n(task, d_vec.begin(), 5000, 99);
        
        auto transform_future = fill_future.then([&](auto&&) {
            return hpx::transform(task, d_vec.begin(), d_vec.end(), d_vec.begin(),
                                  [] __device__ (int x) { return x + 10; });
        });
        
        auto reduce_future = transform_future.then([&](auto&&) {
            return hpx::reduce(task, d_vec.begin(), d_vec.end(), 0);
        });
        
        // Wait for final result
        int result = reduce_future.get();
        return hpx::local::finalize();
    }


Error Handling and Debugging
=============================

Compilation Errors
-------------------

The module provides clear compile-time errors for unsupported operations:

.. code-block:: c++

    // This will produce a compile-time error if 'custom_algorithm' 
    // is not supported
    hpx::custom_algorithm(device_policy, ...);  // Error: unmapped algorithm

Unsupported algorithms will fail to compile with clear error messages.

Runtime Considerations
----------------------

For asynchronous operations, ensure proper CUDA polling is enabled:

.. code-block:: c++

    // Required at the start of hpx_main for thrust_task_policy
    hpx::cuda::experimental::enable_user_polling polling_guard("default");

Without this, ``thrust_task_policy`` operations will fail with runtime assertions
about missing CUDA event polling.

Performance Guidelines
======================

Algorithm Selection
-------------------

* Use ``thrust_device_policy`` for compute-intensive operations on large datasets
* Use ``thrust_host_policy`` for smaller datasets or when GPU is unavailable  
* Use ``thrust_task_policy`` for overlapping computation with other work


Supported Algorithms
====================

The module supports 58+ algorithms from Thrust 12.4.3. Major categories include:

**Data Manipulation**
  ``fill``, ``fill_n``, ``copy``, ``copy_if``, ``copy_n``, ``generate``, ``generate_n``

**Transformations**
  ``transform``, ``replace``, ``replace_if``, ``replace_copy``, ``replace_copy_if``

**Reductions**
  ``reduce``, ``transform_reduce``, ``count``, ``count_if``

**Scanning**
  ``inclusive_scan``, ``exclusive_scan``, ``transform_inclusive_scan``, ``transform_exclusive_scan``

**Sorting and Searching**
  ``sort``, ``stable_sort``, ``find``, ``find_if``, ``find_if_not``

**Set Operations**
  ``set_union``, ``set_intersection``, ``set_difference``, ``set_symmetric_difference``

**Partitioning**
  ``partition``, ``stable_partition``, ``partition_copy``

**Removing and Filtering**
  ``remove``, ``remove_if``, ``remove_copy``, ``remove_copy_if``, ``unique``, ``unique_copy``

**Comparison and Logic**
  ``equal``, ``mismatch``, ``all_of``, ``any_of``, ``none_of``

**Memory Operations**
  ``uninitialized_copy``, ``uninitialized_copy_n``, ``uninitialized_fill``, ``uninitialized_fill_n``

**Utility**
  ``reverse``, ``reverse_copy``, ``swap_ranges``, ``for_each``, ``for_each_n``

For a complete list, see the generated ``thrust_algorithms_coverage.csv`` file.



Dependencies and Build Requirements
===================================

**Build Dependencies**
  * CUDA Toolkit 12.4.0+
  * |hpx| with ``HPX_WITH_CUDA=ON``
  * Thrust library (included with CUDA or available standalone)

**Runtime Dependencies**
  * ``hpx_async_cuda`` module for CUDA integration

**CMake Configuration**

.. code-block:: cmake

    # Enable in HPX build
    cmake -DHPX_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 ...

The module is automatically built when CUDA support is enabled in |hpx|.

Migration from Direct Thrust Usage
===================================

Converting existing Thrust code to use |hpx| thrust policies:

.. code-block:: c++

    // Before: Direct Thrust usage
    thrust::device_vector<int> vec(1000);
    thrust::fill(thrust::device, vec.begin(), vec.end(), 42);
    int sum = thrust::reduce(thrust::device, vec.begin(), vec.end());
    
    // After: HPX thrust integration  
    thrust::device_vector<int> vec(1000);
    hpx::thrust::thrust_device_policy device{};
    hpx::fill(device, vec.begin(), vec.end(), 42);
    int sum = hpx::reduce(device, vec.begin(), vec.end());

Benefits:
* Unified algorithm interface across CPU and GPU
* Integration with |hpx| futures and continuations  
* Better composability with other |hpx| components
* Consistent programming model

See Also
========

* :ref:`modules_async_cuda` - Lower-level CUDA integration
* :ref:`modules_execution` - |hpx| execution policies and algorithms
* |thrust_docs|_ - Official Thrust documentation
* |cuda_docs|_ - CUDA programming guide

.. |thrust_docs| replace:: Thrust Documentation
.. _thrust_docs: https://nvidia.github.io/cccl/thrust/

.. |cuda_docs| replace:: CUDA Documentation  
.. _cuda_docs: https://docs.nvidia.com/cuda/

For more information, see the header files in ``hpx/thrust/`` and the examples in the ``examples/`` directory.