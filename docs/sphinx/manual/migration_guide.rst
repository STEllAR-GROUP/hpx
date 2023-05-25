..
    Copyright (c) 2021 Dimitra Karatza

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _migration_guide:

===============
Migration guide
===============

The Migration Guide serves as a valuable resource for developers seeking to transition their
parallel computing applications from different APIs (i.e. |openmp|) to |hpx|. |hpx|, an advanced C++
library, offers a versatile and high-performance platform for parallel and distributed computing,
providing a wide range of features and capabilities. This guide aims to assist developers in
understanding the key differences between different APIs and |hpx|, and it provides step-by-step
instructions for converting code to |hpx| code effectively.

Some general steps that can be used to migrate code to |hpx| code are the following:

1. Install |hpx| using the :ref:`quickstart` guide.

2. Include the |hpx| header files:

   Add the necessary header files for HPX at the beginning of your code, such as:

    .. code-block:: c++

        #include <hpx/init.hpp>

3. Replace your code with |hpx| code using the guide that follows.

4. Use HPX-specific features and APIs:

   |hpx| provides additional features and APIs that can be used to take advantage of the library's
   capabilities. For example, you can use the HPX asynchronous execution to express fine-grained
   tasks and dependencies, or utilize HPX's distributed computing features for distributed memory systems.

5. Compile and run the |hpx| code:

   Compile the converted code with the |hpx| library and run it using the appropriate HPX runtime environment.

|openmp|
========

The |openmp| API supports multi-platform shared-memory parallel programming in C/C++. Typically it is used for
loop-level parallelism, but it also supports function-level parallelism. Below are some examples on how to
convert |openmp| to |hpx| code:

|openmp| parallel for loop
--------------------------

Parallel for loop
^^^^^^^^^^^^^^^^^

|openmp| code:

.. code-block:: c++

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        // loop body
    }

|hpx| equivalent:

.. code-block:: c++

    #include <hpx/parallel/algorithms/for_loop.hpp>

    hpx::experimental::for_loop(hpx::execution::par, 0, n, [&](int i) {
        // loop body
    });


In the above code, the |openmp| `#pragma omp parallel for` directive is replaced with
`hpx::experimental::for_loop` from the |hpx| library. The loop body within the lambda
function will be executed in parallel for each iteration.

Private variables
^^^^^^^^^^^^^^^^^

|openmp| code:

.. code-block:: c++

    int x = 0;

    #pragma omp parallel for private(x)
    for (int i = 0; i < n; ++i) {
        // loop body
    }

|hpx| equivalent:

.. code-block:: c++

    #include <hpx/parallel/algorithms/for_loop.hpp>

    hpx::experimental::for_loop(hpx::execution::par, 0, n, [&](int i) {
            int x = 0; // Declare 'x' as a local variable inside the loop body
            // loop body
    });

The variable `x` is declared as a local variable inside the loop body, ensuring that
it is private to each thread.


Shared variables
^^^^^^^^^^^^^^^^

|openmp| code:

.. code-block:: c++

    int x = 0;

    #pragma omp parallel for shared(x)
    for (int i = 0; i < n; ++i) {
        // loop body
    }

|hpx| equivalent:

.. code-block:: c++

    #include <hpx/parallel/algorithms/for_loop.hpp>

    std::atomic<int> x = 0; // Declare 'x' as a shared variable outside the loop

    hpx::experimental::for_loop(hpx::execution::par, 0, n, [&](int i) {
        // loop body
    });

To ensure variable `x` is shared among all threads, you simply have to declare it as an
atomic variable outside the `for_loop`.

Number of threads
^^^^^^^^^^^^^^^^^

|openmp| code:

.. code-block:: c++

    #pragma omp parallel for num_threads(2)
    for (int i = 0; i < n; ++i) {
        // loop body
    }

|hpx| equivalent:

.. code-block:: c++

    #include <hpx/parallel/algorithms/for_loop.hpp>
    #include <hpx/execution/executors/num_cores.hpp>

    hpx::execution::experimental::num_cores nc(2);

    hpx::experimental::for_loop(hpx::execution::par.with(nc), 0, n, [&](int i) {
        // loop body
    });


To declare the number of threads to be used for the parallel region, you can use
`hpx::execution::experimental::num_cores` and pass the number of cores (`nc`) to the `for_loop`
using `hpx::execution::par.with(nc)`. This example uses 2 threads for the parallel loop.

Reduction
^^^^^^^^^

|openmp| code:

.. code-block:: c++

    int s = 0;

    #pragma omp parallel for reduction(+: s)
    for (int i = 0; i < n; ++i) {
        s += i;
        // loop body
    }

|hpx| equivalent:

.. code-block:: c++

    #include <hpx/parallel/algorithms/for_loop.hpp>
    #include <hpx/execution/executors/num_cores.hpp>

    int s = 0;

    hpx::experimental::for_loop(hpx::execution::par, 0, n, reduction(s, 0, plus<>()), [&](int i, int accum) {
        accum += i;
        // loop body
    });


The reduction clause specifies that the variable `s`` should be reduced across iterations using the `plus<>`` operation.
It initializes `s` to `0` at the beginning of the loop and accumulates the values of `s` from each iteration using the
`+` operator. The lambda function representing the loop body takes two parameters: `i`, which represents the loop index,
and `accum`, which is the reduction variable `s`. The lambda function is executed for each iteration of the loop.
The reduction ensures that the `accum` value is correctly accumulated across different iterations and threads.

Schedule
^^^^^^^^

|openmp| code:

.. code-block:: c++

    int s = 0;

    // static scheduling with chunk size 1000
    #pragma omp parallel for schedule(static, 1000)
    for (int i = 0; i < n; ++i) {
        // loop body
    }

|hpx| equivalent:

.. code-block:: c++

    #include <hpx/parallel/algorithms/for_loop.hpp>
    #include <hpx/execution/executors/static_chunk_size.hpp>

    hpx::execution::experimental::static_chunk_size cs(1000);

    hpx::experimental::for_loop(hpx::execution::par.with(cs), 0, n, [&](int i) {
        // loop body
    });

To define the scheduling type, you can use the corresponding execution policy from
`hpx::execution::experimental`, define the chunk size (cs, here declared as 1000) and pass
it to the `for_loop` using `hpx::execution::par.with(cs)`.

Accordingly, other types of scheduling are available and can be used in a similar manner:

.. code-block:: c++

    #include <hpx/execution/executors/dynamic_chunk_size.hpp>
    hpx::execution::experimental::dynamic_chunk_size cs(1000);

.. code-block:: c++

    #include <hpx/execution/executors/guided_chunk_size.hpp>
    hpx::execution::experimental::guided_chunk_size cs(1000);

.. code-block:: c++

    #include <hpx/execution/executors/auto_chunk_size.hpp>
    hpx::execution::experimental::auto_chunk_size cs(1000);



