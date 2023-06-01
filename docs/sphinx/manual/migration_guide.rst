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

    hpx::experimental::for_loop(hpx::execution::par, 0, n, reduction(s, 0, plus<>()), [&](int i, int& accum) {
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


|openmp| single thread
----------------------

|openmp| code:

.. code-block:: c++

    {   // parallel code
        #pragma omp single
        {
            // single-threaded code
        }
        // more parallel code
    }

|hpx| equivalent:

.. code-block:: c++

    hpx::mutex mtx;

    {   // parallel code
        {   // single-threaded code
            std::scoped_lock l(mtx);
        }
        // more parallel code
    }

To make sure that only one thread accesses a specific code within a parallel section
you can use `hpx::mutex` and `std::scoped_lock` to take ownership of the given mutex `mtx`.
For more information about mutexes please refer to :ref:`mutex`.

|openmp| tasks
--------------

Simple tasks
^^^^^^^^^^^^

|openmp| code:

.. code-block:: c++

    // executed asynchronously by any available thread
    #pragma omp task
    {
        // task code
    }

|hpx| equivalent:

.. code-block:: c++

    #include <hpx/future.hpp>

    auto future = hpx::async([](){
        // task code
    });

or

.. code-block:: c++

    #include <hpx/async_base/post.hpp>

    hpx::post([](){
        // task code
    }); // fire and forget

The tasks in |hpx| can be defined simply by using the `async` function and passing as argument
the code you wish to run asynchronously. Another alternative is to use `post` which is a
fire-and-forget method.

.. note::

    If you think you will like to synchronize your tasks later on, we suggest you use
    `hpx::async` which provides synchronization options, while `hpx::post` explicitly states
    that there is no return value or way to synchronize with the function execution.
    Synchronization options are listed below.

Task wait
^^^^^^^^^

|openmp| code:

.. code-block:: c++

    #pragma omp task
    {
        // task code
    }

    #pragma omp taskwait
    // code after completion of task

|hpx| equivalent:

.. code-block:: c++

    #include <hpx/future.hpp>

    hpx::async([](){
        // task code
    }).get(); // wait for the task to complete

    // code after completion of task

The `get()` function can be used to ensure that the task created with `hpx::async`
is completed before the code continues executing beyond that point.

Multiple tasks synchronization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|openmp| code:

.. code-block:: c++

    #pragma omp task
    {
        // task 1 code
    }

    #pragma omp task
    {
        // task 2 code
    }

    #pragma omp taskwait
    // code after completion of both tasks 1 and 2

|hpx| equivalent:

.. code-block:: c++

    #include <hpx/future.hpp>

    auto future1 = hpx::async([](){
        // task 1 code
    });

    auto future2 = hpx::async([](){
        // task 2 code
    });

    auto future = hpx::when_all(future1, future2).then([](auto&&){
        // code after completion of both tasks 1 and 2
    });


If you would like to synchronize multiple tasks, you can use the `hpx::when_all` function
to define which futures have to be ready and the `then()` function to declare what should
be executed once these futures are ready.


Dependencies
^^^^^^^^^^^^

|openmp| code:

.. code-block:: c++

    int a = 10;
    int b = 20;
    int c = 0;

    #pragma omp task depend(in: a, b) depend(out: c)
    {
        // task code
        c = 100;
    }

|hpx| equivalent:

.. code-block:: c++

    #include <hpx/future.hpp>
    #include <hpx/async_base/dataflow.hpp>

    int a = 10;
    int b = 20;
    int c = 0;

    // Create a future representing 'a'
    auto future_a = hpx::make_ready_future(a);

    // Create a future representing 'b'
    auto future_b = hpx::make_ready_future(b);

    // Create a task that depends on 'a' and 'b' and executes 'task_code'
    auto future_c = hpx::dataflow([](){
                                        // task code
                                        return 100;
                                      },
                                      future_a,
                                      future_b);

    c = future_c.get();

If one of the arguments of `hpx::dataflow` is a future, then it will wait for the
future to be ready to launch the thread. Hence, to define the dependencies of tasks
you have to create futures representing the variables that create dependencies and pass
them as arguments to `hpx::dataflow`. `get()` is used to save the result of the future
to the desired variable.


Nested tasks
^^^^^^^^^^^^

|openmp| code:

.. code-block:: c++

    #pragma omp task
    {
        // Outer task code
        #pragma omp task
        {
            // Inner task code
        }
    }

|hpx| equivalent:

.. code-block:: c++

    #include <hpx/future.hpp>

    auto future_outer = hpx::async([](){
        // Outer task code

        hpx::async([](){
            // Inner task code
        });
    });

or

.. code-block:: c++

    #include <hpx/async_base/post.hpp>

    auto future_outer = hpx::post([](){ // fire and forget
        // Outer task code

        hpx::post([](){ // fire and forget
            // Inner task code
        });
    });

If you have nested tasks, you can simply use nested `hpx::async` or `hpx::post` calls.
The implementation is similar if you want to take care of synchronization:

|openmp| code:

.. code-block:: c++

    #pragma omp taskwait
    {
        // Outer task code
        #pragma omp taskwait
        {
            // Inner task code
        }
    }

|hpx| equivalent:

.. code-block:: c++

    #include <hpx/future.hpp>

    auto future_outer = hpx::async([](){
        // Outer task code

        hpx::async([](){
            // Inner task code
        }).get(); // Wait for the inner task to complete
    });

    future_outer.get(); // Wait for the outer task to complete


Task yield
^^^^^^^^^^

|openmp| code:

.. code-block:: c++

    #pragma omp task
    {
        // code before yielding
        #pragma omp taskyield
        // code after yielding
    }

|hpx| equivalent:

.. code-block:: c++

    #include <hpx/future.hpp>
    #include <hpx/threading/thread.hpp>

    auto future = hpx::async([](){
        // code before yielding
    });

    // yield execution to potentially allow other tasks to run
    hpx::this_thread::yield();

    // code after yielding

After creating a task using `hpx::async`, `hpx::this_thread::yield()` can be used to
reschedule the execution of threads, allowing other threads to run.

Task group
^^^^^^^^^^

|openmp| code:

.. code-block:: c++

    #pragma omp taskgroup
    {
        #pragma omp task
        {
            // task 1 code
        }

        #pragma omp task
        {
            // task 2 code
        }
    }


|hpx| equivalent:

.. code-block:: c++

    #include <hpx/experimental/task_group.hpp>

    // Declare a task group
    hpx::experimental::task_group tg;

    // Run the tasks
    tg.run([](){
        // task 1 code
    });
    tg.run(
        // task 2 code
    });

    // Wait for the task group
    tg.wait();

To create task groups, you can use `hpx::experimental::task_group`. The function
`run()` can be used to run each task within the task group, while `wait()` can be used to
achieve synchronization. If you do not care about waiting for the task group to complete
its execution, you can simply remove the `wait()` function.

|openmp| sections
-----------------

|openmp| code:

.. code-block:: c++

    #pragma omp sections
    {
        #pragma omp section
        // section 1 code
        #pragma omp section
        // section 2 code
    } // implicit synchronization


|hpx| equivalent:

.. code-block:: c++

    #include <hpx/future.hpp>

    auto future_section1 = hpx::async([](){
        // section 1 code
    });
    auto future_section2 = hpx::async([](){
        // section 2 code
    );

    // synchronization: wait for both sections to complete
    hpx::wait_all(future_section1, future_section2);

Unlike tasks, there is an implicit synchronization barrier at the end of each `sections``
directive in |openmp|. This synchronization is achieved using `hpx::wait_all` function.

.. note::

    If the `nowait` clause is used in the `sections` directive, then you can just remove
    the `hpx::wait_all` function while keeping the rest of the code as it is.
