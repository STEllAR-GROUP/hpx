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
parallel computing applications from different APIs (i.e. |openmp|, |tbb|, |mpi|) to |hpx|. |hpx|, an
advanced C++ library, offers a versatile and high-performance platform for parallel and distributed
computing, providing a wide range of features and capabilities. This guide aims to assist developers
in understanding the key differences between different APIs and |hpx|, and it provides step-by-step
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

    #include <hpx/algorithm.hpp>

    hpx::experimental::for_loop(hpx::execution::par, 0, n, [&](int i) {
        // loop body
    });


In the above code, the |openmp| `#pragma omp parallel for` directive is replaced with
:cpp:func:`hpx::experimental::for_loop` from the |hpx| library. The loop body within the lambda
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

    #include <hpx/algorithm.hpp>

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

    #include <hpx/algorithm.hpp>

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

    #include <hpx/algorithm.hpp>
    #include <hpx/execution.hpp>

    hpx::execution::experimental::num_cores nc(2);

    hpx::experimental::for_loop(hpx::execution::par.with(nc), 0, n, [&](int i) {
        // loop body
    });


To declare the number of threads to be used for the parallel region, you can use
`hpx::execution::experimental::num_cores` and pass the number of cores (`nc`) to
:cpp:func:`hpx::experimental::for_loop` using `hpx::execution::par.with(nc)`.
This example uses 2 threads for the parallel loop.

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

    #include <hpx/algorithm.hpp>
    #include <hpx/execution.hpp>

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

    #include <hpx/algorithm.hpp>
    #include <hpx/execution.hpp>

    hpx::execution::experimental::static_chunk_size cs(1000);

    hpx::experimental::for_loop(hpx::execution::par.with(cs), 0, n, [&](int i) {
        // loop body
    });

To define the scheduling type, you can use the corresponding execution policy from
`hpx::execution::experimental`, define the chunk size (cs, here declared as 1000) and pass
it to the to :cpp:func:`hpx::experimental::for_loop` using `hpx::execution::par.with(cs)`.

Accordingly, other types of scheduling are available and can be used in a similar manner:

.. code-block:: c++

    #include <hpx/execution.hpp>
    hpx::execution::experimental::dynamic_chunk_size cs(1000);

.. code-block:: c++

    #include <hpx/execution.hpp>
    hpx::execution::experimental::guided_chunk_size cs(1000);

.. code-block:: c++

    #include <hpx/execution.hpp>
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

    #include <hpx/mutex.hpp>

    hpx::mutex mtx;

    {   // parallel code
        {   // single-threaded code
            std::scoped_lock l(mtx);
        }
        // more parallel code
    }

To make sure that only one thread accesses a specific code within a parallel section
you can use :cpp:class:`hpx::mutex` and `std::scoped_lock` to take ownership of the given
mutex `mtx`. For more information about mutexes please refer to :ref:`mutex`.

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

    #include <hpx/future.hpp>

    hpx::post([](){
        // task code
    }); // fire and forget

The tasks in |hpx| can be defined simply by using the :cpp:func:`async` function and passing as argument
the code you wish to run asynchronously. Another alternative is to use :cpp:func:`post` which is a
fire-and-forget method.

.. tip::

    If you think you will like to synchronize your tasks later on, we suggest you use
    :cpp:func:`hpx::async` which provides synchronization options, while :cpp:func:`hpx::post`
    explicitly states that there is no return value or way to synchronize with the function
    execution. Synchronization options are listed below.

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

The `get()` function can be used to ensure that the task created with :cpp:func:`hpx::async`
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


If you would like to synchronize multiple tasks, you can use the :cpp:func:`hpx::when_all` function
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

    int a = 10;
    int b = 20;
    int c = 0;

    // Create a future representing 'a'
    auto future_a = hpx::make_ready_future(a);

    // Create a future representing 'b'
    auto future_b = hpx::make_ready_future(b);

    // Create a task that depends on 'a' and 'b' and executes 'task_code'
    auto future_c = hpx::dataflow(
        []() {
            // task code
            return 100;
        },
        future_a, future_b);

    c = future_c.get();

If one of the arguments of :cpp:func:`hpx::dataflow` is a future, then it will wait for the
future to be ready to launch the thread. Hence, to define the dependencies of tasks
you have to create futures representing the variables that create dependencies and pass
them as arguments to :cpp:func:`hpx::dataflow`. `get()` is used to save the result of the future
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

    #include <hpx/future.hpp>

    auto future_outer = hpx::post([](){ // fire and forget
        // Outer task code

        hpx::post([](){ // fire and forget
            // Inner task code
        });
    });

If you have nested tasks, you can simply use nested :cpp:func:`hpx::async` or
:cpp:func:`hpx::post` calls. The implementation is similar if you want to take
care of synchronization:

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

    auto future_outer = hpx::async([]() {
        // Outer task code

        hpx::async([]() {
            // Inner task code
        }).get();    // Wait for the inner task to complete
    });

    future_outer.get();    // Wait for the outer task to complete

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
    #include <hpx/thread.hpp>

    auto future = hpx::async([](){
        // code before yielding
    });

    // yield execution to potentially allow other tasks to run
    hpx::this_thread::yield();

    // code after yielding

After creating a task using :cpp:func:`hpx::async`, :cpp:func:`hpx::this_thread::yield`
can be used to reschedule the execution of threads, allowing other threads to run.

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

    #include <hpx/task_group.hpp>

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

To create task groups, you can use :cpp:class:`hpx::experimental::task_group`. The function
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
directive in |openmp|. This synchronization is achieved using :cpp:func:`hpx::wait_all` function.

.. note::

    If the `nowait` clause is used in the `sections` directive, then you can just remove
    the :cpp:func:`hpx::wait_all` function while keeping the rest of the code as it is.

|tbb|
=====

|tbb| provides a high-level interface for parallelism and concurrent programming using
standard ISO C++ code. Below are some examples on how to convert |tbb| to |hpx| code:

parallel_for
------------

|tbb| code:

.. code-block:: c++

    auto values = std::vector<double>(10000);

    tbb::parallel_for( tbb::blocked_range<int>(0,values.size()),
                        [&](tbb::blocked_range<int> r)
    {
        for (int i=r.begin(); i<r.end(); ++i)
        {
            // loop body
        }
    });


|hpx| equivalent:

.. code-block:: c++

    #include <hpx/algorithm.hpp>

    auto values = std::vector<double>(10000);

    hpx::experimental::for_loop(hpx::execution::par, 0, values.size(), [&](int i) {
        // loop body
    });


In the above code, `tbb::parallel_for` is replaced with :cpp:func:`hpx::experimental::for_loop`
from the |hpx| library. The loop body within the lambda function will be executed in
parallel for each iteration.

parallel_for_each
-----------------

|tbb| code:

.. code-block:: c++

    auto values = std::vector<double>(10000);

    tbb::parallel_for_each(values.begin(), values.end(), [&](){
        // loop body
    });


|hpx| equivalent:

.. code-block:: c++

    #include <hpx/algorithm.hpp>

    auto values = std::vector<double>(10000);

    hpx::for_each(hpx::execution::par, values.begin(), values.end(), [&](){
        // loop body
    });

By utilizing :cpp:func:`hpx::for_each`` and specifying a parallel execution policy with
`hpx::execution::par`, it is possible to transform `tbb::parallel_for_each`` into its
equivalent counterpart in |hpx|.

parallel_invoke
---------------

|tbb| code:

.. code-block:: c++

    tbb::parallel_invoke(task1, task2, task3);

|hpx| equivalent:

.. code-block:: c++

    #include <hpx/future.hpp>

    hpx::wait_all(hpx::async(task1), hpx::async(task2), hpx::async(task3));

To convert `tbb::parallel_invoke` to |hpx|, we use :cpp:func:`hpx::async` to asynchronously
execute each task, which returns a future representing the result of each task.
We then pass these futures to :cpp:func:`hpx::when_all`, which waits for all the futures
to complete before returning.


parallel_pipeline
-----------------

|tbb| code:

.. code-block:: c++

    tbb::parallel_pipeline(4,
        tbb::make_filter<void, int>(tbb::filter::serial_in_order,
            [](tbb::flow_control& fc) -> int {
                // Generate numbers from 1 to 10
                static int i = 1;
                if (i <= 10) {
                    return i++;
                }
                else {
                    fc.stop();
                    return 0;
                }
            }) &
        tbb::make_filter<int, int>(tbb::filter::parallel,
            [](int num) -> int {
                // Multiply each number by 2
                return num * 2;
            }) &
        tbb::make_filter<int, void>(tbb::filter::serial_in_order,
            [](int num) {
                // Print the results
                std::cout << num << " ";
            })
    );

|hpx| equivalent:

.. code-block:: c++

    #include <iostream>
    #include <vector>
    #include <ranges>
    #include <hpx/algorithm.hpp>

    // generate the values
    auto range = std::views::iota(1) | std::views::take(10);

    // materialize the output vector
    std::vector<int> results(10);

    // in parallel execution of pipeline and transformation
    hpx::ranges::transform(
        hpx::execution::par, range, result.begin(), [](int i) { return 2 * i; });

    // print the modified vector
    for (int i : result)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;

The line `auto range = std::views::iota(1) | std::views::take(10);` generates a range of values using
the `std::views::iota` function. It starts from the value 1 and generates an infinite sequence of
incrementing values. The `std::views::take(10)` function is then applied to limit the sequence to the
first 10 values. The result is stored in the `range` variable.

.. hint::

    A view is a lightweight object that represents a particular view of a sequence or range. It acts as
    a read-only interface to the original data, providing a way to query and traverse the elements
    without making any copies or modifications.

    Views can be composed and chained together to form complex pipelines of operations. These operations
    are evaluated lazily, meaning that the actual computation is performed only when the result is
    needed or consumed.

Since views perform lazy evaluation, we use `std::vector<int> results(10);` to meterialize the vector
that will store the transformed values. The `hpx::ranges::transform` function is then used to perform
a parallel transformation on the range. The transformed values will be written to the `results` vector.

.. hint::

    Ranges enable loop fusion by combining multiple operations into a single parallel loop, eliminating
    waiting time and reducing overhead. Using ranges, you can express these operations as a pipeline of
    transformations on a sequence of elements. This pipeline is evaluated in a single pass, performing
    all the desired operations in parallel without the need to wait between them.

    In addition, |hpx| enhances the benefits of range fusion by offering parallel execution policies,
    which can be used to optimize the execution of the fused loop across multiple threads.


parallel_reduce
---------------

Reduction
^^^^^^^^^

|tbb| code:

.. code-block:: c++

    auto values = std::vector<double>{1,2,3,4,5,6,7,8,9};

    auto total = tbb::parallel_reduce(
                    tbb::blocked_range<int>(0,values.size()),
                    0.0,
                    [&](tbb::blocked_range<int> r, double running_total)
                    {
                        for (int i=r.begin(); i<r.end(); ++i)
                        {
                            running_total += values[i];
                        }


                        return running_total;
                    },
                    std::plus<double>());

|hpx| equivalent:

.. code-block:: c++

    #include <hpx/numeric.hpp>

    auto values = std::vector<double>{1,2,3,4,5,6,7,8,9};

    auto total = hpx::reduce(
        hpx::execution::par, values.begin(), values.end(), 0, std::plus{});

By utilizing :cpp:func:`hpx::reduce` and specifying a parallel execution policy with
`hpx::execution::par`, it is possible to transform `tbb::parallel_reduce`` into its
equivalent counterpart in |hpx|. As demonstrated in the previous example, the management
of intermediate results is seamlessly handled internally by |hpx|, eliminating the need
for explicit consideration.

Transformation & Reduction
^^^^^^^^^^^^^^^^^^^^^^^^^^

|tbb| code:

.. code-block:: c++

    auto values = std::vector<double>{1,2,3,4,5,6,7,8,9};

    auto transform_function(double current_value){
        // transformation code
    }

    auto total = tbb::parallel_reduce(
                    tbb::blocked_range<int>(0,values.size()),
                    0.0,
                    [&](tbb::blocked_range<int> r, double transformed_val)
                    {
                        for (int i=r.begin(); i<r.end(); ++i)
                        {
                            transformed_val += transform_function(values[i]);
                        }
                        return transformed_val;
                    },
                    std::plus<double>());


|hpx| equivalent:

.. code-block:: c++

    #include <hpx/numeric.hpp>

    auto values = std::vector<double>{1,2,3,4,5,6,7,8,9};

    auto transform_function(double current_value)
    {
        // transformation code
    }

    auto total = hpx::transform_reduce(hpx::execution::par, values.begin(),
        values.end(), 0, std::plus{},
        [&](double current_value) { return transform_function(current_value); });

In situations where certain values require transformation before the reduction process,
|hpx| provides a straightforward solution through :cpp:func:`hpx::transform_reduce`. The
`transform_function()` allows for the application of the desired transformation to each value.

parallel_scan
-------------

|tbb| code:

.. code-block:: c++

    tbb::parallel_scan(tbb::blocked_range<size_t>(0, input.size()),
        0,
        [&input, &output](const tbb::blocked_range<size_t>& range, int& partial_sum, bool is_final_scan) {
            for (size_t i = range.begin(); i != range.end(); ++i) {
                partial_sum += input[i];
                if (is_final_scan) {
                    output[i] = partial_sum;
                }
            }
            return partial_sum;
        },
        [](int left_sum, int right_sum) {
            return left_sum + right_sum;
        }
    );


|hpx| equivalent:

.. code-block:: c++

    #include <hpx/numeric.hpp>

    hpx::inclusive_scan(hpx::execution::par, input.begin(), input.end(),
        output.begin(),
        [](const int& left, const int& right) { return left + right; });

:cpp:func:`hpx::inclusive_scan` with `hpx::execution::par` as execution policy
 can be used to perform a prefix scan in parallel. The management of intermediate
results is seamlessly handled internally by |hpx|, eliminating the need
for explicit consideration. `input.begin()` and `input.end()` refer to the beginning
and end of the sequence of elements the algorithm will be applied to respectively.
`output.begin()` refers to the beginning of the destination, while the last argument
specifies the function which will be invoked for each of the values of the input sequence.

.. seealso::

Apart from :cpp:func:`hpx::inclusive_scan`, |hpx| provides its users with :cpp:func:`hpx::exclusive_scan`.
The key difference between inclusive scan and exclusive scan lies in the treatment of the current element
during the scan operation. In an inclusive scan, each element in the output sequence includes the
contribution of the corresponding element in the input sequence, while in an exclusive scan, the current
element in the input sequence does not contribute to the corresponding element in the output sequence.


parallel_sort
-------------

|tbb| code:

.. code-block:: c++

    std::vector<int> numbers = {9, 2, 7, 1, 5, 3};

    tbb::parallel_sort(numbers.begin(), numbers.end());


|hpx| equivalent:

.. code-block:: c++

    #include <hpx/algorithm.hpp>

    std::vector<int> numbers = {9, 2, 7, 1, 5, 3};

    hpx::sort(hpx::execution::par, numbers.begin(), numbers.end());

:cpp:func:`hpx::sort` provides an equivalent functionality to `tbb::parallel_sort`.
When given a parallel execution policy with `hpx::execution::par`, the algorithm employs
parallel execution, allowing for efficient sorting across available threads.

task_group
----------

|tbb| code:

.. code-block:: c++

    // Declare a task group
    tbb::task_group tg;

    // Run the tasks
    tg.run(task1);
    tg.run(task2);

    // Wait for the task group
    tg.wait();


|hpx| equivalent:

.. code-block:: c++

    #include <hpx/task_group.hpp>

    // Declare a task group
    hpx::experimental::task_group tg;

    // Run the tasks
    tg.run(task1);
    tg.run(task2);

    // Wait for the task group
    tg.wait();

|hpx| drew inspiration from |tbb| to introduce the :cpp:func:`hpx::experimental::task_group`
feature. Therefore, utilizing :cpp:func:`hpx::experimental::task_group` provides an
equivalent functionality to `tbb::task_group`.

|mpi|
=====

|mpi| is a standardized communication protocol and library that allows multiple processes or
nodes in a parallel computing system to exchange data and coordinate their execution.

MPI_send & MPI_recv
-------------------

Let's assume we have the following simple message passing code where each process sends a
message to the next process in a circular manner. The exchanged message is modified and printed
to the console.

|mpi| code:

.. code-block:: c++

    #include <cstddef>
    #include <cstdint>
    #include <iostream>
    #include <mpi.h>
    #include <vector>

    constexpr int times = 2;

    int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int num_localities;
    MPI_Comm_size(MPI_COMM_WORLD, &num_localities);

    int this_locality;
    MPI_Comm_rank(MPI_COMM_WORLD, &this_locality);

    int next_locality = (this_locality + 1) % num_localities;
    std::vector<int> msg_vec = {0, 1};

    int cnt = 0;
    int msg = msg_vec[this_locality];

    int recv_msg;
    MPI_Request request_send, request_recv;
    MPI_Status status;

    while (cnt < times) {
        cnt += 1;

        MPI_Isend(&msg, 1, MPI_INT, next_locality, cnt, MPI_COMM_WORLD,
                &request_send);
        MPI_Irecv(&recv_msg, 1, MPI_INT, next_locality, cnt, MPI_COMM_WORLD,
                &request_recv);

        MPI_Wait(&request_send, &status);
        MPI_Wait(&request_recv, &status);

        std::cout << "Time: " << cnt << ", Locality " << this_locality
                << " received msg: " << recv_msg << "\n";

        recv_msg += 10;
        msg = recv_msg;
    }

    MPI_Finalize();
    return 0;
    }

|hpx| equivalent:

.. literalinclude:: ../../libs/full/collectives/examples/channel_communicator.cpp
   :start-after: //[doc
   :end-before: //doc]

To perform message passing between different processes in |hpx| we can use a channel communicator.
To understand this example, let's focus on the `hpx_main()` function:

- `hpx::get_num_localities(hpx::launch::sync)` retrieves the number of localities, while
  `hpx::get_locality_id()` returns the ID of the current locality.
- `create_channel_communicator` function is used to create a channel to serve the communication.
  This function takes several arguments, including the launch policy (`hpx::launch::sync`), the
  name of the communicator (`channel_communicator_name`), the number of localities, and the ID
  of the current locality.
- The communication follows a ring pattern, where each process (or locality) sends a message to
  its neighbor in a circular manner. This means that the messages circulate around the localities,
  ensuring that the communication wraps around when reaching the end of the locality sequence.
  To achieve this, the `next_locality` variable is calculated as the ID of the next locality in
  the ring.
- The initial values for the communication are set (`msg_vec`, `cnt`, `msg`).
- The `set()` function is called to send the message to the next locality in the ring. The message
  is sent asynchronously and is associated with a tag (`cnt`).
- The `get()` function is called to receive a message from the next locality. It is also associated
  with the same tag as the `set()` operation.
- The `setf.get()` call blocks until the message sending operation is complete.
- A continuation is set up using the function `then()` to handle the received message.
  Inside the continuation:

  - The received message value (`rec_msg`) is retrieved using `f.get()`.

  - The received message is printed to the console and then modified by adding 10.

  - The `set()` and `get()` operations are repeated to send and receive the modified message to
    the next locality.

  - The `setf.get()` call blocks until the new message sending operation is complete.
- The `done_msg.get()` call blocks until the continuation is complete for the current loop iteration.

Having said that, we conclude to the following table:

.. table:: |hpx| equivalent functions of |mpi|

   =========================  ==============================================================
   |openmpi| function         |hpx| equivalent
   =========================  ==============================================================
   MPI_Comm_create            `hpx::collectives::create_channel_communicator()`
   MPI_Comm_size              `hpx::get_num_localities`
   MPI_Comm_rank              `hpx::get_locality_id()`
   MPI_Isend                  `hpx::collectives::set()`
   MPI_Irecv                  `hpx::collectives::get()`
   MPI_Wait                   `hpx::collectives::get()` used with a future i.e. `setf.get()`
   =========================  ==============================================================
