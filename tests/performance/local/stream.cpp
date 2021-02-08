//  Copyright (c) 2015 Thomas Heller
//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//
// This code is based on the STREAM benchmark:
// https://www.cs.virginia.edu/stream/ref.html
//
// We adopted the code and HPXifyed it.
//

#if defined(HPX_MSVC_NVCC)
// NVCC causes an ICE in MSVC if this is not defined
#define BOOST_NO_CXX11_ALLOCATOR
#endif

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/compute.hpp>
#include <hpx/include/parallel_copy.hpp>
#include <hpx/include/parallel_executor_parameters.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/parallel_fill.hpp>
#include <hpx/include/parallel_transform.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/type_support/unused.hpp>
#include <hpx/version.hpp>
#if defined(HPX_HAVE_MODULE_ASYNC_CUDA)
#include <hpx/async_cuda/cuda_future.hpp>
#include <hpx/async_cuda/target.hpp>
#endif

#include <cstddef>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif

bool csv = false;
bool header = false;

///////////////////////////////////////////////////////////////////////////////
hpx::threads::topology& retrieve_topology()
{
    static hpx::threads::topology& topo = hpx::threads::create_topology();
    return topo;
}

///////////////////////////////////////////////////////////////////////////////
double mysecond()
{
    return hpx::chrono::high_resolution_clock::now() * 1e-9;
}

int checktick()
{
    static const std::size_t M = 20;
    int minDelta, Delta;
    double t1, t2, timesfound[M];

    // Collect a sequence of M unique time values from the system.
    for (std::size_t i = 0; i < M; i++)
    {
        t1 = mysecond();
        while (((t2 = mysecond()) - t1) < 1.0E-6)
            ;
        timesfound[i] = t1 = t2;
    }

    // Determine the minimum difference between these M values.
    // This result will be our estimate (in microseconds) for the
    // clock granularity.
    minDelta = 1000000;
    for (std::size_t i = 1; i < M; i++)
    {
        Delta = (int) (1.0E6 * (timesfound[i] - timesfound[i - 1]));
        minDelta = (std::min)(minDelta, (std::max)(Delta, 0));
    }

    return (minDelta);
}

template <typename Vector>
void check_results(std::size_t iterations, Vector const& a_res,
    Vector const& b_res, Vector const& c_res)
{
    std::vector<STREAM_TYPE> a(a_res.size());
    std::vector<STREAM_TYPE> b(b_res.size());
    std::vector<STREAM_TYPE> c(c_res.size());

    hpx::copy(hpx::execution::par, a_res.begin(), a_res.end(), a.begin());
    hpx::copy(hpx::execution::par, b_res.begin(), b_res.end(), b.begin());
    hpx::copy(hpx::execution::par, c_res.begin(), c_res.end(), c.begin());

    STREAM_TYPE aj, bj, cj, scalar;
    STREAM_TYPE aSumErr, bSumErr, cSumErr;
    STREAM_TYPE aAvgErr, bAvgErr, cAvgErr;
    double epsilon;
    int ierr, err;

    /* reproduce initialization */
    aj = 1.0;
    bj = 2.0;
    cj = 0.0;
    /* a[] is modified during timing check */
    aj = 2.0E0 * aj;
    /* now execute timing loop */
    scalar = 3.0;
    for (std::size_t k = 0; k < iterations; k++)
    {
        cj = aj;
        bj = scalar * cj;
        cj = aj + bj;
        aj = bj + scalar * cj;
    }

    /* accumulate deltas between observed and expected results */
    aSumErr = 0.0;
    bSumErr = 0.0;
    cSumErr = 0.0;
    for (std::size_t j = 0; j < a.size(); j++)
    {
        aSumErr += std::abs(a[j] - aj);
        bSumErr += std::abs(b[j] - bj);
        cSumErr += std::abs(c[j] - cj);
        // if (j == 417) printf("Index 417: c[j]: %f, cj: %f\n",c[j],cj);   // MCCALPIN
    }
    aAvgErr = aSumErr / (STREAM_TYPE) a.size();
    bAvgErr = bSumErr / (STREAM_TYPE) a.size();
    cAvgErr = cSumErr / (STREAM_TYPE) a.size();

    if (sizeof(STREAM_TYPE) == 4)
    {
        epsilon = 1.e-6;
    }
    else if (sizeof(STREAM_TYPE) == 8)
    {
        epsilon = 1.e-13;
    }
    else
    {
        printf("WEIRD: sizeof(STREAM_TYPE) = %zu\n", sizeof(STREAM_TYPE));
        epsilon = 1.e-6;
    }

    err = 0;
    if (std::abs(aAvgErr / aj) > epsilon)
    {
        err++;
        printf("Failed Validation on array a[], AvgRelAbsErr > epsilon (%e)\n",
            epsilon);
        printf("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n", aj,
            aAvgErr, std::abs(aAvgErr) / aj);
        ierr = 0;
        for (std::size_t j = 0; j < a.size(); j++)
        {
            if (std::abs(a[j] / aj - 1.0) > epsilon)
            {
                ierr++;
#ifdef VERBOSE
                if (ierr < 10)
                {
                    printf("         array a: index: %ld, expected: %e, "
                           "observed: %e, relative error: %e\n",
                        (unsigned long) j, aj, a[j],
                        (double) std::abs((aj - a[j]) / aAvgErr));
                }
#endif
            }
        }
        printf("     For array a[], %d errors were found.\n", ierr);
    }
    if (std::abs(bAvgErr / bj) > epsilon)
    {
        err++;
        printf("Failed Validation on array b[], AvgRelAbsErr > epsilon (%e)\n",
            epsilon);
        printf("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n", bj,
            bAvgErr, std::abs(bAvgErr) / bj);
        printf("     AvgRelAbsErr > Epsilon (%e)\n", epsilon);
        ierr = 0;
        for (std::size_t j = 0; j < a.size(); j++)
        {
            if (std::abs(b[j] / bj - 1.0) > epsilon)
            {
                ierr++;
#ifdef VERBOSE
                if (ierr < 10)
                {
                    printf("         array b: index: %ld, expected: %e, "
                           "observed: %e, relative error: %e\n",
                        (unsigned long) j, bj, b[j],
                        (double) std::abs((bj - b[j]) / bAvgErr));
                }
#endif
            }
        }
        printf("     For array b[], %d errors were found.\n", ierr);
    }
    if (std::abs(cAvgErr / cj) > epsilon)
    {
        err++;
        printf("Failed Validation on array c[], AvgRelAbsErr > epsilon (%e)\n",
            epsilon);
        printf("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n", cj,
            cAvgErr, std::abs(cAvgErr) / cj);
        printf("     AvgRelAbsErr > Epsilon (%e)\n", epsilon);
        ierr = 0;
        for (std::size_t j = 0; j < a.size(); j++)
        {
            if (std::abs(c[j] / cj - 1.0) > epsilon)
            {
                ierr++;
#ifdef VERBOSE
                if (ierr < 10)
                {
                    printf("         array c: index: %ld, expected: %e, "
                           "observed: %e, relative error: %e\n",
                        (unsigned long) j, cj, c[j],
                        (double) std::abs((cj - c[j]) / cAvgErr));
                }
#endif
            }
        }
        printf("     For array c[], %d errors were found.\n", ierr);
    }
    if (err == 0)
    {
        if (!csv)
        {
            printf("Solution Validates: avg error less than %e on all three "
                   "arrays\n",
                epsilon);
        }
    }
#ifdef VERBOSE
    printf("Results Validation Verbose Results: \n");
    printf("    Expected a(1), b(1), c(1): %f %f %f \n", aj, bj, cj);
    printf("    Observed a(1), b(1), c(1): %f %f %f \n", a[1], b[1], c[1]);
    printf("    Rel Errors on a, b, c:     %e %e %e \n",
        (double) std::abs(aAvgErr / aj), (double) std::abs(bAvgErr / bj),
        (double) std::abs(cAvgErr / cj));
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename T>
struct multiply_step
{
    multiply_step(T factor)
      : factor_(factor)
    {
    }

    // FIXME : call operator of multiply_step is momentarily defined with
    //         a generic parameter to allow the host_side invoke_result<>
    //         (used in invoke()) to get the return type

    template <typename U>
    HPX_HOST_DEVICE HPX_FORCEINLINE T operator()(U val) const
    {
        return val * factor_;
    }

    T factor_;
};

template <typename T>
struct add_step
{
    // FIXME : call operator of add_step is momentarily defined with
    //         generic parameters to allow the host_side invoke_result<>
    //         (used in invoke()) to get the return type

    template <typename U>
    HPX_HOST_DEVICE HPX_FORCEINLINE T operator()(U val1, U val2) const
    {
        return val1 + val2;
    }
};

template <typename T>
struct triad_step
{
    triad_step(T factor)
      : factor_(factor)
    {
    }

    // FIXME : call operator of triad_step is momentarily defined with
    //         generic parameters to allow the host_side invoke_result<>
    //         (used in invoke()) to get the return type

    template <typename U>
    HPX_HOST_DEVICE HPX_FORCEINLINE T operator()(U val1, U val2) const
    {
        return val1 + val2 * factor_;
    }

    T factor_;
};

///////////////////////////////////////////////////////////////////////////////
template <typename Allocator, typename Policy>
std::vector<std::vector<double>> run_benchmark(std::size_t iterations,
    std::size_t size, Allocator&& alloc, Policy&& policy)
{
    // Allocate our data
    using vector_type = hpx::compute::vector<STREAM_TYPE, Allocator>;

    vector_type a(size, alloc);
    vector_type b(size, alloc);
    vector_type c(size, alloc);

    // Initialize arrays
    hpx::fill(policy, a.begin(), a.end(), 1.0);
    hpx::fill(policy, b.begin(), b.end(), 2.0);
    hpx::fill(policy, c.begin(), c.end(), 0.0);

    // Check clock ticks ...
    double t = mysecond();
    hpx::transform(
        policy, a.begin(), a.end(), a.begin(), multiply_step<STREAM_TYPE>(2.0));
    t = 1.0E6 * (mysecond() - t);

    // Get initial value for system clock.
    int quantum = checktick();
    if (quantum >= 1)
    {
        if (!csv)
        {
            std::cout << "Your clock granularity/precision appears to be "
                      << quantum << " microseconds.\n";
        }
    }
    else
    {
        if (!csv)
        {
            std::cout << "Your clock granularity appears to be less than one "
                         "microsecond.\n";
        }
        quantum = 1;
    }

    if (!csv)
    {
        // clang-format off
        std::cout << "Each test below will take on the order of "
                  << (int) t << " microseconds.\n"
                  << "   (= " << (int) (t / quantum) << " clock ticks)\n"
                  << "Increase the size of the arrays if this shows that\n"
                  << "you are not getting at least 20 clock ticks per test.\n"
                  << "-------------------------------------------------------------\n";
        // clang-format on
    }

    if (!csv)
    {
        // clang-format off
        std::cout << "WARNING -- The above is only a rough guideline.\n"
                  << "For best results, please be sure you know the\n"
                  << "precision of your system timer.\n"
                  << "-------------------------------------------------------------\n";
        // clang-format on
    }

    ///////////////////////////////////////////////////////////////////////////
    // Main Loop
    std::vector<std::vector<double>> timing(4, std::vector<double>(iterations));

    double scalar = 3.0;
    for (std::size_t iteration = 0; iteration != iterations; ++iteration)
    {
        // Copy
        timing[0][iteration] = mysecond();
        hpx::copy(policy, a.begin(), a.end(), c.begin());
        timing[0][iteration] = mysecond() - timing[0][iteration];

        // Scale
        timing[1][iteration] = mysecond();
        hpx::transform(policy, c.begin(), c.end(), b.begin(),
            multiply_step<STREAM_TYPE>(scalar));
        timing[1][iteration] = mysecond() - timing[1][iteration];

        // Add
        timing[2][iteration] = mysecond();
        hpx::ranges::transform(policy, a.begin(), a.end(), b.begin(), b.end(),
            c.begin(), add_step<STREAM_TYPE>());
        timing[2][iteration] = mysecond() - timing[2][iteration];

        // Triad
        timing[3][iteration] = mysecond();
        hpx::ranges::transform(policy, b.begin(), b.end(), c.begin(), c.end(),
            a.begin(), triad_step<STREAM_TYPE>(scalar));
        timing[3][iteration] = mysecond() - timing[3][iteration];
    }

    // Check Results ...
    check_results(iterations, a, b, c);

    if (!csv)
    {
        // clang-format off
        std::cout << "-------------------------------------------------------------\n";
        // clang-format on
    }

    return timing;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    std::size_t vector_size = vm["vector_size"].as<std::size_t>();
    std::size_t offset = vm["offset"].as<std::size_t>();
    std::size_t iterations = vm["iterations"].as<std::size_t>();
    std::size_t chunk_size = vm["chunk_size"].as<std::size_t>();
    std::size_t executor = vm["executor"].as<std::size_t>();
    csv = vm.count("csv") > 0;
    header = vm.count("header") > 0;

    HPX_UNUSED(chunk_size);

    std::string chunker = vm["chunker"].as<std::string>();

    // clang-format off
    if (!csv)
    {
        std::cout
            << "-------------------------------------------------------------\n"
            << "Modified STREAM benchmark based on\nHPX version: "
                << hpx::build_string() << "\n"
            << "-------------------------------------------------------------\n"
            << "This system uses " << sizeof(STREAM_TYPE)
                << " bytes per array element.\n"
            << "-------------------------------------------------------------\n"
            << "Array size = " << vector_size << " (elements), "
               "Offset = " << offset << " (elements)\n"
            << "Memory per array = "
                << sizeof(STREAM_TYPE) * (vector_size / 1024. / 1024.) << " MiB "
            << "(= "
                <<  sizeof(STREAM_TYPE) * (vector_size / 1024. / 1024. / 1024.)
                << " GiB).\n"
            << "Each kernel will be executed " << iterations << " times.\n"
            << " The *best* time for each kernel (excluding the first iteration)\n"
            << " will be used to compute the reported bandwidth.\n"
            << "-------------------------------------------------------------\n"
            << "Number of Threads requested = "
                << hpx::get_os_thread_count() << "\n"
            << "Chunking policy requested: " << chunker << "\n"
            << "Executor requested: " << executor << "\n"
            << "-------------------------------------------------------------\n"
            ;
    }
    // clang-format on

    double time_total = mysecond();
    std::vector<std::vector<double>> timing;

#if defined(HPX_HAVE_MODULE_COMPUTE_CUDA)
    bool use_accel = false;
    if (vm.count("use-accelerator"))
        use_accel = true;

    if (use_accel)
    {
        using executor_type = hpx::cuda::experimental::concurrent_executor<>;
        using allocator_type = hpx::cuda::experimental::allocator<STREAM_TYPE>;

        // Get the cuda targets we want to run on
        hpx::cuda::experimental::target target;

        // Get the host targets we want to run on
        auto host_targets = hpx::compute::host::get_local_targets();

        allocator_type alloc(target);
        executor_type exec(target, host_targets);
        auto policy = hpx::execution::par.on(exec);

        // perform benchmark
        timing = run_benchmark<>(
            iterations, vector_size, std::move(alloc), std::move(policy));
        //iterations, vector_size, std::move(target));
    }
    else
#endif
    {
        if (executor == 0)
        {
            // Default parallel policy with serial allocator.
            timing = run_benchmark<>(iterations, vector_size,
                std::allocator<STREAM_TYPE>{}, hpx::execution::par);
        }
        else if (executor == 1)
        {
            // Block executor with block allocator.
            using executor_type = hpx::compute::host::block_executor<>;
            using allocator_type =
                hpx::compute::host::block_allocator<STREAM_TYPE>;

            auto numa_nodes = hpx::compute::host::numa_domains();
            allocator_type alloc(numa_nodes);
            executor_type exec(numa_nodes);
            auto policy = hpx::execution::par.on(exec);

            timing = run_benchmark<>(
                iterations, vector_size, std::move(alloc), std::move(policy));
        }
        else if (executor == 2)
        {
            // Default parallel policy and allocator with default parallel policy.
            auto policy = hpx::execution::par;
            hpx::compute::host::detail::policy_allocator<STREAM_TYPE,
                decltype(policy)>
                alloc(policy);

            timing = run_benchmark<>(
                iterations, vector_size, std::move(alloc), std::move(policy));
        }
        else if (executor == 3)
        {
            // Fork-join executor and allocator with fork-join executor.
            using executor_type =
                hpx::execution::experimental::fork_join_executor;

            executor_type exec;
            auto policy = hpx::execution::par.on(exec);
            hpx::compute::host::detail::policy_allocator<STREAM_TYPE,
                decltype(policy)>
                alloc(policy);

            timing = run_benchmark<>(
                iterations, vector_size, std::move(alloc), std::move(policy));
        }
        else
        {
            HPX_THROW_EXCEPTION(hpx::commandline_option_error, "hpx_main",
                "Invalid executor id given (0-3 allowed");
        }
    }
    time_total = mysecond() - time_total;

    /* --- SUMMARY --- */
    // clang-format off
    std::size_t const num_stream_tests = 4;
    const char *label[num_stream_tests] = {
        "Copy:      ",
        "Scale:     ",
        "Add:       ",
        "Triad:     "
    };
    // clang-format on

    const double bytes[num_stream_tests] = {
        2 * sizeof(STREAM_TYPE) * static_cast<double>(vector_size),
        2 * sizeof(STREAM_TYPE) * static_cast<double>(vector_size),
        3 * sizeof(STREAM_TYPE) * static_cast<double>(vector_size),
        3 * sizeof(STREAM_TYPE) * static_cast<double>(vector_size)};

    // Note: skip first iteration
    std::vector<double> avgtime(num_stream_tests, 0.0);
    std::vector<double> mintime(
        num_stream_tests, (std::numeric_limits<double>::max)());
    std::vector<double> maxtime(num_stream_tests, 0.0);
    for (std::size_t iteration = 1; iteration != iterations; ++iteration)
    {
        for (std::size_t j = 0; j < num_stream_tests; j++)
        {
            avgtime[j] = avgtime[j] + timing[j][iteration];
            mintime[j] = (std::min)(mintime[j], timing[j][iteration]);
            maxtime[j] = (std::max)(maxtime[j], timing[j][iteration]);
        }
    }

    if (csv)
    {
        if (header)
        {
            printf(
                "executor,threads,vector_size,copy_bytes,copy_bw,copy_avg,copy_"
                "min,copy_max,scale_bytes,scale_bw,scale_avg,scale_min,scale_"
                "max,add_bytes,add_bw,add_avg,add_min,add_max,triad_bytes,"
                "triad_"
                "bw,triad_avg,triad_min,triad_max\n");
        }
        std::size_t const num_executors = 5;
        const char* executors[num_executors] = {"parallel-serial", "block",
            "parallel-parallel", "fork_join_executor"};
        printf("%s,%zu,%zu,", executors[executor], hpx::get_os_thread_count(),
            vector_size);
    }
    else
    {
        printf(
            "Function    Best Rate MB/s  Avg time     Min time     Max time\n");
    }

    for (std::size_t j = 0; j < num_stream_tests; j++)
    {
        avgtime[j] = avgtime[j] / (double) (iterations - 1);

        if (csv)
        {
            printf("%.0f,%.2f,%.9f,%.9f,%.9f", bytes[j],
                1.0E-06 * bytes[j] / mintime[j], avgtime[j], mintime[j],
                maxtime[j]);
            printf(j < num_stream_tests - 1 ? "," : "\n");
        }
        else
        {
            printf("%s%12.1f  %11.6f  %11.6f  %11.6f\n", label[j],
                1.0E-06 * bytes[j] / mintime[j], avgtime[j], mintime[j],
                maxtime[j]);
        }
    }

    if (!csv)
    {
        std::cout << "\nTotal time: " << time_total
                  << " (per iteration: " << time_total / iterations << ")\n";
    }

    if (!csv)
    {
        // clang-format off
        std::cout << "-------------------------------------------------------------\n";
        // clang-format on
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace hpx::program_options;

    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()
        (   "csv", "output results as csv")
        (   "header", "print header for csv results")
        (   "vector_size",
            hpx::program_options::value<std::size_t>()->default_value(1024),
            "size of vector (default: 1024)")
        (   "offset",
            hpx::program_options::value<std::size_t>()->default_value(0),
            "offset (default: 0)")
        (   "iterations",
            hpx::program_options::value<std::size_t>()->default_value(10),
            "number of iterations to repeat each test. (default: 10)")
        (   "chunker",
            hpx::program_options::value<std::string>()->default_value("default"),
            "Which chunker to use for the parallel algorithms. "
            "possible values: dynamic, auto, guided. (default: default)")
        (   "chunk_size",
             hpx::program_options::value<std::size_t>()->default_value(0),
            "size of vector (default: 1024)")
        (   "executor",
            hpx::program_options::value<std::size_t>()->default_value(2),
            "executor to use (0-3) (default: 2, parallel_executor)")

#if defined(HPX_HAVE_COMPUTE)
        (   "use-accelerator",
            "Use this flag to run the stream benchmark on the GPU")
#endif
        ;
    // clang-format on

    // parse command line here to extract the necessary settings for HPX
    parsed_options opts = command_line_parser(argc, argv)
                              .allow_unregistered()
                              .options(cmdline)
                              .style(command_line_style::unix_style)
                              .run();

    variables_map vm;
    store(opts, vm);

    std::vector<std::string> cfg = {
        "hpx.numa_sensitive=2"    // no-cross NUMA stealing
    };

    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}
