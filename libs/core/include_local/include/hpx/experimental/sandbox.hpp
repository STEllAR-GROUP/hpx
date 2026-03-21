//  Copyright (c) 2020-2026 The STE||AR-Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/experimental/sandbox.hpp
/// \page hpx::experimental::sandbox
/// \headerfile hpx/experimental/sandbox.hpp
///
/// \brief HPX Sandbox Laboratory — environment introspection, parallel
///        benchmarking, and scaling telemetry for single-node environments.
///
/// This header provides a lightweight, header-only toolkit for
/// prototyping and benchmarking HPX parallel code in resource-constrained
/// environments such as Compiler Explorer (Godbolt). It includes:
///
///   - **Environment introspection** via hwloc topology queries
///   - **Automatic sandbox detection** for container / CI environments
///   - **Comparative benchmarking** (sequential vs. parallel) with
///     speedup and efficiency metrics
///   - **Console-friendly telemetry** formatted for fixed-width output
///
/// All functions must be called from within a running HPX runtime
/// (e.g. from \c hpx_main or an HPX thread).

#pragma once

#include <hpx/config.hpp>

#include <hpx/execution.hpp>
#include <hpx/runtime_local/runtime_local_fwd.hpp>
#include <hpx/topology/topology.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace hpx::experimental::sandbox {

    // ─── Data Structures ──────────────────────────────────────────────

    /// \brief Describes the hardware and runtime environment.
    struct environment_info
    {
        std::size_t cores = 0;          ///< Physical CPU cores
        std::size_t pus = 0;            ///< Processing units (hardware threads)
        std::size_t numa_nodes = 0;     ///< NUMA domains
        std::size_t hpx_workers = 0;    ///< Active HPX worker threads
        bool is_sandbox = false;        ///< Likely constrained environment

        /// \brief Print a formatted environment report.
        ///
        /// \param os  Output stream (defaults to \c std::cout).
        void print(std::ostream& os = std::cout) const
        {
            os << "\n";
            os << "=== HPX Sandbox --- Environment " << std::string(25, '=')
               << "\n";
            os << "  Cores:           " << cores << "\n";
            os << "  PUs:             " << pus << "\n";
            os << "  NUMA domains:    " << numa_nodes << "\n";
            os << "  HPX workers:     " << hpx_workers << "\n";
            os << "  Sandbox:         "
               << (is_sandbox ? "yes (constrained environment detected)" :
                                "no (full hardware access)")
               << "\n";
            os << std::string(56, '=') << "\n";
        }
    };

    /// \brief Holds the results of a comparative benchmark.
    struct benchmark_report
    {
        std::string label;              ///< User-provided description
        std::size_t iterations = 0;     ///< Measurement iterations
        double seq_median_ms = 0.0;     ///< Sequential median time (ms)
        double par_median_ms = 0.0;     ///< Parallel median time (ms)
        double speedup = 0.0;           ///< seq_time / par_time
        double efficiency_pct = 0.0;    ///< (speedup / workers) * 100
        std::size_t num_workers = 0;    ///< HPX workers during measurement

        /// \brief Print a formatted benchmark report.
        ///
        /// \param os  Output stream (defaults to \c std::cout).
        void print(std::ostream& os = std::cout) const
        {
            os << "\n";
            os << "=== HPX Sandbox --- Benchmark " << std::string(26, '=')
               << "\n";
            os << "  Label:           \"" << label << "\"\n";
            os << "  Iterations:      " << iterations << "\n";
            os << "  Workers:         " << num_workers << "\n";
            os << std::string(56, '-') << "\n";
            os << "  Sequential:      " << std::fixed << std::setprecision(3)
               << seq_median_ms << " ms\n";
            os << "  Parallel:        " << std::fixed << std::setprecision(3)
               << par_median_ms << " ms\n";
            os << std::string(56, '-') << "\n";
            os << "  Speedup:         " << std::fixed << std::setprecision(2)
               << speedup << "x\n";
            os << "  Efficiency:      " << std::fixed << std::setprecision(1)
               << efficiency_pct << "%\n";
            os << std::string(56, '-') << "\n";

            // Verdict
            os << "  Verdict:         ";
            if (efficiency_pct >= 80.0)
                os << "Excellent scaling";
            else if (efficiency_pct >= 60.0)
                os << "Good scaling";
            else if (efficiency_pct >= 40.0)
                os << "Moderate scaling (check granularity)";
            else if (speedup > 1.0)
                os << "Limited scaling (overhead-dominated)";
            else
                os << "No speedup (work too small or contention)";
            os << "\n";
            os << std::string(56, '=') << "\n";
        }
    };

    // ─── Environment Detection ────────────────────────────────────────

    namespace detail {

        inline bool check_sandbox_heuristic()
        {
            // 1. Explicit environment variable (e.g. set in Godbolt properties)
            if (std::getenv("COMPILER_EXPLORER") != nullptr)
                return true;

            // 2. User-controlled override
            if (std::getenv("HPX_SANDBOX") != nullptr)
                return true;

            // 3. Very limited hardware suggests a sandbox / container
            unsigned int hw = std::thread::hardware_concurrency();
            if (hw > 0 && hw <= 4)
                return true;

            return false;
        }

        // Measure a callable, returning wall time in milliseconds
        template <typename F>
        double time_once_ms(F&& fn)
        {
            auto start = std::chrono::high_resolution_clock::now();
            fn();
            auto end = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(end - start)
                .count();
        }

        // Run a callable N times and return the median wall time
        template <typename F>
        double median_ms(F&& fn, std::size_t iterations)
        {
            // Warmup (discard first run)
            fn();

            std::vector<double> times(iterations);
            for (std::size_t i = 0; i < iterations; ++i)
            {
                times[i] = time_once_ms(fn);
            }

            std::sort(times.begin(), times.end());
            return times[iterations / 2];
        }

    }    // namespace detail

    // ─── Public API ───────────────────────────────────────────────────

    /// \brief Detect and return information about the current environment.
    ///
    /// Queries the hwloc topology for hardware details and the HPX runtime
    /// for the number of active worker threads. Also applies heuristics to
    /// determine whether the environment is resource-constrained (sandbox).
    ///
    /// \note Must be called from within a running HPX runtime.
    ///
    /// \returns An \c environment_info struct describing the hardware
    ///          and runtime configuration.
    inline environment_info detect_environment()
    {
        environment_info info;

        auto const& topo = hpx::threads::create_topology();
        info.cores = topo.get_number_of_cores();
        info.pus = topo.get_number_of_pus();
        info.numa_nodes = topo.get_number_of_numa_nodes();
        info.hpx_workers = hpx::get_num_worker_threads();
        info.is_sandbox = detail::check_sandbox_heuristic();

        return info;
    }

    /// \brief Print a formatted environment report to the given stream.
    ///
    /// Convenience wrapper that calls \c detect_environment() and
    /// prints the result.
    ///
    /// \param os  Output stream (defaults to \c std::cout).
    ///
    /// \note Must be called from within a running HPX runtime.
    inline void describe_environment(std::ostream& os = std::cout)
    {
        detect_environment().print(os);
    }

    /// \brief Measure the median execution time of a callable.
    ///
    /// Runs the callable once for warmup, then \p iterations times,
    /// and returns the median wall-clock time in milliseconds.
    ///
    /// \tparam F       Callable type (must be invocable with no arguments).
    /// \param fn       The callable to measure.
    /// \param iterations Number of timed runs (default: 5).
    ///
    /// \returns Median wall-clock time in milliseconds.
    template <typename F>
    double measure(F&& fn, std::size_t iterations = 5)
    {
        return detail::median_ms(std::forward<F>(fn), iterations);
    }

    /// \brief Compare sequential and parallel execution of the same work.
    ///
    /// Runs \p seq_fn and \p par_fn each for \p iterations iterations
    /// (plus warmup), takes the median time of each, and computes
    /// speedup and parallel efficiency.
    ///
    /// Example usage:
    /// \code
    /// auto report = hpx::experimental::sandbox::benchmark(
    ///     "parallel for_each",
    ///     [&]() {
    ///         hpx::for_each(hpx::execution::seq,
    ///             v.begin(), v.end(), work);
    ///     },
    ///     [&]() {
    ///         hpx::for_each(hpx::execution::par,
    ///             v.begin(), v.end(), work);
    ///     });
    /// report.print();
    /// \endcode
    ///
    /// \tparam SeqFn   Callable type for the sequential baseline.
    /// \tparam ParFn   Callable type for the parallel variant.
    /// \param label    Human-readable name for this benchmark.
    /// \param seq_fn   Sequential callable (baseline).
    /// \param par_fn   Parallel callable (test).
    /// \param iterations Number of timed runs per variant (default: 5).
    ///
    /// \returns A \c benchmark_report with timing and scaling metrics.
    ///
    /// \note Must be called from within a running HPX runtime.
    template <typename SeqFn, typename ParFn>
    benchmark_report benchmark(std::string label, SeqFn&& seq_fn,
        ParFn&& par_fn, std::size_t iterations = 5)
    {
        benchmark_report report;
        report.label = std::move(label);
        report.iterations = iterations;
        report.num_workers = hpx::get_num_worker_threads();

        // Measure sequential baseline
        report.seq_median_ms = detail::median_ms(seq_fn, iterations);

        // Measure parallel variant
        report.par_median_ms = detail::median_ms(par_fn, iterations);

        // Compute scaling metrics
        if (report.par_median_ms > 0.0)
        {
            report.speedup = report.seq_median_ms / report.par_median_ms;
        }

        if (report.num_workers > 0)
        {
            report.efficiency_pct =
                (report.speedup / static_cast<double>(report.num_workers)) *
                100.0;
        }

        return report;
    }

}    // namespace hpx::experimental::sandbox
