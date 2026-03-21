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
#include <hpx/format.hpp>
#include <hpx/runtime_local/runtime_local_fwd.hpp>
#include <hpx/topology/topology.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iosfwd>
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
        /// \param os  Output stream.
        void print(std::ostream& os) const
        {
            hpx::util::format_to(os, "\n");
            hpx::util::format_to(os, "=== HPX Sandbox --- Environment {}\n",
                std::string(25, '='));
            hpx::util::format_to(os, "  Cores:           {}\n", cores);
            hpx::util::format_to(os, "  PUs:             {}\n", pus);
            hpx::util::format_to(os, "  NUMA domains:    {}\n", numa_nodes);
            hpx::util::format_to(os, "  HPX workers:     {}\n", hpx_workers);
            hpx::util::format_to(os, "  Sandbox:         {}\n",
                is_sandbox ? "yes (constrained environment detected)" :
                             "no (full hardware access)");
            hpx::util::format_to(os, "{}\n", std::string(56, '='));
        }
    };

    /// \brief Holds the results of a comparative benchmark.
    struct benchmark_report
    {
        std::string label;              ///< User-provided description
        std::size_t iterations = 0;     ///< Measurement iterations
        double seq_mean_ms = 0.0;       ///< Sequential mean time (ms)
        double par_mean_ms = 0.0;       ///< Parallel mean time (ms)
        double speedup = 0.0;           ///< seq_time / par_time
        double efficiency_pct = 0.0;    ///< (speedup / workers) * 100
        std::size_t num_workers = 0;    ///< HPX workers during measurement

        /// \brief Print a formatted benchmark report.
        ///
        /// \param os  Output stream.
        void print(std::ostream& os) const
        {
            hpx::util::format_to(os, "\n");
            hpx::util::format_to(
                os, "=== HPX Sandbox --- Benchmark {}\n", std::string(26, '='));
            hpx::util::format_to(os, "  Label:           \"{}\"\n", label);
            hpx::util::format_to(os, "  Iterations:      {}\n", iterations);
            hpx::util::format_to(os, "  Workers:         {}\n", num_workers);
            hpx::util::format_to(os, "{}\n", std::string(56, '-'));
            hpx::util::format_to(
                os, "  Sequential:      {:.3f} ms\n", seq_mean_ms);
            hpx::util::format_to(
                os, "  Parallel:        {:.3f} ms\n", par_mean_ms);
            hpx::util::format_to(os, "{}\n", std::string(56, '-'));
            hpx::util::format_to(os, "  Speedup:         {:.2f}x\n", speedup);
            hpx::util::format_to(
                os, "  Efficiency:      {:.1f}%\n", efficiency_pct);
            hpx::util::format_to(os, "{}\n", std::string(56, '-'));

            // Verdict
            hpx::util::format_to(os, "  Verdict:         ");
            if (efficiency_pct >= 80.0)
                hpx::util::format_to(os, "Excellent scaling\n");
            else if (efficiency_pct >= 60.0)
                hpx::util::format_to(os, "Good scaling\n");
            else if (efficiency_pct >= 40.0)
                hpx::util::format_to(
                    os, "Moderate scaling (check granularity)\n");
            else if (speedup > 1.0)
                hpx::util::format_to(
                    os, "Limited scaling (overhead-dominated)\n");
            else
                hpx::util::format_to(
                    os, "No speedup (work too small or contention)\n");
            hpx::util::format_to(os, "{}\n", std::string(56, '='));
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

        // Run a callable N times and return the mean wall time
        template <typename F>
        double mean_ms(F&& fn, std::size_t iterations)
        {
            // Warmup (discard first run)
            fn();

            double total = 0.0;
            for (std::size_t i = 0; i < iterations; ++i)
            {
                total += time_once_ms(fn);
            }

            return total / static_cast<double>(iterations);
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
    /// \param os  Output stream.
    ///
    /// \note Must be called from within a running HPX runtime.
    inline void describe_environment(std::ostream& os)
    {
        detect_environment().print(os);
    }

    /// \brief Measure the mean execution time of a callable.
    ///
    /// Runs the callable once for warmup, then \p iterations times,
    /// and returns the mean wall-clock time in milliseconds.
    ///
    /// \tparam F       Callable type (must be invocable with no arguments).
    /// \param fn       The callable to measure.
    /// \param iterations Number of timed runs (default: 5).
    ///
    /// \returns Mean wall-clock time in milliseconds.
    template <typename F>
    double measure(F&& fn, std::size_t iterations = 5)
    {
        return detail::mean_ms(std::forward<F>(fn), iterations);
    }

    /// \brief Compare sequential and parallel execution of the same work.
    ///
    /// Runs \p seq_fn and \p par_fn each for \p iterations iterations
    /// (plus warmup), takes the mean time of each, and computes
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
    /// report.print(std::cout);
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
        report.seq_mean_ms = detail::mean_ms(seq_fn, iterations);

        // Measure parallel variant
        report.par_mean_ms = detail::mean_ms(par_fn, iterations);

        // Compute scaling metrics
        if (report.par_mean_ms > 0.0)
        {
            report.speedup = report.seq_mean_ms / report.par_mean_ms;
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
