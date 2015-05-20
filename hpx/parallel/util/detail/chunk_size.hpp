//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2014-2015 Martin Stumpf
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_DETAIL_AUTO_CHUNK_SIZE_OCT_03_2014_0159PM)
#define HPX_PARALLEL_UTIL_DETAIL_AUTO_CHUNK_SIZE_OCT_03_2014_0159PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // estimate a chunk size by running a quick benchmark
    template <typename ExPolicy, typename Future, typename F1, typename FwdIter,
        typename StartBenchThread>
        // requires traits::is_future<Future>
        // requires is_execution_policy<ExPolicy>
    std::size_t auto_chunk_size(
        ExPolicy const& policy, std::vector<Future>& workitems,
        F1 && f1, FwdIter& first, std::size_t& count,
        StartBenchThread && start_bench_thread)
    {
        // get executor
        threads::executor exec = policy.get_executor();

        // get number of cores available
        std::size_t const cores = hpx::get_os_thread_count(exec);

        // get benchmark size. best value found so far: count/3
        std::size_t const test_chunk_size =
            (std::max)(count / (3*cores), std::size_t(1));

        // get target chunk time, TODO implement, if requested
        boost::chrono::nanoseconds desired_chunktime_ns =
            boost::chrono::nanoseconds(0);//policy.get_chunk_time();

        // If no chunktime is supplied, fall back to 64us * cores
        if (desired_chunktime_ns.count() == 0)
        {
            // TODO subject to change, further measurements necessary
            using boost::chrono::nanoseconds;
            desired_chunktime_ns = nanoseconds(200000 + cores * 25000);
        }

        // make sure we have enough work left to actually run the benchmark
        if (count < test_chunk_size * cores) return 0;

        // run the benchmark iterations
        std::vector<Future> bench_futures;
        bench_futures.reserve(cores);

        boost::uint64_t t = hpx::util::high_resolution_clock::now();

        for (std::size_t i = 0; i != cores; ++i)
            start_bench_thread(bench_futures, f1, first, count, test_chunk_size);
        hpx::wait_any(bench_futures);

        t = hpx::util::high_resolution_clock::now() - t;

        // pass on all of the threads created for benchmarking
        workitems.reserve(cores);
        for (std::size_t i = 0; i != cores; ++i)
            workitems.push_back(std::move(bench_futures[i]));

        // get the timer step size
        t = (t <= 300) ? 0 : t - 300;

        // If time was smaller than we're able to measure, consider it to be
        // the smallest possible amount of time. This will get important,
        // and is an approximation necessary to prevent the creation of too
        // many asyncs.
        boost::uint64_t t_min = (hpx::util::high_resolution_clock::min)();
        if (t <= t_min) t = t_min;

        // calculate number of chunks, round
        std::size_t num_chunks_dividend =
            static_cast<std::size_t>(t * count);
        std::size_t num_chunks_divisor =
            static_cast<std::size_t>(test_chunk_size * desired_chunktime_ns.count());
        std::size_t num_chunks =
            (num_chunks_dividend + num_chunks_divisor / 2) / num_chunks_divisor;

        // If the benchmark returned the smallest amount of time, prevent
        // creation of too many asyncs and do normal geometric distribution.
        if (t <= t_min && num_chunks > cores)
            num_chunks = cores;

        // prevent 0 chunks
        if (num_chunks == 0) num_chunks = 1;

        // calculate desired chunk size from number of chunks, ceil
        return (count + num_chunks - 1) / num_chunks;
    }

    template <typename ExPolicy, typename Future, typename F1,
        typename FwdIter>
        // requires traits::is_future<Future>
        // requires is_execution_policy<ExPolicy>
    std::size_t get_static_chunk_size(
        ExPolicy const& policy, std::vector<Future>& workitems,
        F1 && f1, FwdIter& first, std::size_t& count,
        std::size_t chunk_size)
    {
        threads::executor exec = policy.get_executor();
        if (chunk_size == 0)
        {
            chunk_size = policy.get_chunk_size();
            if (chunk_size == 0)
            {
                std::size_t const cores = hpx::get_os_thread_count(exec) * 2;
                if (count > 100 * cores)
                {
                    chunk_size = auto_chunk_size(
                        policy, workitems, f1, first, count,
                        [&exec] (std::vector<Future>& bench_futures,
                            F1 && f1, FwdIter& first, std::size_t& count,
                            std::size_t test_chunk_size)
                        {
                            if (exec)
                            {
                                bench_futures.push_back(hpx::async(exec, f1,
                                    first, test_chunk_size));
                            }
                            else
                            {
                                bench_futures.push_back(hpx::async(f1, first,
                                    test_chunk_size));
                            }

                            // mark benchmarked items as processed
                            std::advance(first, test_chunk_size);
                            count -= test_chunk_size;
                        });
                }
                if (chunk_size == 0)
                    chunk_size = (count + cores - 1) / cores;
            }
        }
        return chunk_size;
    }

    template <typename ExPolicy, typename Future, typename F1,
        typename FwdIter>
        // requires traits::is_future<Future>
        // requires is_execution_policy<ExPolicy>
    std::size_t get_static_chunk_size_idx(
        ExPolicy const& policy, std::vector<Future>& workitems,
        F1 && f1, std::size_t& base_idx, FwdIter& first,
        std::size_t& count, std::size_t chunk_size)
    {
        threads::executor exec = policy.get_executor();
        if (chunk_size == 0)
        {
            chunk_size = policy.get_chunk_size();
            if (chunk_size == 0)
            {
                std::size_t const cores = hpx::get_os_thread_count(exec) * 2;
                if (count > 100 * cores)
                {
                    chunk_size = auto_chunk_size(
                        policy, workitems, f1, first, count,
                        [&base_idx, &exec] (std::vector<Future>& bench_futures,
                            F1 && f1, FwdIter& first, std::size_t& count,
                            std::size_t test_chunk_size)
                        {
                            if (exec)
                            {
                                bench_futures.push_back(hpx::async(exec, f1,
                                    base_idx, first, test_chunk_size));
                            }
                            else
                            {
                                bench_futures.push_back(hpx::async(f1,
                                    base_idx, first, test_chunk_size));
                            }

                            // mark benchmarked items as processed
                            base_idx += test_chunk_size;
                            std::advance(first, test_chunk_size);
                            count -= test_chunk_size;
                        });
                }
                if (chunk_size == 0)
                    chunk_size = (count + cores - 1) / cores;
            }
        }
        return chunk_size;
    }
}}}}

#endif
