//  Copyright (c) 2007-2014 Hartmut Kaiser
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
    template <typename R, typename F, typename FwdIter>
    boost::uint64_t add_ready_future(std::vector<hpx::future<R> >& workitems,
        F && f, FwdIter first, std::size_t count)
    {
        boost::uint64_t t = hpx::util::high_resolution_clock::now();
        R ret = f(first, count);
        t = (hpx::util::high_resolution_clock::now() - t);
        workitems.push_back(hpx::make_ready_future(ret));
        return t;
    }

    template <typename R, typename F, typename FwdIter>
    boost::uint64_t add_ready_future(std::vector<hpx::shared_future<R> >& workitems,
        F && f, FwdIter first, std::size_t count)
    {
        boost::uint64_t t = hpx::util::high_resolution_clock::now();
        R ret = f(first, count);
        t = (hpx::util::high_resolution_clock::now() - t);
        workitems.push_back(hpx::make_ready_future(ret));
        return t;
    }

    template <typename F, typename FwdIter>
    boost::uint64_t add_ready_future(std::vector<hpx::future<void> >& workitems,
        F && f, FwdIter first, std::size_t count)
    {
        boost::uint64_t t = hpx::util::high_resolution_clock::now();
        f(first, count);
        t = (hpx::util::high_resolution_clock::now() - t);
        workitems.push_back(hpx::make_ready_future());
        return t;
    }

    template <typename F, typename FwdIter>
    boost::uint64_t add_ready_future(std::vector<hpx::shared_future<void> >& workitems,
        F && f, FwdIter first, std::size_t count)
    {
        boost::uint64_t t = hpx::util::high_resolution_clock::now();
        f(first, count);
        t = (hpx::util::high_resolution_clock::now() - t);
        workitems.push_back(hpx::make_ready_future());
        return t;
    }

    // estimate a chunk size based on number of cores used
    template <typename ExPolicy, typename Future, typename F1, typename FwdIter>
        // requires traits::is_future<Future>
    std::size_t auto_chunk_size(
        ExPolicy const& policy,
        std::vector<Future>& workitems,
        F1 && f1, FwdIter& first, std::size_t& count)
    {
        std::size_t startup_size = 0; // one startup iteration
        std::size_t test_chunk_size = 1; //(std::max)(count / 1000, (std::size_t)1);
        
        // get executor
        threads::executor exec = policy.get_executor();
        
        // get number of cores available
        std::size_t const cores = hpx::get_os_thread_count(exec);
        
        // get target chunk time
        //TODO
        boost::chrono::nanoseconds desired_chunktime_ns = boost::chrono::nanoseconds(0);//policy.get_chunk_time();
        
        // If no chunktime is supplied, fall back to 64us * cores
        if(desired_chunktime_ns.count() == 0)
            desired_chunktime_ns = boost::chrono::nanoseconds(1000000);
        
        // make sure we have enough work left to actually run the benchmark
        if(count < test_chunk_size * cores + startup_size) return 0;
        
        // add startup iteration(s), as in some cases the first iteration(s)
        // are slower. (cache effects and stuff)
        if(startup_size > 0)
        {
            add_ready_future(workitems, f1, first, startup_size);
            std::advance(first, startup_size);
            count -= startup_size;
        }
        
        // run the benchmark iterations
        std::vector<Future> bench_futures(32);
        boost::uint64_t t = hpx::util::high_resolution_clock::now();
        for(int i = 0; i < cores; i++){
            bench_futures[i] = std::move(hpx::async(f1, first, test_chunk_size));
            // mark benchmarked items as processed
            std::advance(first, test_chunk_size);
            count -= test_chunk_size;
        }
        hpx::wait_any(bench_futures); 
        t = (hpx::util::high_resolution_clock::now() - t);
        
        for(int i = 0; i < cores; i++){
            workitems.push_back(std::move(bench_futures[i]));
        }

        // get the timer step size
        boost::uint64_t t_min = hpx::util::high_resolution_clock::min();
        
        // subtract 300 (constant measurement overhead)
        if(t <= 300)
        {
            t = 0;
        }
        else
        {
            t -= 300;
        }
        
        // if time was smaller than being able to measure, consider it to be
        // the smallest possible amount of time. this will get important,
        // and is an approximation necessary to prevent the creation of too
        // many asyncs.
        if(t <= t_min) t = t_min;
        
        // calculate number of chunks, round
        std::size_t num_chunks_dividend = static_cast<std::size_t>
                            (t * count);
        std::size_t num_chunks_divisor  = static_cast<std::size_t>
                            (test_chunk_size * desired_chunktime_ns.count());
        std::size_t num_chunks =
                            (num_chunks_dividend + num_chunks_divisor / 2) 
                                                        / num_chunks_divisor;
        
        // if benchmark returned smallest amount of time, prevent creation
        // of too many asyncs and do normal geometric distribution
        if((t <= t_min) && num_chunks > cores)
            num_chunks = cores;
        
        // prevent 0 chunks
        if(num_chunks == 0) num_chunks = 1;
        
        // calculate desired chunksize from number of chunks, ceil
        std::size_t chunksize = (count + num_chunks - 1) / num_chunks; 
        
        return chunksize;

    }

    template <typename ExPolicy, typename Future, typename F1,
        typename FwdIter>
        // requires traits::is_future<Future>
    std::size_t get_static_chunk_size(ExPolicy const& policy,
        std::vector<Future>& workitems,
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
                if (count > 100*cores)
                    chunk_size = auto_chunk_size(policy, workitems, f1,
                                                 first, count);

                if (chunk_size == 0)
                    chunk_size = (count + cores - 1) / cores;
            }
        }
        return chunk_size;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename F, typename FwdIter>
        // requires traits::is_future<Future>
    void add_ready_future_idx(std::vector<Future>& workitems,
        F && f, std::size_t base_idx, FwdIter first, std::size_t count)
    {
        workitems.push_back(
            hpx::make_ready_future(f(base_idx, first, count)));
    }

    template <typename F, typename FwdIter>
    void add_ready_future_idx(std::vector<hpx::future<void> >& workitems,
        F && f, std::size_t base_idx, FwdIter first, std::size_t count)
    {
        f(base_idx, first, count);
        workitems.push_back(hpx::make_ready_future());
    }

    template <typename F, typename FwdIter>
    void add_ready_future_idx(std::vector<hpx::shared_future<void> >& workitems,
        F && f, std::size_t base_idx, FwdIter first, std::size_t count)
    {
        f(base_idx, first, count);
        workitems.push_back(hpx::make_ready_future());
    }

    // estimate a chunk size based on number of cores used, take into
    // account base index
    template <typename Future, typename F1, typename FwdIter>
        // requires traits::is_future<Future>
    std::size_t auto_chunk_size_idx(
        std::vector<Future>& workitems, F1 && f1,
        std::size_t& base_idx, FwdIter& first, std::size_t& count)
    {
        std::size_t test_chunk_size = count / 100;
        if (0 == test_chunk_size) return 0;

        boost::uint64_t t = hpx::util::high_resolution_clock::now();
        add_ready_future_idx(workitems, f1, base_idx, first, test_chunk_size);

        t = (hpx::util::high_resolution_clock::now() - t) / test_chunk_size;

        base_idx += test_chunk_size;
        std::advance(first, test_chunk_size);
        count -= test_chunk_size;

        // return chunk size which will create 80 microseconds of work
        return t == 0 ? 0 : (std::min)(count, (std::size_t)(80000 / t));
    }

    template <typename ExPolicy, typename Future, typename F1,
        typename FwdIter>
        // requires traits::is_future<Future>
    std::size_t get_static_chunk_size_idx(ExPolicy const& policy,
        std::vector<Future>& workitems,
        F1 && f1, std::size_t& base_idx, FwdIter& first,
        std::size_t& count, std::size_t chunk_size)
    {
        threads::executor exec = policy.get_executor();
        if (chunk_size == 0)
        {
            chunk_size = policy.get_chunk_size();
            if (chunk_size == 0)
            {
                std::size_t const cores = hpx::get_os_thread_count(exec);
                if (count > 100*cores)
                    chunk_size = auto_chunk_size_idx(workitems, f1,
                        base_idx, first, count);

                if (chunk_size == 0)
                    chunk_size = (count + cores - 1) / cores;
            }
        }
        return chunk_size;
    }
}}}}

#endif
