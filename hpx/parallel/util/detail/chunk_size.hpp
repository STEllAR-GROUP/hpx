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
    template <typename F, typename Future, typename FwdIter>
        // requires traits::is_future<Future>
    void add_ready_future(std::vector<Future>& workitems,
        F && f, FwdIter first, std::size_t count)
    {
        workitems.push_back(hpx::make_ready_future(f(first, count)));
    }

    template <typename F, typename FwdIter>
    void add_ready_future(std::vector<hpx::future<void> >& workitems,
        F && f, FwdIter first, std::size_t count)
    {
        f(first, count);
        workitems.push_back(hpx::make_ready_future());
    }

    template <typename F, typename FwdIter>
    void add_ready_future(std::vector<hpx::shared_future<void> >& workitems,
        F && f, FwdIter first, std::size_t count)
    {
        f(first, count);
        workitems.push_back(hpx::make_ready_future());
    }

    // estimate a chunk size based on number of cores used
    template <typename Future, typename F1, typename FwdIter>
        // requires traits::is_future<Future>
    std::size_t auto_chunk_size(
        std::vector<Future>& workitems,
        F1 && f1, FwdIter& first, std::size_t& count)
    {
        std::size_t test_chunk_size = count / 100;
        if (0 == test_chunk_size) return 0;

        boost::uint64_t t = hpx::util::high_resolution_clock::now();
        add_ready_future(workitems, f1, first, test_chunk_size);

        t = (hpx::util::high_resolution_clock::now() - t) / test_chunk_size;

        std::advance(first, test_chunk_size);
        count -= test_chunk_size;

        // return chunk size which will create 80 microseconds of work
        return t == 0 ? 0 : (std::min)(count, (std::size_t)(80000 / t));
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
                std::size_t const cores = hpx::get_os_thread_count(exec);
                if (count > 100*cores)
                    chunk_size = auto_chunk_size(workitems, f1, first, count);

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
