//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_DETAIL_AUTO_CHUNK_SIZE_OCT_03_2014_0159PM)
#define HPX_PARALLEL_UTIL_DETAIL_AUTO_CHUNK_SIZE_OCT_03_2014_0159PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/util/tuple.hpp>

#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/executors/executor_parameter_traits.hpp>

#include <vector>
#include <algorithm>

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

    template <typename ExPolicy, typename Future, typename F1,
        typename FwdIter>
        // requires traits::is_future<Future>
    std::vector<hpx::util::tuple<FwdIter, std::size_t> >
    get_bulk_iteration_shape(
        ExPolicy policy, std::vector<Future>& workitems, F1 && f1,
        FwdIter& first, std::size_t& count, std::size_t chunk_size)
    {
        typedef typename ExPolicy::executor_parameters_type parameters_type;
        typedef executor_parameter_traits<parameters_type> traits;
        typedef hpx::util::tuple<FwdIter, std::size_t> tuple_type;

        typedef typename ExPolicy::executor_type executor_type;
        std::size_t const cores = executor_information_traits<executor_type>::
            processing_units_count(policy.executor(), policy.parameters());

        bool variable_chunk_sizes = traits::variable_chunk_size(
            policy.parameters(), policy.executor());

        std::vector<tuple_type> shape;

        if (!variable_chunk_sizes || chunk_size != 0)
        {
            if (chunk_size == 0)
            {
                auto test_function =
                    [&]() -> std::size_t
                    {
                        std::size_t test_chunk_size = count / 100;
                        if (test_chunk_size == 0)
                            return 0;

                        add_ready_future(workitems, f1, first, test_chunk_size);

                        std::advance(first, test_chunk_size);
                        count -= test_chunk_size;

                        return test_chunk_size;
                    };

                chunk_size = traits::get_chunk_size(policy.parameters(),
                    policy.executor(), test_function, count);
            }

            if (chunk_size == 0)
                chunk_size = (count + cores - 1) / cores;

            shape.reserve(count / chunk_size + 1);
            while (count != 0)
            {
                std::size_t chunk = (std::min)(chunk_size, count);

                shape.push_back(hpx::util::make_tuple(first, chunk));
                count -= chunk;
                std::advance(first, chunk);
            }
        }
        else
        {
            while (count != 0)
            {
                chunk_size = traits::get_chunk_size(
                    policy.parameters(), policy.executor(),
                    [](){ return 0; }, count);

                if (chunk_size == 0)
                    chunk_size = (count + cores - 1) / cores;

                std::size_t chunk = (std::min)(chunk_size, count);

                shape.push_back(hpx::util::make_tuple(first, chunk));
                count -= chunk;
                std::advance(first, chunk);
            }
        }

        return shape;
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

    template <typename ExPolicy, typename Future, typename F1,
        typename FwdIter>
        // requires traits::is_future<Future>
    std::vector<hpx::util::tuple<std::size_t, FwdIter, std::size_t > >
    get_bulk_iteration_shape_idx(
        ExPolicy policy, std::vector<Future>& workitems, F1 && f1,
        std::size_t& base_idx, FwdIter& first, std::size_t& count,
        std::size_t chunk_size)
    {
        typedef typename ExPolicy::executor_parameters_type parameters_type;
        typedef executor_parameter_traits<parameters_type> traits;
        typedef hpx::util::tuple<std::size_t, FwdIter, std::size_t> tuple_type;

        bool variable_chunk_sizes = traits::variable_chunk_size(
            policy.parameters(), policy.executor());

        std::vector<tuple_type> shape;

        if (!variable_chunk_sizes || chunk_size != 0)
        {
            if (chunk_size == 0)
            {
                auto test_function =
                    [&]() -> std::size_t
                    {
                        std::size_t test_chunk_size = count / 100;
                        if (test_chunk_size == 0)
                            return 0;

                        add_ready_future_idx(workitems, f1, base_idx, first,
                            test_chunk_size);

                        base_idx += test_chunk_size;
                        std::advance(first, test_chunk_size);
                        count -= test_chunk_size;

                        return test_chunk_size;
                    };

                chunk_size = traits::get_chunk_size(policy.parameters(),
                    policy.executor(), test_function, count);
            }

            shape.reserve(count / (chunk_size + 1));
            while (count != 0)
            {
                std::size_t chunk = (std::min)(chunk_size, count);

                shape.push_back(hpx::util::make_tuple(base_idx, first, chunk));

                count -= chunk;
                std::advance(first, chunk);
                base_idx += chunk;
            }
        }
        else
        {
            while (count != 0)
            {
                chunk_size = traits::get_chunk_size(
                    policy.parameters(), policy.executor(),
                    [](){ return 0; }, count);

                std::size_t chunk = (std::min)(chunk_size, count);

                shape.push_back(hpx::util::make_tuple(base_idx, first, chunk));
                count -= chunk;
                std::advance(first, chunk);
                base_idx += chunk;
            }
        }

        return shape;
    }

    template <typename ExPolicy, typename Future, typename F1,
        typename RndIter>
        // requires traits::is_future<Future>
    std::vector<hpx::util::tuple<RndIter, std::size_t> >
    get_topdown_heap_bulk_iteration_shape(
        ExPolicy policy, std::vector<Future>& workitems, F1 && f1,
        RndIter first, std::size_t count, std::size_t& chunk_size)
    {
        typedef typename ExPolicy::executor_parameters_type parameters_type;
        typedef executor_parameter_traits<parameters_type> traits;
        typedef hpx::util::tuple<RndIter, std::size_t> tuple_type;
        typedef typename std::iterator_traits<RndIter>::difference_type dtype;

        typedef typename ExPolicy::executor_type executor_type;
        std::size_t const cores = executor_information_traits<executor_type>::
            processing_units_count(policy.executor(), policy.parameters());

        std::vector<tuple_type> shape;

        // TO-DO
        // write test function for get_chunk_size, though this is a bit 
        // tricky because chunking a heap algorithm is very different from
        // other algorithms

        // Take a standard chunk size ( amount of work / cores ), and multiply
        // that by a fourth. If our chunk size is 1/4 of the work a LOT of the 
        // work will be done sequentially due to the level barrier of heap
        // parallelism, so I chose a 1/4th for now as it's a much smaller size
        if(chunk_size == 0)
            chunk_size = (count + cores - 1) / (cores * 4);

        std::size_t level = 1;
        while(level < (std::size_t)ceil(log2(count))) {
            // Index of start of level, and amount of items in the level
            std::size_t start = (std::size_t)pow(2, level-1)-1;
            std::size_t level_items = ((std::size_t)pow(2, level)-1) - start;
            // If we can't at least run two chunks in parallel, don't bother 
            // parallelizing and simply run sequentially
            if(chunk_size * 2 > level_items) {
                f1(first, level_items);
                std::advance(first, level_items);
            } else {
                shape.push_back(hpx::util::make_tuple(first, level_items));
                std::advance(first, level_items);
            }
            level++;
        }

        // Perform the operation above, but for the very last level which 
        // requires a special check in that it may not full
        std::size_t start = (std::size_t)pow(2, level-1)-1;
        if(chunk_size * 2 > count - start) {
            f1(first, (count-start));
            std::advance(first, (count-start));
        } else {
            shape.push_back(hpx::util::make_tuple(first, count-start));
            std::advance(first, (count-start));
        }

        return shape;
    }

    template <typename ExPolicy, typename Future, typename F1,
        typename RndIter>
        // requires traits::is_future<Future>
    std::vector<hpx::util::tuple<RndIter, std::size_t> >
    get_bottomup_heap_bulk_iteration_shape(
        ExPolicy policy, std::vector<Future>& workitems, F1 && f1,
        RndIter& first, std::size_t& count, std::size_t chunk_size)
    {

    }
}}}}

#endif
