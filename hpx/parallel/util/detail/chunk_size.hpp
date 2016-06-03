//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_DETAIL_AUTO_CHUNK_SIZE_OCT_03_2014_0159PM)
#define HPX_PARALLEL_UTIL_DETAIL_AUTO_CHUNK_SIZE_OCT_03_2014_0159PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/decay.hpp>

#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/executors/executor_parameter_traits.hpp>
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/algorithms/detail/is_negative.hpp>

#include <algorithm>
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

    template <typename ExPolicy, typename Future, typename F1,
        typename FwdIter, typename Stride>
        // requires traits::is_future<Future>
    std::vector<hpx::util::tuple<FwdIter, std::size_t> >
    get_bulk_iteration_shape(
        ExPolicy && policy, std::vector<Future>& workitems, F1 && f1,
        FwdIter& first, std::size_t& count, Stride s)
    {
        typedef typename hpx::util::decay<ExPolicy>::type::executor_parameters_type
            parameters_type;
        typedef executor_parameter_traits<parameters_type> traits;
        typedef hpx::util::tuple<FwdIter, std::size_t> tuple_type;

        typedef typename hpx::util::decay<ExPolicy>::type::executor_type
            executor_type;
        std::size_t const cores = executor_information_traits<executor_type>::
            processing_units_count(policy.executor(), policy.parameters());

        bool variable_chunk_sizes = traits::variable_chunk_size(
            policy.parameters(), policy.executor());

        std::vector<tuple_type> shape;

        Stride stride = parallel::v1::detail::abs(s);
        if (!variable_chunk_sizes)
        {
            auto test_function =
                [&]() -> std::size_t
                {
                    std::size_t test_chunk_size = count / 100;
                    if (test_chunk_size == 0)
                        return 0;

                    if (stride != 1)
                    {
                        test_chunk_size = (std::max)(std::size_t(stride),
                            (test_chunk_size / stride) * stride);
                    }

                    add_ready_future(workitems, f1, first, test_chunk_size);

                    // modifies 'test_chunk_size'
                    first = parallel::v1::detail::next(
                        first, count, test_chunk_size);

                    count -= test_chunk_size;
                    return test_chunk_size;
                };

            std::size_t chunk_size =
                traits::get_chunk_size(policy.parameters(),
                    policy.executor(), test_function, count);

            if (chunk_size == 0)
                chunk_size = (count + cores - 1) / cores;

            if (stride != 1)
            {
                chunk_size = (std::max)(std::size_t(stride),
                    (chunk_size / stride) * stride);
            }

            shape.reserve(count / chunk_size + 1);
            while (count != 0)
            {
                std::size_t chunk = (std::min)(chunk_size, count);

                shape.push_back(hpx::util::make_tuple(first, chunk));

                // modifies 'chunk'
                first = parallel::v1::detail::next(first, count, chunk);
                count -= chunk;
            }
        }
        else
        {
            while (count != 0)
            {
                std::size_t chunk_size =
                    traits::get_chunk_size(policy.parameters(),
                        policy.executor(), [](){ return 0; }, count);

                if (chunk_size == 0)
                    chunk_size = (count + cores - 1) / cores;

                if (stride != 1)
                {
                    chunk_size = (std::max)(std::size_t(stride),
                        (chunk_size / stride) * stride);
                }

                std::size_t chunk = (std::min)(chunk_size, count);

                shape.push_back(hpx::util::make_tuple(first, chunk));

                // modifies 'chunk'
                first = parallel::v1::detail::next(first, count, chunk);
                count -= chunk;
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
        typename FwdIter, typename Stride>
        // requires traits::is_future<Future>
    std::vector<hpx::util::tuple<std::size_t, FwdIter, std::size_t > >
    get_bulk_iteration_shape_idx(
        ExPolicy && policy, std::vector<Future>& workitems, F1 && f1,
        FwdIter& first, std::size_t& count, Stride s)
    {
        typedef typename hpx::util::decay<ExPolicy>::type::executor_parameters_type
            parameters_type;
        typedef executor_parameter_traits<parameters_type> traits;
        typedef hpx::util::tuple<std::size_t, FwdIter, std::size_t> tuple_type;

        typedef typename hpx::util::decay<ExPolicy>::type::executor_type
            executor_type;
        std::size_t const cores = executor_information_traits<executor_type>::
            processing_units_count(policy.executor(), policy.parameters());

        bool variable_chunk_sizes = traits::variable_chunk_size(
            policy.parameters(), policy.executor());

        std::vector<tuple_type> shape;

        Stride stride = parallel::v1::detail::abs(s);
        std::size_t base_idx = 0;
        if (!variable_chunk_sizes)
        {
            auto test_function =
                [&]() -> std::size_t
                {
                    std::size_t test_chunk_size = count / 100;
                    if (test_chunk_size == 0)
                        return 0;

                    if (stride != 1)
                    {
                        test_chunk_size = (std::max)(std::size_t(stride),
                            (test_chunk_size / stride) * stride);
                    }

                    add_ready_future_idx(workitems, f1, base_idx, first,
                        test_chunk_size);

                    // modifies 'test_chunk_size'
                    first = parallel::v1::detail::next(
                        first, count, test_chunk_size);

                    base_idx += test_chunk_size;
                    count -= test_chunk_size;

                    return test_chunk_size;
                };

            std::size_t chunk_size =
                traits::get_chunk_size(policy.parameters(),
                    policy.executor(), test_function, count);

            if (chunk_size == 0)
                chunk_size = (count + cores - 1) / cores;

            if (stride != 1)
            {
                chunk_size = (std::max)(std::size_t(stride),
                    (chunk_size / stride) * stride);
            }

            shape.reserve(count / (chunk_size + 1));
            while (count != 0)
            {
                std::size_t chunk = (std::min)(chunk_size, count);

                shape.push_back(hpx::util::make_tuple(base_idx, first, chunk));

                // modifies 'chunk'
                first = parallel::v1::detail::next(first, count, chunk);

                count -= chunk;
                base_idx += chunk;
            }
        }
        else
        {
            while (count != 0)
            {
                std::size_t chunk_size =
                    traits::get_chunk_size(policy.parameters(),
                        policy.executor(), [](){ return 0; }, count);

                if (chunk_size == 0)
                    chunk_size = (count + cores - 1) / cores;

                if (stride != 1)
                {
                    chunk_size = (std::max)(std::size_t(stride),
                        (chunk_size / stride) * stride);
                }

                std::size_t chunk = (std::min)(chunk_size, count);

                shape.push_back(hpx::util::make_tuple(base_idx, first, chunk));

                // modifies 'chunk'
                first = parallel::v1::detail::next(first, count, chunk);

                count -= chunk;
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

        // Take a standard chunk size ( amount of work / cores ), and only
        // take half of that. If our chunk size is too large a LOT of the work
        // will be done sequentially due to the level barrier of heap parallelism.
        // 1/2 of the standard chunk size is an estimate to lower the average
        // number of levels done sequentially
        if(chunk_size == 0)
            chunk_size = (count + cores - 1) / (cores * 2);

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
        RndIter& first, std::size_t count, std::size_t& chunk_size)
    {
        typedef typename ExPolicy::executor_parameters_type parameters_type;
        typedef typename std::iterator_traits<RndIter>::difference_type dtyle;
        typedef executor_parameter_traits<parameters_type> traits;
        typedef hpx::util::tuple<RndIter, std::size_t> tuple_type;

        typedef typename ExPolicy::executor_type executor_type;
        std::size_t const cores = executor_information_traits<executor_type>::
            processing_units_count(policy.executor(), policy.parameters());

        std::vector<tuple_type> shape;

        // TO-DO
        // write test function for get_chunk_size, though this is a bit
        // tricky because chunking a heap algorithm is very different
        // from other algorithms

        // Take a standard chunk size ( amount of work / cores ), and only
        // take half of that. If our chunk size is too large a LOT of the work
        // will be done sequentially due to the level barrier of heap parallelism.
        // 1/2 of the standard chunk size is an estimate to lower the average
        // number of levels done sequentially
        if(chunk_size == 0)
            chunk_size = (count + cores - 1) / (cores * 2);

        std::size_t start = (count-2)/2;
        while(start > 0) {
            // Index of start of level, and amount of items in level
            std::size_t end_exclusive = (std::size_t)pow(2, floor(log2(start)))-2;
            std::size_t level_items = (start - end_exclusive);

            // If we can't at least run two chunks in parallel, don't bother
            // parallelizng and simply run sequentially
            if(chunk_size * 2 > level_items) {
                f1(first+start, level_items);
            } else {
                shape.push_back(hpx::util::make_tuple(first+start, level_items));
            }

            start = end_exclusive;
        }

        // Perform f1 on head node
        f1(first, 1);
        return shape;
    }
}}}}

#endif
