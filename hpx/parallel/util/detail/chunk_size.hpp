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
#include <hpx/parallel/algorithms/detail/predicates.hpp>

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
        FwdIter& first, std::size_t& count)
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

        if (!variable_chunk_sizes)
        {
            auto test_function =
                [&]() -> std::size_t
                {
                    std::size_t test_chunk_size = count / 100;
                    if (test_chunk_size == 0)
                        return 0;

                    add_ready_future(workitems, f1, first, test_chunk_size);

                    first = parallel::v1::detail::next(first, test_chunk_size);
                    count -= test_chunk_size;

                    return test_chunk_size;
                };

            std::size_t chunk_size =
                traits::get_chunk_size(policy.parameters(),
                    policy.executor(), test_function, count);

            if (chunk_size == 0)
                chunk_size = (count + cores - 1) / cores;

            shape.reserve(count / chunk_size + 1);
            while (count != 0)
            {
                std::size_t chunk = (std::min)(chunk_size, count);

                shape.push_back(hpx::util::make_tuple(first, chunk));

                count -= chunk;
                first = parallel::v1::detail::next(first, chunk);
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

                std::size_t chunk = (std::min)(chunk_size, count);

                shape.push_back(hpx::util::make_tuple(first, chunk));

                count -= chunk;
                first = parallel::v1::detail::next(first, chunk);
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
        FwdIter& first, std::size_t& count)
    {
        typedef typename ExPolicy::executor_parameters_type parameters_type;
        typedef executor_parameter_traits<parameters_type> traits;
        typedef hpx::util::tuple<std::size_t, FwdIter, std::size_t> tuple_type;

        bool variable_chunk_sizes = traits::variable_chunk_size(
            policy.parameters(), policy.executor());

        std::vector<tuple_type> shape;

        std::size_t base_idx = 0;
        if (!variable_chunk_sizes)
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
                    first = parallel::v1::detail::next(first, test_chunk_size);
                    count -= test_chunk_size;

                    return test_chunk_size;
                };

            std::size_t chunk_size =
                traits::get_chunk_size(policy.parameters(),
                    policy.executor(), test_function, count);

            shape.reserve(count / (chunk_size + 1));
            while (count != 0)
            {
                std::size_t chunk = (std::min)(chunk_size, count);

                shape.push_back(hpx::util::make_tuple(base_idx, first, chunk));

                count -= chunk;
                first = parallel::v1::detail::next(first, chunk);
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

                std::size_t chunk = (std::min)(chunk_size, count);

                shape.push_back(hpx::util::make_tuple(base_idx, first, chunk));
                count -= chunk;
                first = parallel::v1::detail::next(first, chunk);
                base_idx += chunk;
            }
        }

        return shape;
    }
}}}}

#endif
