//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_DETAIL_AUTO_CHUNK_SIZE_OCT_03_2014_0159PM)
#define HPX_PARALLEL_UTIL_DETAIL_AUTO_CHUNK_SIZE_OCT_03_2014_0159PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/tuple.hpp>

#include <hpx/parallel/algorithms/detail/is_negative.hpp>
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/executors/executor_parameter_traits.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/util/detail/chunk_size_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <vector>

#include <boost/range/iterator_range.hpp>

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
    boost::iterator_range<parallel::util::detail::chunk_size_iterator<FwdIter> >
    get_bulk_iteration_shape(
        ExPolicy && policy, std::vector<Future>& workitems, F1 && f1,
        FwdIter& begin, std::size_t& count, Stride s, std::false_type)
    {
        typedef typename hpx::util::decay<ExPolicy>::type::executor_parameters_type
            parameters_type;
        typedef executor_parameter_traits<parameters_type> traits;

        typedef typename hpx::util::decay<ExPolicy>::type::executor_type
            executor_type;
        std::size_t const cores = executor_information_traits<executor_type>::
            processing_units_count(policy.executor(), policy.parameters());

        std::size_t max_chunks = traits::maximal_number_of_chunks(
            policy.parameters(), policy.executor(), cores, count);
        HPX_ASSERT(0 != max_chunks);

        FwdIter last = begin;
        std::advance(last, count);

        Stride stride = parallel::v1::detail::abs(s);
        auto test_function =
            [&]() -> std::size_t
            {
                std::size_t test_chunk_size = count / 100;
                if (test_chunk_size == 0)
                    return 0;

                if (stride != 1)
                {
                    // rounding up
                    test_chunk_size = (std::max)(std::size_t(stride),
                        ((test_chunk_size + stride - 1) / stride) * stride);
                }

                add_ready_future(workitems, f1, begin, test_chunk_size);

                // modifies 'test_chunk_size'
                begin = parallel::v1::detail::next(
                    begin, count, test_chunk_size);

                count -= test_chunk_size;
                return test_chunk_size;
            };

        std::size_t chunk_size =
            traits::get_chunk_size(policy.parameters(),
                policy.executor(), test_function, cores, count);

        // we should not consider more chunks than we have elements
        max_chunks = (std::min)(max_chunks, count);

        // we should not make chunks smaller than what's determined by the
        // max chunk size
        chunk_size = (std::max)(chunk_size,
            (count + max_chunks - 1) / max_chunks);

        if (stride != 1)
        {
            chunk_size = (std::max)(std::size_t(stride),
                ((chunk_size + stride) / stride - 1) * stride);
        }

        typedef parallel::util::detail::chunk_size_iterator<FwdIter> iterator;

        iterator shape_begin(begin, chunk_size, count);
        iterator shape_end(last, chunk_size);

        return boost::make_iterator_range(shape_begin, shape_end);
    }

    template <typename ExPolicy, typename Future, typename F1,
        typename FwdIter, typename Stride>
        // requires traits::is_future<Future>
    std::vector<hpx::util::tuple<FwdIter, std::size_t> >
    get_bulk_iteration_shape(
        ExPolicy && policy, std::vector<Future>& workitems, F1 && f1,
        FwdIter& first, std::size_t& count, Stride s, std::true_type)
    {
        typedef typename hpx::util::decay<ExPolicy>::type::executor_parameters_type
            parameters_type;
        typedef executor_parameter_traits<parameters_type> traits;
        typedef hpx::util::tuple<FwdIter, std::size_t> tuple_type;

        typedef typename hpx::util::decay<ExPolicy>::type::executor_type
            executor_type;
        std::size_t const cores = executor_information_traits<executor_type>::
            processing_units_count(policy.executor(), policy.parameters());

        std::size_t max_chunks = traits::maximal_number_of_chunks(
            policy.parameters(), policy.executor(), cores, count);
        HPX_ASSERT(0 != max_chunks);

        std::vector<tuple_type> shape;
        Stride stride = parallel::v1::detail::abs(s);

        // we should not consider more chunks than we have elements
        max_chunks = (std::min)(max_chunks, count);

        while (count != 0)
        {
            std::size_t chunk_size =
                traits::get_chunk_size(policy.parameters(),
                    policy.executor(), [](){ return 0; }, cores, count);

            // we should not make chunks smaller than what's determined by the
            // max chunk size
            chunk_size = (std::max)(chunk_size,
                (count + max_chunks - 1) / max_chunks);

            if (stride != 1)
            {
                chunk_size = (std::max)(std::size_t(stride),
                    ((chunk_size + stride) / stride - 1) * stride);
            }

            std::size_t chunk = (std::min)(chunk_size, count);

            shape.push_back(hpx::util::make_tuple(first, chunk));

            // modifies 'chunk'
            first = parallel::v1::detail::next(first, count, chunk);
            count -= chunk;
        }

        return shape;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename F, typename FwdIter>
        // requires traits::is_future<Future>
    void add_ready_future_idx(std::vector<Future>& workitems,
        F && f, FwdIter first, std::size_t base_idx, std::size_t count)
    {
        workitems.push_back(
            hpx::make_ready_future(f(first, count, base_idx)));
    }

    template <typename F, typename FwdIter>
    void add_ready_future_idx(std::vector<hpx::future<void> >& workitems,
        F && f, FwdIter first, std::size_t base_idx, std::size_t count)
    {
        f(first, count, base_idx);
        workitems.push_back(hpx::make_ready_future());
    }

    template <typename F, typename FwdIter>
    void add_ready_future_idx(std::vector<hpx::shared_future<void> >& workitems,
        F && f, std::size_t base_idx, FwdIter first, std::size_t count)
    {
        f(first, count, base_idx);
        workitems.push_back(hpx::make_ready_future());
    }

    template <typename ExPolicy, typename Future, typename F1,
        typename FwdIter, typename Stride>
        // requires traits::is_future<Future>
    boost::iterator_range<
        parallel::util::detail::chunk_size_idx_iterator<FwdIter>
    >
    get_bulk_iteration_shape_idx(
        ExPolicy && policy, std::vector<Future>& workitems, F1 && f1,
        FwdIter begin, std::size_t count, Stride s, std::false_type)
    {
        typedef typename hpx::util::decay<ExPolicy>::type::executor_parameters_type
            parameters_type;
        typedef executor_parameter_traits<parameters_type> traits;

        typedef typename hpx::util::decay<ExPolicy>::type::executor_type
            executor_type;
        std::size_t const cores = executor_information_traits<executor_type>::
            processing_units_count(policy.executor(), policy.parameters());

        std::size_t max_chunks = traits::maximal_number_of_chunks(
            policy.parameters(), policy.executor(), cores, count);
        HPX_ASSERT(0 != max_chunks);

        FwdIter last = parallel::v1::detail::next(begin, count);

        Stride stride = parallel::v1::detail::abs(s);
        std::size_t base_idx = 0;
        auto test_function =
            [&]() -> std::size_t
            {
                std::size_t test_chunk_size = count / 100;
                if (test_chunk_size == 0)
                    return 0;

                if (stride != 1)
                {
                    test_chunk_size = (std::max)(std::size_t(stride),
                        ((test_chunk_size + stride) / stride - 1) * stride);
                }

                add_ready_future_idx(workitems, f1, begin, base_idx,
                    test_chunk_size);

                // modifies 'test_chunk_size'
                begin = parallel::v1::detail::next(
                    begin, count, test_chunk_size);

                base_idx += test_chunk_size;
                count -= test_chunk_size;

                return test_chunk_size;
            };

        std::size_t chunk_size =
            traits::get_chunk_size(policy.parameters(),
                policy.executor(), test_function, cores, count);

        // we should not consider more chunks than we have elements
        max_chunks = (std::min)(max_chunks, count);

        // we should not make chunks smaller than what's determined by the
        // max chunk size
        chunk_size = (std::max)(chunk_size,
            (count + max_chunks - 1) / max_chunks);

        if (stride != 1)
        {
            chunk_size = (std::max)(std::size_t(stride),
                ((chunk_size + stride) / stride - 1) * stride);
        }

        typedef parallel::util::detail::chunk_size_idx_iterator<FwdIter> iterator;

        iterator shape_begin(begin, chunk_size, count, base_idx);
        iterator shape_end(last, chunk_size);

        return boost::make_iterator_range(shape_begin, shape_end);
    }

    template <typename ExPolicy, typename Future, typename F1,
        typename FwdIter, typename Stride>
        // requires traits::is_future<Future>
    std::vector<hpx::util::tuple<FwdIter, std::size_t, std::size_t> >
    get_bulk_iteration_shape_idx(
        ExPolicy && policy, std::vector<Future>& workitems, F1 && f1,
        FwdIter first, std::size_t count, Stride s, std::true_type)
    {
        typedef typename hpx::util::decay<ExPolicy>::type::executor_parameters_type
            parameters_type;
        typedef executor_parameter_traits<parameters_type> traits;
        typedef hpx::util::tuple<FwdIter, std::size_t, std::size_t> tuple_type;

        typedef typename hpx::util::decay<ExPolicy>::type::executor_type
            executor_type;
        std::size_t const cores = executor_information_traits<executor_type>::
            processing_units_count(policy.executor(), policy.parameters());

        std::size_t max_chunks = traits::maximal_number_of_chunks(
            policy.parameters(), policy.executor(), cores, count);
        HPX_ASSERT(0 != max_chunks);

        std::vector<tuple_type> shape;
        Stride stride = parallel::v1::detail::abs(s);
        std::size_t base_idx = 0;

        // we should not consider more chunks than we have elements
        max_chunks = (std::min)(max_chunks, count);

        while (count != 0)
        {
            std::size_t chunk_size =
                traits::get_chunk_size(policy.parameters(),
                    policy.executor(), [](){ return 0; }, cores, count);

            // we should not make chunks smaller than what's determined by the
            // max chunk size
            chunk_size = (std::max)(chunk_size,
                (count + max_chunks - 1) / max_chunks);

            if (stride != 1)
            {
                chunk_size = (std::max)(std::size_t(stride),
                    ((chunk_size + stride) / stride - 1) * stride);
            }

            std::size_t chunk = (std::min)(chunk_size, count);

            shape.push_back(hpx::util::make_tuple(first, chunk, base_idx));

            // modifies 'chunk'
            first = parallel::v1::detail::next(first, count, chunk);

            count -= chunk;
            base_idx += chunk;
        }

        return shape;
    }
}}}}

#endif
