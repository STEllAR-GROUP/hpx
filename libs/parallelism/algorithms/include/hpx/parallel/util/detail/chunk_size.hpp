//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/iterator_support/iterator_range.hpp>

#include <hpx/execution/algorithms/detail/is_negative.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/execution/executors/execution_information.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/parallel/util/detail/chunk_size_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Future, typename FwdIter>
    // requires traits::is_future<Future>
    void add_ready_future(
        std::vector<Future>& workitems, F&& f, FwdIter first, std::size_t count)
    {
        workitems.push_back(hpx::make_ready_future(f(first, count)));
    }

    template <typename F, typename FwdIter>
    void add_ready_future(std::vector<hpx::future<void>>& workitems, F&& f,
        FwdIter first, std::size_t count)
    {
        f(first, count);
        workitems.push_back(hpx::make_ready_future());
    }

    template <typename F, typename FwdIter>
    void add_ready_future(std::vector<hpx::shared_future<void>>& workitems,
        F&& f, FwdIter first, std::size_t count)
    {
        f(first, count);
        workitems.push_back(hpx::make_ready_future());
    }

    ////////////////////////////////////////////////////////////////////////////
    inline void adjust_chunk_size_and_max_chunks(std::size_t cores,
        std::size_t count, std::size_t& max_chunks, std::size_t& chunk_size,
        bool has_variable_chunk_size = false)
    {
        if (max_chunks == 0)
        {
            if (chunk_size == 0)
            {
                // try to calculate chunk-size and maximum number of chunks
                chunk_size = (count + 4 * cores - 1) / (4 * cores);    // -V112

                // we should not consider more chunks than we have elements
                max_chunks = (std::min)(4 * cores, count);    // -V112

                // we should not make chunks smaller than what's determined by
                // the max chunk size
                chunk_size = (std::max)(
                    chunk_size, (count + max_chunks - 1) / max_chunks);
            }
            else
            {
                // max_chunks == 0 && chunk_size != 0
                max_chunks = (count + chunk_size - 1) / chunk_size;
            }
            return;
        }

        if (has_variable_chunk_size)
        {
            HPX_ASSERT(chunk_size != 0);
            return;
        }

        if (chunk_size == 0)
        {
            // max_chunks != 0
            chunk_size = (count + max_chunks - 1) / max_chunks;
        }
        else
        {
            // max_chunks != 0 && chunk_size != 0

            // in this case we make sure that there are no more chunks than
            // max_chunks
            std::size_t calculated_max_chunks =
                (count + chunk_size - 1) / chunk_size;

            if (calculated_max_chunks > max_chunks)
            {
                chunk_size = (count + max_chunks - 1) / max_chunks;
            }
        }
    }

    template <typename ExPolicy, typename Future, typename F1, typename FwdIter,
        typename Stride>
    // requires traits::is_future<Future>
    hpx::util::iterator_range<
        parallel::util::detail::chunk_size_iterator<FwdIter>>
    get_bulk_iteration_shape(std::false_type /*has_variable_chunk_size*/,
        ExPolicy&& policy, std::vector<Future>& workitems, F1&& f1,
        FwdIter& begin, std::size_t& count, Stride s)
    {
        std::size_t const cores = execution::processing_units_count(
            policy.parameters(), policy.executor());

        std::size_t max_chunks = execution::maximal_number_of_chunks(
            policy.parameters(), policy.executor(), cores, count);

        FwdIter last = begin;
        std::advance(last, count);

        Stride stride = parallel::v1::detail::abs(s);
        auto test_function = [&](std::size_t test_chunk_size) -> std::size_t {
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
            begin = parallel::v1::detail::next(begin, count, test_chunk_size);

            count -= test_chunk_size;
            return test_chunk_size;
        };

        std::size_t chunk_size = execution::get_chunk_size(policy.parameters(),
            policy.executor(), test_function, cores, count);

        // make sure, chunk size and max_chunks are consistent
        adjust_chunk_size_and_max_chunks(cores, count, max_chunks, chunk_size);

        if (stride != 1)
        {
            chunk_size = (std::max)(std::size_t(stride),
                ((chunk_size + stride) / stride - 1) * stride);
        }

        using iterator = parallel::util::detail::chunk_size_iterator<FwdIter>;

        iterator shape_begin(begin, chunk_size, count);
        iterator shape_end(last, chunk_size);

        return hpx::util::make_iterator_range(shape_begin, shape_end);
    }

    template <typename ExPolicy, typename Future, typename F1, typename FwdIter,
        typename Stride>
    // requires traits::is_future<Future>
    std::vector<hpx::tuple<FwdIter, std::size_t>> get_bulk_iteration_shape(
        std::true_type /*has_variable_chunk_size*/, ExPolicy&& policy,
        std::vector<Future>& /*workitems*/, F1&& /*f1*/, FwdIter& first,
        std::size_t& count, Stride s)
    {
        using tuple_type = hpx::tuple<FwdIter, std::size_t>;

        std::size_t const cores = execution::processing_units_count(
            policy.parameters(), policy.executor());

        std::size_t max_chunks = execution::maximal_number_of_chunks(
            policy.parameters(), policy.executor(), cores, count);
        HPX_ASSERT(0 != max_chunks);

        std::vector<tuple_type> shape;
        Stride stride = parallel::v1::detail::abs(s);

        // we should not consider more chunks than we have elements
        if (max_chunks != 0)
        {
            max_chunks = (std::min)(max_chunks, count);
        }

        while (count != 0)
        {
            std::size_t chunk_size = execution::get_chunk_size(
                policy.parameters(), policy.executor(),
                [](std::size_t) { return 0; }, cores, count);

            // make sure, chunk size and max_chunks are consistent
            adjust_chunk_size_and_max_chunks(
                cores, count, max_chunks, chunk_size, true);

            if (stride != 1)
            {
                chunk_size = (std::max)(std::size_t(stride),
                    ((chunk_size + stride) / stride - 1) * stride);
            }

            // in last chunk, consider only remaining number of elements
            std::size_t chunk = (std::min)(chunk_size, count);

            shape.push_back(hpx::make_tuple(first, chunk));

            // modifies 'chunk'
            first = parallel::v1::detail::next(first, count, chunk);
            count -= chunk;
        }

        return shape;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename F, typename FwdIter>
    // requires traits::is_future<Future>
    void add_ready_future_idx(std::vector<Future>& workitems, F&& f,
        FwdIter first, std::size_t base_idx, std::size_t count)
    {
        workitems.push_back(hpx::make_ready_future(f(first, count, base_idx)));
    }

    template <typename F, typename FwdIter>
    void add_ready_future_idx(std::vector<hpx::future<void>>& workitems, F&& f,
        FwdIter first, std::size_t base_idx, std::size_t count)
    {
        f(first, count, base_idx);
        workitems.push_back(hpx::make_ready_future());
    }

    template <typename F, typename FwdIter>
    void add_ready_future_idx(std::vector<hpx::shared_future<void>>& workitems,
        F&& f, std::size_t base_idx, FwdIter first, std::size_t count)
    {
        f(first, count, base_idx);
        workitems.push_back(hpx::make_ready_future());
    }

    template <typename ExPolicy, typename Future, typename F1, typename FwdIter,
        typename Stride>
    // requires traits::is_future<Future>
    hpx::util::iterator_range<
        parallel::util::detail::chunk_size_idx_iterator<FwdIter>>
    get_bulk_iteration_shape_idx(std::false_type /*has_variable_chunk_size*/,
        ExPolicy&& policy, std::vector<Future>& workitems, F1&& f1,
        FwdIter begin, std::size_t count, Stride s)
    {
        std::size_t const cores = execution::processing_units_count(
            policy.parameters(), policy.executor());

        std::size_t max_chunks = execution::maximal_number_of_chunks(
            policy.parameters(), policy.executor(), cores, count);

        FwdIter last = parallel::v1::detail::next(begin, count);

        Stride stride = parallel::v1::detail::abs(s);
        std::size_t base_idx = 0;
        auto test_function = [&](std::size_t test_chunk_size) -> std::size_t {
            if (test_chunk_size == 0)
                return 0;

            if (stride != 1)
            {
                test_chunk_size = (std::max)(std::size_t(stride),
                    ((test_chunk_size + stride) / stride - 1) * stride);
            }

            add_ready_future_idx(
                workitems, f1, begin, base_idx, test_chunk_size);

            // modifies 'test_chunk_size'
            begin = parallel::v1::detail::next(begin, count, test_chunk_size);

            base_idx += test_chunk_size;
            count -= test_chunk_size;

            return test_chunk_size;
        };

        std::size_t chunk_size = execution::get_chunk_size(policy.parameters(),
            policy.executor(), test_function, cores, count);

        // make sure, chunk size and max_chunks are consistent
        adjust_chunk_size_and_max_chunks(cores, count, max_chunks, chunk_size);

        if (stride != 1)
        {
            chunk_size = (std::max)(std::size_t(stride),
                ((chunk_size + stride) / stride - 1) * stride);
        }

        using iterator =
            parallel::util::detail::chunk_size_idx_iterator<FwdIter>;

        iterator shape_begin(begin, chunk_size, count, base_idx);
        iterator shape_end(last, chunk_size);

        return hpx::util::make_iterator_range(shape_begin, shape_end);
    }

    template <typename ExPolicy, typename Future, typename F1, typename FwdIter,
        typename Stride>
    // requires traits::is_future<Future>
    std::vector<hpx::tuple<FwdIter, std::size_t, std::size_t>>
    get_bulk_iteration_shape_idx(std::true_type /*has_variable_chunk_size*/,
        ExPolicy&& policy, std::vector<Future>& /* workitems */, F1&& /* f1 */,
        FwdIter first, std::size_t count, Stride s)
    {
        using tuple_type = hpx::tuple<FwdIter, std::size_t, std::size_t>;

        std::size_t const cores = execution::processing_units_count(
            policy.parameters(), policy.executor());

        std::size_t max_chunks = execution::maximal_number_of_chunks(
            policy.parameters(), policy.executor(), cores, count);

        std::vector<tuple_type> shape;
        Stride stride = parallel::v1::detail::abs(s);
        std::size_t base_idx = 0;

        // we should not consider more chunks than we have elements
        if (max_chunks != 0)
        {
            max_chunks = (std::min)(max_chunks, count);
        }

        while (count != 0)
        {
            std::size_t chunk_size = execution::get_chunk_size(
                policy.parameters(), policy.executor(),
                [](std::size_t) { return 0; }, cores, count);

            // make sure, chunk size and max_chunks are consistent
            adjust_chunk_size_and_max_chunks(
                cores, count, max_chunks, chunk_size, true);

            if (stride != 1)
            {
                chunk_size = (std::max)(std::size_t(stride),
                    ((chunk_size + stride) / stride - 1) * stride);
            }

            // in last chunk, consider only remaining number of elements
            std::size_t chunk = (std::min)(chunk_size, count);

            shape.push_back(hpx::make_tuple(first, chunk, base_idx));

            // modifies 'chunk'
            first = parallel::v1::detail::next(first, count, chunk);

            count -= chunk;
            base_idx += chunk;
        }

        return shape;
    }
}}}}    // namespace hpx::parallel::util::detail
