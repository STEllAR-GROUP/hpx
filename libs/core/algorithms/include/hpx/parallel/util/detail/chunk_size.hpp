//  Copyright (c) 2007-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_base/scheduling_properties.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/algorithms/detail/is_negative.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/execution/executors/execution_information.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/iterator_support/iterator_range.hpp>
#include <hpx/parallel/util/detail/chunk_size_iterator.hpp>
#include <hpx/properties/property.hpp>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::parallel::util::detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Future, typename FwdIter>
    // requires traits::is_future<Future>
    void add_ready_future(
        std::vector<Future>& workitems, F&& f, FwdIter first, std::size_t count)
    {
        workitems.push_back(
            hpx::make_ready_future(HPX_FORWARD(F, f)(first, count)));
    }

    template <typename F, typename FwdIter>
    void add_ready_future(std::vector<hpx::future<void>>& workitems, F&& f,
        FwdIter first, std::size_t count)
    {
        HPX_FORWARD(F, f)(first, count);
        workitems.push_back(hpx::make_ready_future());
    }

    template <typename F, typename FwdIter>
    void add_ready_future(std::vector<hpx::shared_future<void>>& workitems,
        F&& f, FwdIter first, std::size_t count)
    {
        HPX_FORWARD(F, f)(first, count);
        workitems.push_back(hpx::make_ready_future());
    }

    ////////////////////////////////////////////////////////////////////////////
    constexpr void adjust_chunk_size_and_max_chunks(std::size_t cores,
        std::size_t count, std::size_t& max_chunks, std::size_t& chunk_size,
        bool has_variable_chunk_size = false) noexcept
    {
        if (max_chunks == 0)
        {
            if (chunk_size == 0)
            {
                std::size_t const cores_times_4 = 4 * cores;    // -V112

                // try to calculate chunk-size and maximum number of chunks
                chunk_size = (count + cores_times_4 - 1) / cores_times_4;

                // different versions of clang-format do different things
                // clang-format off

                // we should not consider more chunks than we have elements
                max_chunks = (std::min) (cores_times_4, count);    // -V112

                // we should not make chunks smaller than what's determined by
                // the max chunk size
                chunk_size = (std::max) (chunk_size,
                    (count + max_chunks - 1) / max_chunks);
                // clang-format on
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
            std::size_t const calculated_max_chunks =
                (count + chunk_size - 1) / chunk_size;

            if (calculated_max_chunks > max_chunks)
            {
                chunk_size = (count + max_chunks - 1) / max_chunks;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    template <typename IterOrR>
    constexpr auto next_or_subrange(IterOrR const& target, std::size_t first,
        [[maybe_unused]] std::size_t size)
    {
        if constexpr (hpx::traits::is_iterator_v<IterOrR> ||
            std::is_integral_v<IterOrR>)
        {
            return parallel::detail::next(target, first);
        }
        else
        {
            return hpx::util::subrange(target, first, size);
        }
    }

    template <typename ExPolicy, typename IterOrR,
        typename Stride = std::size_t>
    hpx::util::iterator_range<chunk_size_iterator<IterOrR>>
    get_bulk_iteration_shape(ExPolicy& policy, IterOrR& it_or_r,
        std::size_t& count, Stride s = Stride(1))
    {
        if (count == 0)
        {
            auto it = chunk_size_iterator(it_or_r, 1);
            return hpx::util::iterator_range(it, it);
        }

        std::size_t const cores =
            hpx::execution::experimental::processing_units_count(
                policy.parameters(), policy.executor(),
                hpx::chrono::null_duration, count);

        std::size_t max_chunks =
            hpx::execution::experimental::maximal_number_of_chunks(
                policy.parameters(), policy.executor(), cores, count);

        std::size_t chunk_size =
            hpx::execution::experimental::get_chunk_size(policy.parameters(),
                policy.executor(), hpx::chrono::null_duration, cores, count);

        // make sure, chunk size and max_chunks are consistent
        adjust_chunk_size_and_max_chunks(cores, count, max_chunks, chunk_size);

        auto last = next_or_subrange(it_or_r, count, 0);
        Stride stride = parallel::detail::abs(s);

        if (stride != 1)
        {
            // different versions of clang-format do different things
            // clang-format off
            chunk_size = (std::max) (static_cast<std::size_t>(stride),
                (chunk_size + stride - 1) / stride * stride);
            // clang-format on
        }

        // update executor with new values
        policy = hpx::experimental::prefer(
            hpx::execution::experimental::with_processing_units_count, policy,
            cores);

        auto shape_begin = chunk_size_iterator(it_or_r, chunk_size, count);
        auto shape_end = chunk_size_iterator(last, chunk_size, count, count);

        return hpx::util::iterator_range(shape_begin, shape_end);
    }

    template <typename ExPolicy, typename Future, typename F1, typename IterOrR,
        typename Stride = std::size_t>
    hpx::util::iterator_range<chunk_size_iterator<IterOrR>>
    get_bulk_iteration_shape(ExPolicy& policy, std::vector<Future>& workitems,
        F1&& f1, IterOrR& it_or_r, std::size_t& count, Stride s = Stride(1))
    {
        if (count == 0)
        {
            auto it = chunk_size_iterator(it_or_r, 1);
            return hpx::util::iterator_range(it, it);
        }

        Stride stride = parallel::detail::abs(s);

        auto test_function = [&](std::size_t test_chunk_size) -> std::size_t {
            if (test_chunk_size == 0)
                return 0;

            if (stride != 1)
            {
                // different versions of clang-format do different things
                // clang-format off

                // rounding up
                test_chunk_size = (std::max) (static_cast<std::size_t>(stride),
                    (test_chunk_size + stride - 1) / stride * stride);
                // clang-format on
            }

            add_ready_future(workitems, f1, it_or_r, test_chunk_size);

            // different versions of clang-format do different things
            // clang-format off
            test_chunk_size = (std::min) (count, test_chunk_size);
            // clang-format on

            count -= test_chunk_size;
            it_or_r = next_or_subrange(it_or_r, test_chunk_size, count);

            return test_chunk_size;
        };

        // note: running the test function will modify 'count'
        auto iteration_duration =
            hpx::execution::experimental::measure_iteration(
                policy.parameters(), policy.executor(), test_function, count);

        std::size_t const cores =
            hpx::execution::experimental::processing_units_count(
                policy.parameters(), policy.executor(), iteration_duration,
                count);

        std::size_t max_chunks =
            hpx::execution::experimental::maximal_number_of_chunks(
                policy.parameters(), policy.executor(), cores, count);

        std::size_t chunk_size =
            hpx::execution::experimental::get_chunk_size(policy.parameters(),
                policy.executor(), iteration_duration, cores, count);

        // make sure, chunk size and max_chunks are consistent
        adjust_chunk_size_and_max_chunks(cores, count, max_chunks, chunk_size);

        auto last = next_or_subrange(it_or_r, count, 0);

        if (stride != 1)
        {
            // different versions of clang-format do different things
            // clang-format off
            chunk_size = (std::max) (static_cast<std::size_t>(stride),
                (chunk_size + stride - 1) / stride * stride);
            // clang-format on
        }

        // update executor with new values
        policy = hpx::experimental::prefer(
            hpx::execution::experimental::with_processing_units_count, policy,
            cores);

        auto shape_begin = chunk_size_iterator(it_or_r, chunk_size, count);
        auto shape_end = chunk_size_iterator(last, chunk_size, count, count);

        return hpx::util::iterator_range(shape_begin, shape_end);
    }

    template <typename ExPolicy, typename IterOrR,
        typename Stride = std::size_t>
    std::vector<hpx::tuple<IterOrR, std::size_t>>
    get_bulk_iteration_shape_variable(ExPolicy& policy, IterOrR& it_or_r,
        std::size_t& count, Stride s = Stride(1))
    {
        using tuple_type = hpx::tuple<IterOrR, std::size_t>;
        std::vector<tuple_type> shape;

        if (count == 0)
        {
            return shape;
        }

        std::size_t const cores =
            hpx::execution::experimental::processing_units_count(
                policy.parameters(), policy.executor(),
                hpx::chrono::null_duration, count);

        std::size_t max_chunks =
            hpx::execution::experimental::maximal_number_of_chunks(
                policy.parameters(), policy.executor(), cores, count);
        HPX_ASSERT(0 != max_chunks);

        Stride stride = parallel::detail::abs(s);

        // different versions of clang-format do different things
        // clang-format off

        // we should not consider more chunks than we have elements
        if (max_chunks != 0)
        {
            max_chunks = (std::min) (max_chunks, count);
        }

        while (count != 0)
        {
            std::size_t chunk_size =
                hpx::execution::experimental::get_chunk_size(
                    policy.parameters(), policy.executor(),
                    hpx::chrono::null_duration, cores, count);

            // make sure, chunk size and max_chunks are consistent
            adjust_chunk_size_and_max_chunks(
                cores, count, max_chunks, chunk_size, true);

            if (stride != 1)
            {
                chunk_size = (std::max) (static_cast<std::size_t>(stride),
                    (chunk_size + stride - 1) / stride * stride);
            }

            // in last chunk, consider only remaining number of elements
            std::size_t chunk = (std::min) (chunk_size, count);

            shape.emplace_back(it_or_r, chunk);

            chunk = (std::min) (count, chunk);
            count -= chunk;

            it_or_r = next_or_subrange(it_or_r, count, chunk);
        }
        // clang-format on

        // update executor with new values
        policy = hpx::experimental::prefer(
            hpx::execution::experimental::with_processing_units_count, policy,
            cores);

        return shape;
    }

    template <typename ExPolicy, typename Future, typename F1, typename FwdIter,
        typename Stride = std::size_t>
    decltype(auto) get_bulk_iteration_shape(std::false_type, ExPolicy& policy,
        std::vector<Future>& workitems, F1&& f1, FwdIter& begin,
        std::size_t& count, Stride s = Stride(1))
    {
        return get_bulk_iteration_shape(
            policy, workitems, HPX_FORWARD(F1, f1), begin, count, s);
    }

    template <typename ExPolicy, typename Future, typename F1, typename FwdIter,
        typename Stride = std::size_t>
    decltype(auto) get_bulk_iteration_shape(std::true_type, ExPolicy& policy,
        std::vector<Future>& workitems, F1&& f1, FwdIter& begin,
        std::size_t& count, Stride s = Stride(1))
    {
        return get_bulk_iteration_shape_variable(
            policy, workitems, HPX_FORWARD(F1, f1), begin, count, s);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename F, typename FwdIter>
    void add_ready_future_idx(std::vector<Future>& workitems, F&& f,
        FwdIter first, std::size_t base_idx, std::size_t count)
    {
        workitems.push_back(
            hpx::make_ready_future(HPX_FORWARD(F, f)(first, count, base_idx)));
    }

    template <typename F, typename FwdIter>
    void add_ready_future_idx(std::vector<hpx::future<void>>& workitems, F&& f,
        FwdIter first, std::size_t base_idx, std::size_t count)
    {
        HPX_FORWARD(F, f)(first, count, base_idx);
        workitems.push_back(hpx::make_ready_future());
    }

    template <typename F, typename FwdIter>
    void add_ready_future_idx(std::vector<hpx::shared_future<void>>& workitems,
        F&& f, std::size_t base_idx, FwdIter first, std::size_t count)
    {
        HPX_FORWARD(F, f)(first, count, base_idx);
        workitems.push_back(hpx::make_ready_future());
    }

    template <typename ExPolicy, typename FwdIter,
        typename Stride = std::size_t>
    hpx::util::iterator_range<
        parallel::util::detail::chunk_size_idx_iterator<FwdIter>>
    get_bulk_iteration_shape_idx(ExPolicy& policy, FwdIter begin,
        std::size_t count, Stride s = Stride(1))
    {
        using iterator =
            parallel::util::detail::chunk_size_idx_iterator<FwdIter>;

        if (count == 0)
        {
            auto it = iterator(begin, 1);
            return hpx::util::iterator_range(it, it);
        }

        std::size_t const cores =
            hpx::execution::experimental::processing_units_count(
                policy.parameters(), policy.executor(),
                hpx::chrono::null_duration, count);

        std::size_t max_chunks =
            hpx::execution::experimental::maximal_number_of_chunks(
                policy.parameters(), policy.executor(), cores, count);

        FwdIter last = parallel::detail::next(begin, count);
        Stride stride = parallel::detail::abs(s);

        std::size_t chunk_size =
            hpx::execution::experimental::get_chunk_size(policy.parameters(),
                policy.executor(), hpx::chrono::null_duration, cores, count);

        // make sure, chunk size and max_chunks are consistent
        adjust_chunk_size_and_max_chunks(cores, count, max_chunks, chunk_size);

        if (stride != 1)
        {
            // different versions of clang-format do different things
            // clang-format off
            chunk_size = (std::max) (static_cast<std::size_t>(stride),
                static_cast<std::size_t>((chunk_size + stride - 1) / stride * stride));
            // clang-format on
        }

        // update executor with new values
        policy = hpx::experimental::prefer(
            hpx::execution::experimental::with_processing_units_count, policy,
            cores);

        using iterator =
            parallel::util::detail::chunk_size_idx_iterator<FwdIter>;

        iterator shape_begin(begin, chunk_size, count, 0, 0);
        iterator shape_end(last, chunk_size, count, count, 0);

        return hpx::util::iterator_range(shape_begin, shape_end);
    }

    template <typename ExPolicy, typename Future, typename F1, typename FwdIter,
        typename Stride = std::size_t>
    hpx::util::iterator_range<
        parallel::util::detail::chunk_size_idx_iterator<FwdIter>>
    get_bulk_iteration_shape_idx(ExPolicy& policy,
        std::vector<Future>& workitems, F1&& f1, FwdIter begin,
        std::size_t count, Stride s = Stride(1))
    {
        using iterator =
            parallel::util::detail::chunk_size_idx_iterator<FwdIter>;

        if (count == 0)
        {
            auto it = iterator(begin, 1);
            return hpx::util::iterator_range(it, it);
        }

        Stride stride = parallel::detail::abs(s);

        std::size_t base_idx = 0;
        auto test_function = [&](std::size_t test_chunk_size) -> std::size_t {
            if (test_chunk_size == 0)
                return 0;

            if (stride != 1)
            {
                // different versions of clang-format do different things
                // clang-format off
                test_chunk_size = (std::max)(static_cast<std::size_t>(stride),
                    (test_chunk_size + stride - 1) / stride * stride);
                // clang-format on
            }

            add_ready_future_idx(
                workitems, f1, begin, base_idx, test_chunk_size);

            // modifies 'test_chunk_size'
            begin = parallel::detail::next(begin, count, test_chunk_size);

            base_idx += test_chunk_size;
            count -= test_chunk_size;

            return test_chunk_size;
        };

        // note: running the test function will modify 'count'
        auto iteration_duration =
            hpx::execution::experimental::measure_iteration(
                policy.parameters(), policy.executor(), test_function, count);

        std::size_t const cores =
            hpx::execution::experimental::processing_units_count(
                policy.parameters(), policy.executor(), iteration_duration,
                count);

        std::size_t max_chunks =
            hpx::execution::experimental::maximal_number_of_chunks(
                policy.parameters(), policy.executor(), cores, count);

        FwdIter last = parallel::detail::next(begin, count);

        std::size_t chunk_size =
            hpx::execution::experimental::get_chunk_size(policy.parameters(),
                policy.executor(), iteration_duration, cores, count);

        // make sure, chunk size and max_chunks are consistent
        adjust_chunk_size_and_max_chunks(cores, count, max_chunks, chunk_size);

        if (stride != 1)
        {
            // different versions of clang-format do different things
            // clang-format off
            chunk_size = (std::max) (static_cast<std::size_t>(stride),
                (chunk_size + stride - 1) / stride * stride);
            // clang-format on
        }

        // update executor with new values
        policy = hpx::experimental::prefer(
            hpx::execution::experimental::with_processing_units_count, policy,
            cores);

        using iterator =
            parallel::util::detail::chunk_size_idx_iterator<FwdIter>;

        iterator shape_begin(begin, chunk_size, count, 0, base_idx);
        iterator shape_end(last, chunk_size, count, count, base_idx);

        return hpx::util::iterator_range(shape_begin, shape_end);
    }

    template <typename ExPolicy, typename FwdIter,
        typename Stride = std::size_t>
    std::vector<hpx::tuple<FwdIter, std::size_t, std::size_t>>
    get_bulk_iteration_shape_idx_variable(ExPolicy& policy, FwdIter first,
        std::size_t count, Stride s = Stride(1))
    {
        using tuple_type = hpx::tuple<FwdIter, std::size_t, std::size_t>;
        std::vector<tuple_type> shape;

        if (count == 0)
        {
            return shape;
        }

        std::size_t const cores =
            hpx::execution::experimental::processing_units_count(
                policy.parameters(), policy.executor(),
                hpx::chrono::null_duration, count);

        std::size_t max_chunks =
            hpx::execution::experimental::maximal_number_of_chunks(
                policy.parameters(), policy.executor(), cores, count);

        Stride stride = parallel::detail::abs(s);

        // different versions of clang-format do different things
        // clang-format off

        // we should not consider more chunks than we have elements
        if (max_chunks != 0)
        {
            max_chunks = (std::min) (max_chunks, count);
        }

        std::size_t base_idx = 0;
        while (count != 0)
        {
            std::size_t chunk_size =
                hpx::execution::experimental::get_chunk_size(
                    policy.parameters(), policy.executor(),
                    hpx::chrono::null_duration,  cores, count);

            // make sure, chunk size and max_chunks are consistent
            adjust_chunk_size_and_max_chunks(
                cores, count, max_chunks, chunk_size, true);

            if (stride != 1)
            {
                chunk_size = (std::max) (static_cast<std::size_t>(stride),
                    (chunk_size + stride - 1) / stride * stride);
            }

            // in last chunk, consider only remaining number of elements
            std::size_t chunk = (std::min) (chunk_size, count);

            shape.emplace_back(first, chunk, base_idx);

            // modifies 'chunk'
            first = parallel::detail::next(first, count, chunk);

            count -= chunk;
            base_idx += chunk;
        }
        // clang-format on

        // update executor with new values
        policy = hpx::experimental::prefer(
            hpx::execution::experimental::with_processing_units_count, policy,
            cores);

        return shape;
    }
}    // namespace hpx::parallel::util::detail
