//  Copyright (c) 2020 Mikael Simberg
//  Copyright (c) 2022-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/assert.hpp>
#include <hpx/concurrency/cache_line_data.hpp>
#include <hpx/concurrency/detail/contiguous_index_queue.hpp>
#include <hpx/datastructures/optional.hpp>
#include <hpx/datastructures/tuple.hpp>

#include <atomic>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace hpx::concurrency::detail {

    /// \brief A concurrent queue which can only hold non-contiguous ranges of
    ///        integers.
    ///
    /// A concurrent queue which can be initialized with a range of integers.
    /// Items can be popped from both ends of the queue. Popping from the right
    /// decrements the next element that will be popped from the right, if there
    /// are items left. Popping from the left increments the next element that
    /// will be popped from the left, if there are items left.
    ///
    /// The range of integers is non-contiguous in the sense that the returned
    /// integers are apart by a given step.
    template <typename T = std::uint32_t>
    class non_contiguous_index_queue
    {
        static_assert(sizeof(T) <= 4,    //-V112
            "non_contiguous_index_queue assumes at most 32 bit indices to fit "
            "two indices in an at most 64 bit struct");
        static_assert(std::is_integral_v<T>,
            "non_contiguous_index_queue only works with integral indices");

        struct range
        {
            T first = 0;
            T last = 0;

            range() = default;

            constexpr range(T first, T last) noexcept
              : first(first)
              , last(last)
            {
            }

            constexpr range increment_first(T step) const noexcept
            {
                return range{first + step, last};
            }

            constexpr range decrement_last(T step) const noexcept
            {
                return range{first, last - step};
            }

            constexpr bool empty() const noexcept
            {
                return first >= last;
            }
        };

    public:
        /// \brief Reset the queue with the given range.
        ///
        /// Reset the queue with the given range. No additional synchronization
        /// is done to ensure that other threads are not accessing elements from
        /// the queue. It is the caller's responsibility to ensure that it is
        /// safe to reset the queue.
        ///
        /// \param first Beginning of the new range.
        /// \param last  End of the new range.
        /// \param newstep  Steps size to use when advancing through the range.
        constexpr void reset(T first, T last, T newstep = 1) noexcept
        {
            initial_range = range{first, last};
            step = newstep;
            current_range.data_ = range{first, last};

            // range shouldn't be empty
            HPX_ASSERT(first <= last);

            // range must be cleanly divisible by step
            HPX_ASSERT((last - first) % step == 0);
        }

        /// \brief Construct a new non_contiguous_index_queue.
        ///
        /// Construct a new queue with an empty range.
        constexpr non_contiguous_index_queue() noexcept
          : initial_range{}
          , step(1)
          , current_range{}
        {
        }

        /// \brief Construct a new non_contiguous_index_queue with the given
        ///        range.
        ///
        /// Construct a new queue with the given range as the initial range.
        constexpr non_contiguous_index_queue(T first, T last, T step_) noexcept
          : initial_range{}
          , step{}
          , current_range{}
        {
            reset(first, last, step_);
        }

        /// \brief Copy-construct a queue.
        ///
        /// No additional synchronization is done to ensure that other threads
        /// are not accessing elements from the queue being copied. It is the
        /// caller's responsibility to ensure that it is safe to copy the queue.
        non_contiguous_index_queue(
            non_contiguous_index_queue<T> const& other) noexcept
          : initial_range{other.initial_range}
          , step{other.step}
          , current_range{}
        {
            current_range.data_ =
                other.current_range.data_.load(std::memory_order_relaxed);
        }

        /// \brief Copy-assign a queue.
        ///
        /// No additional synchronization is done to ensure that other threads
        /// are not accessing elements from the queue being copied. It is the
        /// caller's responsibility to ensure that it is safe to copy the queue.
        non_contiguous_index_queue& operator=(
            non_contiguous_index_queue const& other) noexcept
        {
            initial_range = other.initial_range;
            step = other.step;
            current_range.data_ =
                other.current_range.data_.load(std::memory_order_relaxed);
            return *this;
        }

        /// \brief Attempt to pop an item from the left of the queue.
        ///
        /// Attempt to pop an item from the left (beginning) of the queue. If no
        /// items are left hpx::nullopt is returned.
        hpx::optional<T> pop_left() noexcept
        {
            range desired_range{0, 0};
            T index = 0;

            range expected_range =
                current_range.data_.load(std::memory_order_relaxed);

            do
            {
                if (expected_range.empty())
                {
                    return hpx::optional<T>(hpx::nullopt);
                }

                // reduce pipeline pressure
                HPX_SMT_PAUSE;

                index = expected_range.first;
                desired_range = expected_range.increment_first(step);

            } while (!current_range.data_.compare_exchange_weak(
                expected_range, desired_range));

            return hpx::optional<T>(HPX_MOVE(index));
        }

        /// \brief Attempt to pop an item from the right of the queue.
        ///
        /// Attempt to pop an item from the right (end) of the queue. If no
        /// items are left hpx::nullopt is returned.
        hpx::optional<T> pop_right() noexcept
        {
            range desired_range{0, 0};
            T index = 0;

            range expected_range =
                current_range.data_.load(std::memory_order_relaxed);

            do
            {
                if (expected_range.empty())
                {
                    return hpx::optional<T>(hpx::nullopt);
                }

                // reduce pipeline pressure
                HPX_SMT_PAUSE;

                desired_range = expected_range.decrement_last(step);
                index = desired_range.last;

            } while (!current_range.data_.compare_exchange_weak(
                expected_range, desired_range));

            return hpx::optional<T>(HPX_MOVE(index));
        }

        /// \brief Attempt to pop an item from the given end of the queue.
        ///
        /// Attempt to pop an item from the given end of the queue. If no items
        /// are left hpx::nullopt is returned.
        template <queue_end Which>
        hpx::optional<T> pop() noexcept
        {
            if constexpr (Which == queue_end::left)
            {
                return pop_left();
            }
            else
            {
                return pop_right();
            }
        }

        constexpr bool empty() const noexcept
        {
            return current_range.data_.load(std::memory_order_relaxed).empty();
        }

        hpx::tuple<T, T, T> get_current_range() const noexcept
        {
            auto r = current_range.data_.load(std::memory_order_relaxed);
            return {r.first, r.last, step};
        }

    private:
        range initial_range;
        T step = 1;
        hpx::util::cache_line_data<std::atomic<range>> current_range;
    };
}    // namespace hpx::concurrency::detail
