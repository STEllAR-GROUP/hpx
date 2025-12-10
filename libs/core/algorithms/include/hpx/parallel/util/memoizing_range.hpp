//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/concurrency.hpp>
#include <hpx/modules/iterator_support.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/type_support.hpp>

#include <atomic>
#include <cstddef>
#include <iterator>
#include <mutex>
#include <type_traits>
#include <vector>

namespace hpx::parallel::util {

    HPX_CXX_EXPORT template <typename BaseRange>
    struct memoizing_range;

    namespace detail {

        HPX_CXX_EXPORT template <typename BaseRange>
        struct memoizing_range_data;

        HPX_CXX_EXPORT template <typename BaseRange>
        struct memoizing_iterator;

        HPX_CXX_EXPORT template <typename BaseRange>
        struct memoizing_iterator_base
        {
            using base_iterator = hpx::traits::range_iterator_t<BaseRange>;
            using value_type = hpx::traits::iter_value_t<base_iterator>;
            using type =
                hpx::util::iterator_facade<memoizing_iterator<BaseRange>,
                    value_type, std::random_access_iterator_tag>;
        };

        HPX_CXX_EXPORT template <typename BaseRange>
        struct memoizing_iterator : memoizing_iterator_base<BaseRange>::type
        {
            explicit memoizing_iterator(memoizing_range_data<BaseRange>& base,
                std::ptrdiff_t const index, bool const end = false)
              : range(&base)
              , current(index)
              , at_end(end)
            {
                // fill in first element, if needed (not at end)
                if (current < static_cast<std::ptrdiff_t>(range->size()))
                {
                    fill_next(1);
                }
            }

        private:
            friend class hpx::util::iterator_core_access;

            using base_type = memoizing_iterator_base<BaseRange>::type;

            [[nodiscard]] constexpr bool equal(
                memoizing_iterator const& rhs) const noexcept
            {
                return current == rhs.current || (at_end && rhs.at_end);
            }

            void fill_next(std::ptrdiff_t step)
            {
                if (!at_end)
                {
                    if (current + step >= range->filled())
                    {
                        std::unique_lock<memoizing_range_data<BaseRange>> l(
                            *range);

                        auto filled = range->filled(std::memory_order_acquire);
                        if (current + step >= filled)
                        {
                            HPX_ASSERT_LOCKED(l, current < filled);

                            auto delta = filled - current - 1;
                            step -= delta;
                            current += delta;

                            while (step > 0)
                            {
                                --step;
                                ++current;
                                if (range->is_at_end())
                                {
                                    at_end = true;
                                    break;
                                }
                                else
                                {
                                    range->fill_next(l);
                                }
                            }
                        }
                    }
                }

                current += step;
                HPX_ASSERT(
                    current <= static_cast<std::ptrdiff_t>(range->size()));

                // if the iterator is at its end, the current index must be
                // out of range
                HPX_ASSERT(!at_end || current >= range->filled());
            }

            void increment()
            {
                fill_next(1);
            }

            void decrement() noexcept
            {
                --current;
            }

            template <typename Distance>
            void advance(Distance n)
            {
                if (n > 0)
                {
                    fill_next(n);
                }
                else
                {
                    current += n;
                }
            }

            [[nodiscard]] constexpr base_type::reference dereference()
                const noexcept
            {
                HPX_ASSERT(current >= 0 &&
                    current < static_cast<std::ptrdiff_t>(range->size()));
                return range->data[current].data_;
            }

            [[nodiscard]] std::ptrdiff_t distance_to(
                memoizing_iterator const& y) const noexcept
            {
                return static_cast<std::ptrdiff_t>(y.current) -
                    static_cast<std::ptrdiff_t>(current);
            }

            memoizing_range_data<BaseRange>* range;
            std::ptrdiff_t current;
            bool at_end;
        };

        ////////////////////////////////////////////////////////////////////////
        HPX_CXX_EXPORT template <typename BaseRange>
        void intrusive_ptr_add_ref(memoizing_range_data<BaseRange>* p) noexcept;
        HPX_CXX_EXPORT template <typename BaseRange>
        void intrusive_ptr_release(memoizing_range_data<BaseRange>* p) noexcept;

        HPX_CXX_EXPORT template <typename BaseRange>
        struct memoizing_range_data
        {
            using value_type = hpx::traits::iter_value_t<
                hpx::traits::range_iterator_t<BaseRange>>;

            explicit memoizing_range_data(BaseRange&& base, std::size_t size)
              : base_range(HPX_MOVE(base))
              , base_begin(hpx::util::begin(base_range))
              , data(size)    // fill with default constructed elements
              , num_valid(0)
              , count(1)
            {
            }

            memoizing_range_data(memoizing_range_data const&) = delete;
            memoizing_range_data(memoizing_range_data&&) = delete;
            memoizing_range_data& operator=(
                memoizing_range_data const&) = delete;
            memoizing_range_data& operator=(memoizing_range_data&&) = delete;

            [[nodiscard]] memoizing_iterator<BaseRange> begin()
            {
                return memoizing_iterator<BaseRange>(*this, -1);
            }
            [[nodiscard]] memoizing_iterator<BaseRange> end()
            {
                return memoizing_iterator<BaseRange>(*this, data.size(), true);
            }

            [[nodiscard]] std::size_t size() const noexcept
            {
                return data.size();
            }

            [[nodiscard]] std::ptrdiff_t filled(
                std::memory_order const order =
                    std::memory_order_relaxed) const noexcept
            {
                return num_valid.load(order);
            }

            void lock()
            {
                mtx.data_.lock();
            }
            void unlock() noexcept(noexcept(mtx.data_.unlock()))
            {
                mtx.data_.unlock();
            }

        private:
            friend struct memoizing_iterator<BaseRange>;

            [[nodiscard]] bool is_at_end() const noexcept
            {
                return base_begin == hpx::util::end(base_range);
            }

            template <typename Lock>
            void fill_next(Lock& l)
            {
                HPX_ASSERT_OWNS_LOCK(l);
                HPX_ASSERT_LOCKED(
                    l, data.size() > static_cast<std::size_t>(filled()));
                HPX_ASSERT_LOCKED(l, !is_at_end());

                // overwrite the default constructed element
                auto num = num_valid.load(std::memory_order_relaxed);
                num_valid.store(num + 1, std::memory_order_release);

                data[num].data_ = *base_begin;
                ++base_begin;
            }

            hpx::util::cache_aligned_data<hpx::spinlock_no_backoff> mtx;

            BaseRange base_range;
            hpx::util::detail::result_of_begin<BaseRange>::type base_begin;

            std::vector<hpx::util::cache_aligned_data<value_type>> data;
            std::atomic<std::ptrdiff_t> num_valid;

            template <typename Range>
            friend void intrusive_ptr_add_ref(
                memoizing_range_data<Range>* p) noexcept;

            template <typename Range>
            friend void intrusive_ptr_release(
                memoizing_range_data<Range>* p) noexcept;

            hpx::util::atomic_count count;
        };

        HPX_CXX_EXPORT template <typename BaseRange>
        void intrusive_ptr_add_ref(memoizing_range_data<BaseRange>* p) noexcept
        {
            p->count.increment();
        }

        HPX_CXX_EXPORT template <typename BaseRange>
        void intrusive_ptr_release(memoizing_range_data<BaseRange>* p) noexcept
        {
            HPX_ASSERT(p->count != 0);
            if (p->count.decrement() == 0)
            {
                // The thread that decrements the reference count to zero must
                // perform an acquire to ensure that it doesn't start
                // destructing the object until all previous writes have
                // drained.
                std::atomic_thread_fence(std::memory_order_acquire);
                delete p;
            }
        }
    }    // namespace detail

    ////////////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT template <typename BaseRange>
    struct memoizing_range
    {
        explicit memoizing_range(BaseRange&& base, std::size_t size)
          : data(new detail::memoizing_range_data<BaseRange>(
                     HPX_MOVE(base), size),
                false)
        {
        }

        [[nodiscard]] detail::memoizing_iterator<BaseRange> begin() const
        {
            return data->begin();
        }
        [[nodiscard]] detail::memoizing_iterator<BaseRange> end() const
        {
            return data->end();
        }
        [[nodiscard]] std::size_t size() const
        {
            return data->size();
        }

    private:
        hpx::intrusive_ptr<detail::memoizing_range_data<BaseRange>> data;
    };
}    // namespace hpx::parallel::util
