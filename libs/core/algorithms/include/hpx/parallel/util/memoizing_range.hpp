//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/concurrency/cache_line_data.hpp>
#include <hpx/iterator_support/iterator_facade.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/memory/intrusive_ptr.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/thread_support/atomic_count.hpp>

#include <atomic>
#include <cstddef>
#include <iterator>
#include <mutex>
#include <type_traits>

namespace hpx::parallel::util {

    template <typename BaseRange>
    struct memoizing_range;

    namespace detail {

        template <typename BaseRange>
        struct memoizing_range_data;

        template <typename BaseRange>
        struct memoizing_iterator;

        template <typename BaseRange>
        struct memoizing_iterator_base
        {
            using base_iterator = hpx::traits::range_iterator_t<BaseRange>;
            using value_type = hpx::traits::iter_value_t<base_iterator>;
            using type =
                hpx::util::iterator_facade<memoizing_iterator<BaseRange>,
                    value_type, std::random_access_iterator_tag>;
        };

        template <typename BaseRange>
        struct memoizing_iterator : memoizing_iterator_base<BaseRange>::type
        {
            explicit memoizing_iterator(memoizing_range_data<BaseRange>& base,
                std::ptrdiff_t const index, bool const end = false)
              : range(&base)
              , current(index)
              , at_end(end)
            {
                // fill in first element, if needed
                fill_next();
            }

        private:
            friend class hpx::util::iterator_core_access;

            using base_type = typename memoizing_iterator_base<BaseRange>::type;

            [[nodiscard]] constexpr bool equal(
                memoizing_iterator const& rhs) const noexcept
            {
                return current == rhs.current || (at_end && rhs.at_end);
            }

            void fill_next()
            {
                if (!at_end)
                {
                    if (current + 1 >= range->filled())
                    {
                        std::scoped_lock<memoizing_range_data<BaseRange>> l(
                            *range);
                        if (current + 1 >= range->filled())
                        {
                            if (range->is_at_end())
                            {
                                at_end = true;
                            }
                            else
                            {
                                range->fill_next();
                            }
                        }
                    }
                    ++current;
                }

                // if the iterator is at its end, the current index must be
                // out of range
                HPX_ASSERT(!at_end || current >= range->filled());
            }

            void increment() noexcept
            {
                fill_next();
            }

            void decrement() noexcept
            {
                --current;
            }

            template <typename Distance>
            void advance(Distance n) noexcept
            {
                if (n > 0)
                {
                    while (n-- != 0 && !at_end)
                    {
                        fill_next();
                    }
                }
                else
                {
                    current += n;
                }
            }

            [[nodiscard]] constexpr typename base_type::reference dereference()
                const noexcept
            {
                HPX_ASSERT(current >= 0 &&
                    current < static_cast<std::ptrdiff_t>(range->data.size()));
                return range->data[current];
            }

            [[nodiscard]] std::ptrdiff_t distance_to(
                memoizing_iterator const& y) const noexcept
            {
                return static_cast<std::ptrdiff_t>(y.current) -
                    static_cast<std::ptrdiff_t>(current);
            }

            hpx::intrusive_ptr<memoizing_range_data<BaseRange>> range;
            std::ptrdiff_t current;
            bool at_end;
        };

        ////////////////////////////////////////////////////////////////////////
        template <typename BaseRange>
        void intrusive_ptr_add_ref(memoizing_range_data<BaseRange>* p) noexcept;
        template <typename BaseRange>
        void intrusive_ptr_release(memoizing_range_data<BaseRange>* p) noexcept;

        template <typename BaseRange>
        struct memoizing_range_data
        {
            using value_type = hpx::traits::iter_value_t<
                hpx::traits::range_iterator_t<BaseRange>>;

            explicit memoizing_range_data(BaseRange&& base, std::size_t size)
              : base_range(HPX_MOVE(base))
              , base_begin(hpx::util::begin(base_range))
              , base_end(hpx::util::end(base_range))
              , data(size)    // fill with default constructed elements
              , num_valid(0)
              , count(1)
            {
            }

            memoizing_iterator<BaseRange> begin()
            {
                return memoizing_iterator<BaseRange>(*this, -1);
            }
            memoizing_iterator<BaseRange> end()
            {
                return memoizing_iterator<BaseRange>(
                    *this, data.capacity(), true);
            }

            std::size_t size() const
            {
                return data.size();
            }

            std::ptrdiff_t filled(
                std::memory_order order = std::memory_order_relaxed) const
            {
                return num_valid.load(order);
            }

            void lock()
            {
                mtx.data_.lock();
            }
            void unlock()
            {
                mtx.data_.unlock();
            }

        private:
            friend struct memoizing_iterator<BaseRange>;

            bool is_at_end() const
            {
                return base_begin == base_end;
            }

            void fill_next()
            {
                HPX_ASSERT(data.size() > static_cast<std::size_t>(filled()));
                HPX_ASSERT(!is_at_end());

                // overwrite the default constructed element
                data[num_valid.load(std::memory_order_relaxed)] = *base_begin;
                ++base_begin;
                ++num_valid;
            }

            hpx::util::cache_aligned_data<hpx::spinlock> mtx;

            BaseRange base_range;
            typename hpx::util::detail::result_of_begin<BaseRange>::type
                base_begin;
            typename hpx::util::detail::result_of_end<BaseRange>::type base_end;

            std::vector<value_type> data;
            std::atomic<std::ptrdiff_t> num_valid;

            template <typename Range>
            friend void intrusive_ptr_add_ref(
                memoizing_range_data<Range>* p) noexcept;

            template <typename Range>
            friend void intrusive_ptr_release(
                memoizing_range_data<Range>* p) noexcept;

            hpx::util::atomic_count count;
        };

        template <typename BaseRange>
        void intrusive_ptr_add_ref(memoizing_range_data<BaseRange>* p) noexcept
        {
            ++p->count;
        }

        template <typename BaseRange>
        void intrusive_ptr_release(memoizing_range_data<BaseRange>* p) noexcept
        {
            HPX_ASSERT(p->count != 0);
            if (--p->count == 0)
            {
                delete p;
            }
        }
    }    // namespace detail

    ////////////////////////////////////////////////////////////////////////////
    template <typename BaseRange>
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
