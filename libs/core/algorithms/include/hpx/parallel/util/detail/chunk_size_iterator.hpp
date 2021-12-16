//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/iterator_support/iterator_facade.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/util/min.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util { namespace detail {

    template <typename Iterator, typename Enable = void>
    struct chunk_size_iterator_category;

    template <typename Iterator>
    struct chunk_size_iterator_category<Iterator,
        std::enable_if_t<std::is_integral_v<Iterator>>>
    {
        using type = std::random_access_iterator_tag;
    };

    template <typename Iterator>
    struct chunk_size_iterator_category<Iterator,
        std::enable_if_t<hpx::traits::is_iterator_v<Iterator>>>
    {
        using type = typename std::iterator_traits<Iterator>::iterator_category;
    };

    template <typename Iterator>
    using chunk_size_iterator_category_t =
        typename chunk_size_iterator_category<Iterator>::type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator>
    struct chunk_size_iterator
      : public hpx::util::iterator_facade<chunk_size_iterator<Iterator>,
            hpx::tuple<Iterator, std::size_t> const,
            chunk_size_iterator_category_t<Iterator>>
    {
    private:
        using base_type =
            hpx::util::iterator_facade<chunk_size_iterator<Iterator>,
                hpx::tuple<Iterator, std::size_t> const,
                chunk_size_iterator_category_t<Iterator>>;

        HPX_HOST_DEVICE static constexpr std::size_t get_last_chunk_size(
            std::size_t count, std::size_t chunk_size) noexcept
        {
            std::ptrdiff_t remainder = count % chunk_size;
            if (remainder != 0)
            {
                return remainder;
            }
            return chunk_size;
        }

        HPX_HOST_DEVICE static constexpr std::size_t get_current(
            std::size_t current, std::size_t chunk_size) noexcept
        {
            return (current + chunk_size - 1) / chunk_size * chunk_size;
        }

    public:
        HPX_HOST_DEVICE chunk_size_iterator() = default;

        HPX_HOST_DEVICE chunk_size_iterator(Iterator it, std::size_t chunk_size,
            std::size_t count = 0, std::size_t current = 0)
          : data_(it, (hpx::detail::min)(chunk_size, count))
          , chunk_size_(chunk_size)
          , last_chunk_size_(get_last_chunk_size(count, chunk_size))
          , count_(count)
          , current_(get_current(current, chunk_size))
        {
        }

    private:
        HPX_HOST_DEVICE Iterator& iterator() noexcept
        {
            return hpx::get<0>(data_);
        }
        HPX_HOST_DEVICE constexpr Iterator iterator() const noexcept
        {
            return hpx::get<0>(data_);
        }

        HPX_HOST_DEVICE std::size_t& chunk() noexcept
        {
            return hpx::get<1>(data_);
        }
        HPX_HOST_DEVICE constexpr std::size_t chunk() const noexcept
        {
            return hpx::get<1>(data_);
        }

    protected:
        friend class hpx::util::iterator_core_access;

        HPX_HOST_DEVICE bool equal(chunk_size_iterator const& other) const
        {
            return iterator() == other.iterator() &&
                chunk_size_ == other.chunk_size_ && current_ == other.current_;
        }

        HPX_HOST_DEVICE typename base_type::reference dereference()
            const noexcept
        {
            return data_;
        }

        HPX_HOST_DEVICE void increment(std::size_t offset)
        {
            current_ += offset + chunk_size_;
            if (current_ >= count_)
            {
                // reached the end of the sequence
                iterator() = parallel::v1::detail::next(
                    iterator(), offset + last_chunk_size_);
                chunk() = 0;
            }
            else if (current_ == count_ - last_chunk_size_)
            {
                // reached last chunk
                iterator() = parallel::v1::detail::next(
                    iterator(), offset + chunk_size_);
                chunk() = last_chunk_size_;
            }
            else
            {
                // normal chunk
                HPX_ASSERT(current_ < count_ - last_chunk_size_);
                iterator() = parallel::v1::detail::next(
                    iterator(), offset + chunk_size_);
                chunk() = chunk_size_;
            }
        }

        HPX_HOST_DEVICE void increment()
        {
            increment(0);
        }

        HPX_HOST_DEVICE void decrement(std::size_t offset)
        {
            current_ -= offset + chunk_size_;
            if (current_ == 0)
            {
                // reached the begin of the sequence
                chunk() = current_ + chunk_size_ >= count_ ? last_chunk_size_ :
                                                             chunk_size_;
            }
            else if (current_ == count_ - last_chunk_size_)
            {
                // reached last chunk (was at end)
                chunk() = last_chunk_size_;
            }
            else
            {
                // normal chunk
                HPX_ASSERT(current_ < count_ - last_chunk_size_);
                chunk() = chunk_size_;
            }

            iterator() = parallel::v1::detail::next(
                iterator(), -static_cast<std::ptrdiff_t>(offset + chunk()));
        }

        template <typename Iter = Iterator,
            typename Enable = std::enable_if_t<
                hpx::traits::is_bidirectional_iterator_v<Iter> ||
                std::is_integral_v<Iter>>>
        HPX_HOST_DEVICE void decrement()
        {
            decrement(0);
        }

        template <typename Iter = Iterator,
            typename Enable = std::enable_if_t<
                hpx::traits::is_random_access_iterator_v<Iter> ||
                std::is_integral_v<Iter>>>
        HPX_HOST_DEVICE void advance(std::ptrdiff_t n)
        {
            // prepare next value
            if (n > 0)
            {
                increment((n - 1) * chunk_size_);
            }
            else if (n < 0)
            {
                decrement(-(n + 1) * chunk_size_);
            }
        }

        template <typename Iter = Iterator,
            typename Enable = std::enable_if_t<
                hpx::traits::is_random_access_iterator_v<Iter> ||
                std::is_integral_v<Iter>>>
        HPX_HOST_DEVICE std::ptrdiff_t distance_to(
            chunk_size_iterator const& rhs) const
        {
            return static_cast<std::ptrdiff_t>(
                ((rhs.iterator() - iterator()) + chunk_size_ - 1) /
                chunk_size_);
        }

    private:
        hpx::tuple<Iterator, std::size_t> data_;
        std::size_t chunk_size_ = 0;
        std::size_t last_chunk_size_ = 0;
        std::size_t count_ = 0;
        std::size_t current_ = 0;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator>
    struct chunk_size_idx_iterator
      : public hpx::util::iterator_facade<chunk_size_idx_iterator<Iterator>,
            hpx::tuple<Iterator, std::size_t, std::size_t> const,
            chunk_size_iterator_category_t<Iterator>>
    {
    private:
        using base_type =
            hpx::util::iterator_facade<chunk_size_idx_iterator<Iterator>,
                hpx::tuple<Iterator, std::size_t, std::size_t> const,
                chunk_size_iterator_category_t<Iterator>>;

        HPX_HOST_DEVICE static constexpr std::size_t get_last_chunk_size(
            std::size_t count, std::size_t chunk_size) noexcept
        {
            std::ptrdiff_t remainder = count % chunk_size;
            if (remainder != 0)
            {
                return remainder;
            }
            return chunk_size;
        }

        HPX_HOST_DEVICE static constexpr std::size_t get_current(
            std::size_t current, std::size_t chunk_size) noexcept
        {
            return (current + chunk_size - 1) / chunk_size * chunk_size;
        }

    public:
        HPX_HOST_DEVICE chunk_size_idx_iterator() = default;

        HPX_HOST_DEVICE chunk_size_idx_iterator(Iterator it,
            std::size_t chunk_size, std::size_t count = 0,
            std::size_t current = 0, std::size_t base_idx = 0)
          : data_(it, (hpx::detail::min)(chunk_size, count), base_idx)
          , chunk_size_(chunk_size)
          , last_chunk_size_(get_last_chunk_size(count, chunk_size))
          , count_(count)
          , current_(get_current(current, chunk_size))
        {
        }

    private:
        HPX_HOST_DEVICE Iterator& iterator() noexcept
        {
            return hpx::get<0>(data_);
        }
        HPX_HOST_DEVICE constexpr Iterator iterator() const noexcept
        {
            return hpx::get<0>(data_);
        }

        HPX_HOST_DEVICE std::size_t& chunk() noexcept
        {
            return hpx::get<1>(data_);
        }
        HPX_HOST_DEVICE constexpr std::size_t chunk() const noexcept
        {
            return hpx::get<1>(data_);
        }

        HPX_HOST_DEVICE std::size_t& base_index() noexcept
        {
            return hpx::get<2>(data_);
        }

        HPX_HOST_DEVICE constexpr std::size_t base_index() const noexcept
        {
            return hpx::get<2>(data_);
        }

    protected:
        friend class hpx::util::iterator_core_access;

        HPX_HOST_DEVICE bool equal(chunk_size_idx_iterator const& other) const
        {
            return iterator() == other.iterator() &&
                chunk_size_ == other.chunk_size_ && current_ == other.current_;
        }

        HPX_HOST_DEVICE typename base_type::reference dereference()
            const noexcept
        {
            return data_;
        }

        HPX_HOST_DEVICE void increment(std::size_t offset)
        {
            base_index() += offset + chunk_size_;
            current_ += offset + chunk_size_;
            if (current_ >= count_)
            {
                // reached the end of the sequence
                iterator() = parallel::v1::detail::next(
                    iterator(), offset + last_chunk_size_);
                chunk() = 0;
            }
            else if (current_ == count_ - last_chunk_size_)
            {
                // reached last chunk
                iterator() = parallel::v1::detail::next(
                    iterator(), offset + chunk_size_);
                chunk() = last_chunk_size_;
            }
            else
            {
                // normal chunk
                HPX_ASSERT(current_ < count_ - last_chunk_size_);
                iterator() = parallel::v1::detail::next(
                    iterator(), offset + chunk_size_);
                chunk() = chunk_size_;
            }
        }

        HPX_HOST_DEVICE void increment()
        {
            increment(0);
        }

        HPX_HOST_DEVICE void decrement(std::size_t offset)
        {
            base_index() -= offset + chunk_size_;
            current_ -= offset + chunk_size_;
            if (current_ == 0)
            {
                // reached the begin of the sequence
                chunk() = base_index() + chunk_size_ >= count_ ?
                    last_chunk_size_ :
                    chunk_size_;
            }
            else if (current_ == count_ - last_chunk_size_)
            {
                // reached last chunk (was at end)
                chunk() = last_chunk_size_;
            }
            else
            {
                // normal chunk
                HPX_ASSERT(current_ < count_ - last_chunk_size_);
                chunk() = chunk_size_;
            }

            iterator() = parallel::v1::detail::next(
                iterator(), -static_cast<std::ptrdiff_t>(offset + chunk()));
        }

        template <typename Iter = Iterator,
            typename Enable = std::enable_if_t<
                hpx::traits::is_bidirectional_iterator_v<Iter> ||
                std::is_integral_v<Iter>>>
        HPX_HOST_DEVICE void decrement()
        {
            decrement(0);
        }

        template <typename Iter = Iterator,
            typename Enable = std::enable_if_t<
                hpx::traits::is_random_access_iterator_v<Iter> ||
                std::is_integral_v<Iter>>>
        HPX_HOST_DEVICE void advance(std::ptrdiff_t n)
        {
            // prepare next value
            if (n > 0)
            {
                increment((n - 1) * chunk_size_);
            }
            else if (n < 0)
            {
                decrement(-(n + 1) * chunk_size_);
            }
        }

        template <typename Iter = Iterator,
            typename Enable = std::enable_if_t<
                hpx::traits::is_random_access_iterator_v<Iter> ||
                std::is_integral_v<Iter>>>
        HPX_HOST_DEVICE std::ptrdiff_t distance_to(
            chunk_size_idx_iterator const& rhs) const
        {
            return static_cast<std::ptrdiff_t>(
                ((rhs.iterator() - iterator()) + chunk_size_ - 1) /
                chunk_size_);
        }

    private:
        hpx::tuple<Iterator, std::size_t, std::size_t> data_;
        std::size_t chunk_size_ = 0;
        std::size_t last_chunk_size_ = 0;
        std::size_t count_ = 0;
        std::size_t current_ = 0;
    };
}}}}    // namespace hpx::parallel::util::detail
