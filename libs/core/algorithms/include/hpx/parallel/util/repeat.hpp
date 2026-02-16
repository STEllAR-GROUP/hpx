//  Copyright (c) 2026 Arpit Khandelwal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/util/repeat.hpp
/// \page hpx::views::repeat
/// \headerfile hpx/algorithm.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx { namespace views {
    // clang-format off

    /// A view that repeats a value n times or infinitely.
    template <typename T>
    constexpr auto repeat(T&& value);

    template <typename T>
    constexpr auto repeat(T&& value, std::ptrdiff_t n);

    // clang-format on
}}    // namespace hpx::views

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/type_support/decay.hpp>

#include <concepts>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx::ranges::detail {

    template <typename T>
    struct repeat_sentinel
    {
        std::ptrdiff_t bound = -1;    // -1 means infinite
    };

    template <typename T>
    class repeat_iterator
    {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T const*;
        using reference = T const&;

        constexpr repeat_iterator() = default;

        constexpr explicit repeat_iterator(
            T const* value, std::ptrdiff_t current = 0)
          : value_(value)
          , current_(current)
        {
        }

        constexpr reference operator*() const
        {
            return *value_;
        }

        constexpr repeat_iterator& operator++()
        {
            ++current_;
            return *this;
        }

        constexpr repeat_iterator operator++(int)
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        constexpr repeat_iterator& operator--()
        {
            --current_;
            return *this;
        }

        constexpr repeat_iterator operator--(int)
        {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        constexpr repeat_iterator& operator+=(difference_type n)
        {
            current_ += n;
            return *this;
        }

        constexpr repeat_iterator& operator-=(difference_type n)
        {
            current_ -= n;
            return *this;
        }

        constexpr reference operator[](difference_type n) const
        {
            return *value_;
        }

        friend constexpr bool operator==(
            repeat_iterator const& x, repeat_iterator const& y)
        {
            return x.current_ == y.current_;
        }

        friend constexpr bool operator!=(
            repeat_iterator const& x, repeat_iterator const& y)
        {
            return !(x == y);
        }

        friend constexpr bool operator==(
            repeat_iterator const& x, repeat_sentinel<T> const& y)
        {
            if (y.bound < 0)
                return false;    // Infinite, never equal to sentinel
            return x.current_ == y.bound;
        }

        friend constexpr bool operator!=(
            repeat_iterator const& x, repeat_sentinel<T> const& y)
        {
            return !(x == y);
        }

        friend constexpr bool operator<(
            repeat_iterator const& x, repeat_iterator const& y)
        {
            return x.current_ < y.current_;
        }

        friend constexpr bool operator>(
            repeat_iterator const& x, repeat_iterator const& y)
        {
            return x.current_ > y.current_;
        }

        friend constexpr bool operator<=(
            repeat_iterator const& x, repeat_iterator const& y)
        {
            return x.current_ <= y.current_;
        }

        friend constexpr bool operator>=(
            repeat_iterator const& x, repeat_iterator const& y)
        {
            return x.current_ >= y.current_;
        }

        friend constexpr difference_type operator-(
            repeat_iterator const& x, repeat_iterator const& y)
        {
            return x.current_ - y.current_;
        }

        friend constexpr repeat_iterator operator+(
            repeat_iterator const& x, difference_type y)
        {
            return repeat_iterator(x.value_, x.current_ + y);
        }

        friend constexpr repeat_iterator operator+(
            difference_type x, repeat_iterator const& y)
        {
            return y + x;
        }

        friend constexpr repeat_iterator operator-(
            repeat_iterator const& x, difference_type y)
        {
            return repeat_iterator(x.value_, x.current_ - y);
        }

        friend constexpr difference_type operator-(
            repeat_iterator const& x, repeat_sentinel<T> const& y)
        {
            return x.current_ - y.bound;
        }

        friend constexpr difference_type operator-(
            repeat_sentinel<T> const& y, repeat_iterator const& x)
        {
            return y.bound - x.current_;
        }

    private:
        T const* value_{nullptr};
        difference_type current_{0};
    };

    template <typename T>
    class repeat_view
    {
    public:
        using iterator = repeat_iterator<T>;
        using sentinel = repeat_sentinel<T>;

        constexpr explicit repeat_view(T value)
          : value_(std::move(value))
          , n_(-1)    // Infinite
        {
        }

        constexpr repeat_view(T value, std::ptrdiff_t n)
          : value_(std::move(value))
          , n_(n)
        {
        }

        constexpr auto begin() const
        {
            return iterator(&value_, 0);
        }

        constexpr auto end() const
        {
            return sentinel{n_};
        }

    private:
        T value_;
        std::ptrdiff_t n_;
    };

}    // namespace hpx::ranges::detail

namespace hpx::views {

    HPX_CXX_CORE_EXPORT struct repeat_t
    {
        template <typename T>
        constexpr auto operator()(T&& value) const
        {
            return hpx::ranges::detail::repeat_view<std::decay_t<T>>(
                std::forward<T>(value));
        }

        template <typename T>
        constexpr auto operator()(T&& value, std::ptrdiff_t n) const
        {
            return hpx::ranges::detail::repeat_view<std::decay_t<T>>(
                std::forward<T>(value), n);
        }
    };

    inline constexpr repeat_t repeat{};

}    // namespace hpx::views

#endif
