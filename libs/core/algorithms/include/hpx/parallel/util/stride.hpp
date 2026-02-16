//  Copyright (c) 2026 Arpit Khandelwal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/util/stride.hpp
/// \page hpx::views::stride
/// \headerfile hpx/algorithm.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx { namespace views {
    // clang-format off

    /// A view that traverses the range with a given stride.
    HPX_CXX_CORE_EXPORT template <typename R>
    constexpr auto stride(R&& r, std::ptrdiff_t n);

    // clang-format on
}}    // namespace hpx::views

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/type_support/decay.hpp>

#include <algorithm>
#include <concepts>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx::ranges::detail {

    template <typename I, typename S>
    class stride_iterator
    {
    public:
        using iterator_category = std::forward_iterator_tag;    // simplified
        using value_type = typename std::iterator_traits<I>::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = typename std::iterator_traits<I>::pointer;
        using reference = typename std::iterator_traits<I>::reference;

        constexpr stride_iterator() = default;

        constexpr stride_iterator(I current, S last, std::ptrdiff_t n)
          : current_(current)
          , last_(last)
          , stride_(n)
        {
        }

        constexpr reference operator*() const
        {
            return *current_;
        }

        constexpr stride_iterator& operator++()
        {
            std::ranges::advance(current_, stride_, last_);
            return *this;
        }

        constexpr stride_iterator operator++(int)
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        friend constexpr bool operator==(
            stride_iterator const& x, stride_iterator const& y)
        {
            return x.current_ == y.current_;
        }

        friend constexpr bool operator!=(
            stride_iterator const& x, stride_iterator const& y)
        {
            return !(x == y);
        }

        friend constexpr bool operator==(stride_iterator const& x, S const& y)
        {
            return x.current_ == y;
        }

        friend constexpr bool operator!=(stride_iterator const& x, S const& y)
        {
            return !(x == y);
        }

    private:
        I current_{};
        HPX_NO_UNIQUE_ADDRESS S last_{};
        std::ptrdiff_t stride_{};
    };

    template <typename R>
    class stride_view
    {
        using iterator_t = hpx::traits::range_iterator_t<R>;
        using sentinel_t = hpx::traits::range_sentinel_t<R>;

    public:
        constexpr stride_view(R r, std::ptrdiff_t n)
          : r_(std::move(r))
          , n_(n)
        {
        }

        constexpr auto begin()
        {
            return stride_iterator<iterator_t, sentinel_t>(
                hpx::util::begin(r_), hpx::util::end(r_), n_);
        }

        constexpr auto end()
        {
            return hpx::util::end(r_);
        }

    private:
        R r_;
        std::ptrdiff_t n_;
    };
}    // namespace hpx::ranges::detail

namespace hpx::views {

    struct stride_t
    {
        template <typename R>
        constexpr auto operator()(R&& r, std::ptrdiff_t n) const
        {
            return hpx::ranges::detail::stride_view<
                hpx::util::decay_unwrap_t<R>>(std::forward<R>(r), n);
        }
    };

    inline constexpr stride_t stride{};

}    // namespace hpx::views

#endif
