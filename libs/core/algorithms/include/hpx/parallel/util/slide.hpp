//  Copyright (c) 2026 Arpit Khandelwal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/util/slide.hpp
/// \page hpx::views::slide
/// \headerfile hpx/algorithm.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx { namespace views {
    // clang-format off

    /// A view that creates a sliding window of size N over a range.
    template <typename R>
    constexpr auto slide(R&& r, std::ptrdiff_t n);

    // clang-format on
}}    // namespace hpx::views

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/iterator_support/iterator_range.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/type_support/decay.hpp>

#include <algorithm>
#include <iterator>
#include <utility>

namespace hpx::ranges::detail {

    template <typename I, typename S>
    class slide_iterator
    {
    public:
        using iterator_category = std::forward_iterator_tag;    // simplified
        using value_type = hpx::util::iterator_range<I, I>;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type;

        constexpr slide_iterator() = default;

        constexpr slide_iterator(
            I current, I window_end, S last, bool at_end = false)
          : current_(current)
          , window_end_(window_end)
          , last_(last)
          , at_end_(at_end)
        {
        }

        constexpr reference operator*() const
        {
            return value_type(current_, window_end_);
        }

        constexpr slide_iterator& operator++()
        {
            if (window_end_ == last_)
            {
                at_end_ = true;
            }
            else
            {
                ++current_;
                ++window_end_;
            }
            return *this;
        }

        constexpr slide_iterator operator++(int)
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        friend constexpr bool operator==(
            slide_iterator const& x, slide_iterator const& y)
        {
            if (x.at_end_ && y.at_end_)
                return true;
            if (x.at_end_ != y.at_end_)
                return false;
            return x.current_ == y.current_;
        }

        friend constexpr bool operator!=(
            slide_iterator const& x, slide_iterator const& y)
        {
            return !(x == y);
        }

        friend constexpr bool operator==(
            slide_iterator const& x, S const& /* y */)
        {
            return x.at_end_;
        }

        friend constexpr bool operator!=(slide_iterator const& x, S const& y)
        {
            return !(x == y);
        }

    private:
        I current_{};
        I window_end_{};
        HPX_NO_UNIQUE_ADDRESS S last_{};
        bool at_end_{false};
    };

    template <typename R>
    class slide_view
    {
        using iterator_t = hpx::traits::range_iterator_t<R>;
        using sentinel_t = hpx::traits::range_sentinel_t<R>;

    public:
        constexpr slide_view(R r, std::ptrdiff_t n)
          : r_(std::move(r))
          , n_(n)
        {
        }

        constexpr auto begin()
        {
            auto first = hpx::util::begin(r_);
            auto last = hpx::util::end(r_);

            // Calculate window end for the first window
            auto window_end = first;
            std::ptrdiff_t d = 0;
            // Advance window_end n steps, checking for end
            while (d < n_ && window_end != last)
            {
                ++window_end;
                ++d;
            }

            if (d < n_)
            {
                // Range is smaller than n, return end iterator (constructed as slide_iterator)
                return slide_iterator<iterator_t, sentinel_t>(
                    last, last, last, true);
            }

            return slide_iterator<iterator_t, sentinel_t>(
                first, window_end, last, false);
        }

        constexpr auto end()
        {
            // Sentinel type logic could be refined but using default constructed iterator
            // as sentinel if implemented correctly, or just returning sentinel type.
            // My iterator comparison uses sentinel type S directly.
            // So end() returns a 'S' (the original sentinel)?
            // Wait, begin() returns an iterator.
            // If iterator comparis with 'S' is valid.
            // slide_iterator operator== takes S.
            // So end() should return S (sentinel_t).
            return hpx::util::end(r_);
        }

    private:
        R r_;
        std::ptrdiff_t n_;
    };
}    // namespace hpx::ranges::detail

namespace hpx::views {

    HPX_CXX_CORE_EXPORT struct slide_t
    {
        template <typename R>
        constexpr auto operator()(R&& r, std::ptrdiff_t n) const
        {
            return hpx::ranges::detail::slide_view<
                hpx::util::decay_unwrap_t<R>>(std::forward<R>(r), n);
        }
    };

    inline constexpr slide_t slide{};

}    // namespace hpx::views

#endif
