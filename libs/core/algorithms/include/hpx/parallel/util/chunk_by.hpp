//  Copyright (c) 2026 Arpit Khandelwal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/util/chunk_by.hpp
/// \page hpx::views::chunk_by
/// \headerfile hpx/algorithm.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx { namespace views {
    // clang-format off

    /// A view that splits a range into subranges based on a binary predicate.
    template <typename R, typename Pred>
    constexpr auto chunk_by(R&& r, Pred pred);

    // clang-format on
}}    // namespace hpx::views

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/iterator_support/iterator_range.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/type_support/decay.hpp>

#include <hpx/config.hpp>
#include <algorithm>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx::ranges::detail {

    template <typename I, typename S, typename Pred>
    class chunk_by_iterator
    {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = hpx::util::iterator_range<I, I>;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type;

        constexpr chunk_by_iterator() = default;

        constexpr chunk_by_iterator(I current, S last, Pred pred)
          : current_(current)
          , next_(current)
          , last_(last)
          , pred_(pred)
        {
            if (current_ != last_)
            {
                find_next();
            }
        }

        constexpr reference operator*() const
        {
            return {current_, next_};
        }

        constexpr chunk_by_iterator& operator++()
        {
            current_ = next_;
            if (current_ != last_)
            {
                find_next();
            }
            return *this;
        }

        constexpr chunk_by_iterator operator++(int)
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        friend constexpr bool operator==(
            chunk_by_iterator const& x, chunk_by_iterator const& y)
        {
            return x.current_ == y.current_;
        }

        friend constexpr bool operator!=(
            chunk_by_iterator const& x, chunk_by_iterator const& y)
        {
            return !(x == y);
        }

        friend constexpr bool operator==(chunk_by_iterator const& x, S const& y)
        {
            return x.current_ == y;
        }

        friend constexpr bool operator!=(chunk_by_iterator const& x, S const& y)
        {
            return !(x == y);
        }

    private:
        constexpr void find_next()
        {
            next_ = std::adjacent_find(current_, last_, std::not_fn(pred_));
            if (next_ != last_)
            {
                ++next_;
            }
        }

        I current_{};
        I next_{};
        HPX_NO_UNIQUE_ADDRESS S last_{};
        HPX_NO_UNIQUE_ADDRESS Pred pred_{};
    };

    template <typename R, typename Pred>
    class chunk_by_view
    {
        using iterator_t = hpx::traits::range_iterator_t<R>;
        using sentinel_t = hpx::traits::range_sentinel_t<R>;

    public:
        constexpr chunk_by_view(R r, Pred pred)
          : r_(std::move(r))
          , pred_(std::move(pred))
        {
        }

        constexpr auto begin()
        {
            return chunk_by_iterator<iterator_t, sentinel_t, Pred>(
                hpx::util::begin(r_), hpx::util::end(r_), pred_);
        }

        constexpr auto end()
        {
            return hpx::util::end(r_);
        }

    private:
        R r_;
        HPX_NO_UNIQUE_ADDRESS Pred pred_;
    };
}    // namespace hpx::ranges::detail

namespace hpx::views {

    HPX_CXX_CORE_EXPORT struct chunk_by_t
    {
        template <typename R, typename Pred>
        constexpr auto operator()(R&& r, Pred&& pred) const
        {
            return hpx::ranges::detail::chunk_by_view<
                hpx::util::decay_unwrap_t<R>, std::decay_t<Pred>>(
                std::forward<R>(r), std::forward<Pred>(pred));
        }

        template <typename Pred>
        constexpr auto operator()(Pred&& /*pred*/) const
        {
            return hpx::functional::detail::tag_fallback<chunk_by_t>{};
        }
    };

    inline constexpr chunk_by_t chunk_by{};

}    // namespace hpx::views

#endif
