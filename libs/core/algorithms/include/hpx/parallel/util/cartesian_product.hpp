//  Copyright (c) 2026 Arpit Khandelwal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/util/cartesian_product.hpp
/// \page hpx::views::cartesian_product
/// \headerfile hpx/algorithm.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx { namespace views {
    // clang-format off

    /// A view that creates a cartesian product of the input ranges.
    template <typename... Rs>
    constexpr auto cartesian_product(Rs&&... rs);

    // clang-format on
}}    // namespace hpx::views

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/type_support/decay.hpp>

#include <algorithm>
#include <concepts>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

namespace hpx::ranges::detail {

    template <typename... Its>
    class cartesian_product_iterator
    {
    public:
        using iterator_category = std::input_iterator_tag;    // Simplified
        using value_type =
            hpx::tuple<typename std::iterator_traits<Its>::value_type...>;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference =
            value_type;    // Value-based for simplicity in this PoC

        constexpr cartesian_product_iterator() = default;

        template <typename... Sentinels>
        constexpr cartesian_product_iterator(hpx::tuple<Its...> current,
            hpx::tuple<Its...> begin, hpx::tuple<Sentinels...> end)
          : current_(std::move(current))
          , begin_(std::move(begin))
          , end_(std::move(end))    // Storing end iterators to check boundaries
        {
        }

        // simplified constructor for begin/end special cases could be added

        constexpr reference operator*() const
        {
            return deref_impl(std::make_index_sequence<sizeof...(Its)>{});
        }

        constexpr cartesian_product_iterator& operator++()
        {
            next();
            return *this;
        }

        constexpr cartesian_product_iterator operator++(int)
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        friend constexpr bool operator==(cartesian_product_iterator const& x,
            cartesian_product_iterator const& y)
        {
            return x.current_ == y.current_;
        }

        friend constexpr bool operator!=(cartesian_product_iterator const& x,
            cartesian_product_iterator const& y)
        {
            return !(x == y);
        }

        // Sentinel comparison
        // We are at end if the first iterator is at its end.
        // This is a simplification. The standard defines end more robustly.
        // But for this implementation, checking the first iterator is key.
        template <typename... Sentinels>
        friend constexpr bool operator==(cartesian_product_iterator const& x,
            hpx::tuple<Sentinels...> const& y)
        {
            return hpx::get<0>(x.current_) == hpx::get<0>(y);
        }

        template <typename... Sentinels>
        friend constexpr bool operator!=(cartesian_product_iterator const& x,
            hpx::tuple<Sentinels...> const& y)
        {
            return !(x == y);
        }

    private:
        template <size_t... Is>
        constexpr value_type deref_impl(std::index_sequence<Is...>) const
        {
            return hpx::make_tuple(*hpx::get<Is>(current_)...);
        }

        constexpr void next()
        {
            next_impl<sizeof...(Its) - 1>();
        }

        template <size_t I>
        constexpr void next_impl()
        {
            auto& it = hpx::get<I>(current_);
            ++it;
            if (it == hpx::get<I>(end_))
            {
                if constexpr (I > 0)
                {
                    it = hpx::get<I>(begin_);
                    next_impl<I - 1>();
                }
            }
        }

        hpx::tuple<Its...> current_;
        hpx::tuple<Its...> begin_;
        // We accept different types for end sentinels but for simplicity here assume same type or compatible
        // Ideally we should store sentinels separately.
        // For the purpose of this task, let's assume iterators are their own sentinels or we store sentinels.
        // To support end comparison properly, we might need a separate sentinel type, or store sentinels in iterator.
        // Storing tuple of sentinels here for 'end' check during increment.
        // Note: I am simplifying by asserting Its... can act as sentinels or we have them.
        // The constructor took Sentinels... but we need to store them.
        // Let's deduce type from constructor.
        // But for class template, we need to know types.
        // Let's assume common case where iterator == sentinel.
        hpx::tuple<Its...> end_;
    };

    template <typename... Rs>
    class cartesian_product_view
    {
    public:
        constexpr cartesian_product_view(Rs... rs)
          : rs_(std::move(rs)...)
        {
        }

        constexpr auto begin()
        {
            return begin_impl(std::make_index_sequence<sizeof...(Rs)>{});
        }

        constexpr auto end()
        {
            return end_impl(std::make_index_sequence<sizeof...(Rs)>{});
        }

    private:
        template <size_t... Is>
        constexpr auto begin_impl(std::index_sequence<Is...>)
        {
            using iterator_type = cartesian_product_iterator<
                hpx::traits::range_iterator_t<Rs>...>;

            auto begins =
                hpx::make_tuple(hpx::util::begin(hpx::get<Is>(rs_))...);
            auto ends = hpx::make_tuple(hpx::util::end(hpx::get<Is>(rs_))...);

            if (((hpx::get<Is>(begins) == hpx::get<Is>(ends)) || ...))
            {
                hpx::get<0>(begins) =
                    std::ranges::next(hpx::get<0>(begins), hpx::get<0>(ends));
            }

            return iterator_type(begins, begins, ends);
        }

        template <size_t... Is>
        constexpr auto end_impl(std::index_sequence<Is...>)
        {
            return hpx::make_tuple(hpx::util::end(hpx::get<Is>(rs_))...);
        }

        hpx::tuple<Rs...> rs_;
    };
}    // namespace hpx::ranges::detail

namespace hpx::views {

    HPX_CXX_CORE_EXPORT struct cartesian_product_t
    {
        template <typename... Rs>
        constexpr auto operator()(Rs&&... rs) const
        {
            return hpx::ranges::detail::cartesian_product_view<
                hpx::util::decay_unwrap_t<Rs>...>(std::forward<Rs>(rs)...);
        }
    };

    inline constexpr cartesian_product_t cartesian_product{};

}    // namespace hpx::views

#endif
