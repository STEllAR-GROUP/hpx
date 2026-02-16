//  Copyright (c) 2026 Arpit Khandelwal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/fold_left.hpp
/// \page hpx::ranges::fold_left
/// \headerfile hpx/algorithm.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx { namespace ranges {
    // clang-format off

    /// Loose implementation of the C++23 fold_left algorithm
    template <typename R, typename T, typename F>
    constexpr auto fold_left(R&& r, T init, F f);

    /// Loose implementation of the C++23 fold_left_with_iter algorithm
    template <typename R, typename T, typename F>
    constexpr auto fold_left_with_iter(R&& r, T init, F f);

    /// Loose implementation of the C++23 fold_left_first algorithm
    template <typename R, typename F>
    constexpr auto fold_left_first(R&& r, F f);

    // clang-format on
}}    // namespace hpx::ranges

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <algorithm>
#include <concepts>
#include <iterator>
#include <optional>
#include <type_traits>
#include <utility>

namespace hpx::ranges::detail {
    template <typename I, typename S, typename T, typename F>
    constexpr auto fold_left_with_iter_impl(I first, S last, T init, F f)
    {
        using U = std::decay_t<T>;
        if (first == last)
        {
            return hpx::parallel::util::in_value_result<I, U>{
                std::move(first), std::move(init)};
        }

        U accum = std::invoke(f, std::move(init), *first);
        for (++first; first != last; ++first)
        {
            accum = std::invoke(f, std::move(accum), *first);
        }

        return hpx::parallel::util::in_value_result<I, U>{
            std::move(first), std::move(accum)};
    }

    template <typename I, typename S, typename F>
    constexpr auto fold_left_first_with_iter_impl(I first, S last, F f)
    {
        using U = typename std::iterator_traits<I>::value_type;
        using Res = hpx::parallel::util::in_value_result<I, std::optional<U>>;

        if (first == last)
        {
            return Res{std::move(first), std::nullopt};
        }

        std::optional<U> accum(std::in_place, *first);
        for (++first; first != last; ++first)
        {
            *accum = std::invoke(f, std::move(*accum), *first);
        }

        return Res{std::move(first), std::move(accum)};
    }
}    // namespace hpx::ranges::detail

namespace hpx::ranges {

    // ----------------------------------------------------------------------
    // fold_left_with_iter
    // ----------------------------------------------------------------------
    HPX_CXX_CORE_EXPORT struct fold_left_with_iter_t final
      : hpx::functional::detail::tag_fallback<fold_left_with_iter_t>
    {
    private:
        template <typename I, typename S, typename T, typename F>
        friend constexpr auto tag_fallback_invoke(
            fold_left_with_iter_t, I first, S last, T init, F f)
        {
            return hpx::ranges::detail::fold_left_with_iter_impl(
                std::move(first), std::move(last), std::move(init),
                std::move(f));
        }

        template <typename R, typename T, typename F>
        friend constexpr auto tag_fallback_invoke(
            fold_left_with_iter_t, R&& r, T init, F f)
        {
            return hpx::ranges::detail::fold_left_with_iter_impl(
                hpx::util::begin(r), hpx::util::end(r), std::move(init),
                std::move(f));
        }
    };

    inline constexpr fold_left_with_iter_t fold_left_with_iter{};

    // ----------------------------------------------------------------------
    // fold_left
    // ----------------------------------------------------------------------
    HPX_CXX_CORE_EXPORT struct fold_left_t final
      : hpx::functional::detail::tag_fallback<fold_left_t>
    {
    private:
        template <typename I, typename S, typename T, typename F>
        friend constexpr auto tag_fallback_invoke(
            fold_left_t, I first, S last, T init, F f)
        {
            return hpx::ranges::detail::fold_left_with_iter_impl(
                std::move(first), std::move(last), std::move(init),
                std::move(f))
                .value;
        }

        template <typename R, typename T, typename F>
        friend constexpr auto tag_fallback_invoke(
            fold_left_t, R&& r, T init, F f)
        {
            return hpx::ranges::detail::fold_left_with_iter_impl(
                hpx::util::begin(r), hpx::util::end(r), std::move(init),
                std::move(f))
                .value;
        }
    };

    inline constexpr fold_left_t fold_left{};

    // ----------------------------------------------------------------------
    // fold_left_first_with_iter
    // ----------------------------------------------------------------------
    HPX_CXX_CORE_EXPORT struct fold_left_first_with_iter_t final
      : hpx::functional::detail::tag_fallback<fold_left_first_with_iter_t>
    {
    private:
        template <typename I, typename S, typename F>
        friend constexpr auto tag_fallback_invoke(
            fold_left_first_with_iter_t, I first, S last, F f)
        {
            return hpx::ranges::detail::fold_left_first_with_iter_impl(
                std::move(first), std::move(last), std::move(f));
        }

        template <typename R, typename F>
        friend constexpr auto tag_fallback_invoke(
            fold_left_first_with_iter_t, R&& r, F f)
        {
            return hpx::ranges::detail::fold_left_first_with_iter_impl(
                hpx::util::begin(r), hpx::util::end(r), std::move(f));
        }
    };

    inline constexpr fold_left_first_with_iter_t fold_left_first_with_iter{};

    // ----------------------------------------------------------------------
    // fold_left_first
    // ----------------------------------------------------------------------
    HPX_CXX_CORE_EXPORT struct fold_left_first_t final
      : hpx::functional::detail::tag_fallback<fold_left_first_t>
    {
    private:
        template <typename I, typename S, typename F>
        friend constexpr auto tag_fallback_invoke(
            fold_left_first_t, I first, S last, F f)
        {
            return hpx::ranges::detail::fold_left_first_with_iter_impl(
                std::move(first), std::move(last), std::move(f))
                .value;
        }

        template <typename R, typename F>
        friend constexpr auto tag_fallback_invoke(fold_left_first_t, R&& r, F f)
        {
            return hpx::ranges::detail::fold_left_first_with_iter_impl(
                hpx::util::begin(r), hpx::util::end(r), std::move(f))
                .value;
        }
    };

    inline constexpr fold_left_first_t fold_left_first{};

}    // namespace hpx::ranges

#endif
