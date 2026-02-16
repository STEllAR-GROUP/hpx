//  Copyright (c) 2026 Arpit Khandelwal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/fold_right.hpp
/// \page hpx::ranges::fold_right
/// \headerfile hpx/algorithm.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx { namespace ranges {
    // clang-format off

    /// Implementation of the C++23 fold_right algorithm
    template <typename R, typename T, typename F>
    constexpr auto fold_right(R&& r, T init, F f);

    /// Implementation of the C++23 fold_right_last algorithm
    template <typename R, typename F>
    constexpr auto fold_right_last(R&& r, F f);

    // clang-format on
}}    // namespace hpx::ranges

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_foldable.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/parallel/algorithms/detail/advance_to_sentinel.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <algorithm>
#include <concepts>
#include <iterator>
#include <optional>
#include <type_traits>
#include <utility>

namespace hpx::ranges::detail {
    template <typename I, typename S, typename T, typename F>
    constexpr auto fold_right_impl(I first, S last, T init, F f)
    {
        using U = std::decay_t<T>;

        auto end_iter = hpx::parallel::detail::advance_to_sentinel(first, last);
        auto rfirst = std::make_reverse_iterator(end_iter);
        auto rlast = std::make_reverse_iterator(first);

        U accum = HPX_MOVE(init);
        for (; rfirst != rlast; ++rfirst)
        {
            accum = std::invoke(f, *rfirst, HPX_MOVE(accum));
        }

        return accum;
    }

    template <typename I, typename S, typename F>
    constexpr auto fold_right_last_impl(I first, S last, F f)
    {
        using U = typename std::iterator_traits<I>::value_type;

        if (first == last)
        {
            return std::optional<U>{std::nullopt};
        }

        auto end_iter = hpx::parallel::detail::advance_to_sentinel(first, last);
        auto rfirst = std::make_reverse_iterator(end_iter);
        auto rlast = std::make_reverse_iterator(first);

        std::optional<U> accum(std::in_place, *rfirst);
        for (++rfirst; rfirst != rlast; ++rfirst)
        {
            *accum = std::invoke(f, *rfirst, HPX_MOVE(*accum));
        }

        return accum;
    }
}    // namespace hpx::ranges::detail

namespace hpx::ranges {

    // ----------------------------------------------------------------------
    // fold_right
    // ----------------------------------------------------------------------
    HPX_CXX_CORE_EXPORT struct fold_right_t final
      : hpx::detail::tag_parallel_algorithm<fold_right_t>
    {
    private:
        template <typename I, typename S, typename T, typename F>
        friend constexpr auto tag_fallback_invoke(
            fold_right_t, I first, S last, T init, F f)
            requires hpx::is_indirectly_binary_right_foldable<F, T, I>
        {
            return hpx::ranges::detail::fold_right_impl(
                HPX_MOVE(first), HPX_MOVE(last), HPX_MOVE(init), HPX_MOVE(f));
        }

        template <typename R, typename T, typename F>
        friend constexpr auto tag_fallback_invoke(
            fold_right_t, R&& r, T init, F f)
            requires hpx::is_indirectly_binary_right_foldable<F, T,
                typename hpx::traits::range_traits<R>::iterator_type>
        {
            return hpx::ranges::detail::fold_right_impl(hpx::util::begin(r),
                hpx::util::end(r), HPX_MOVE(init), HPX_MOVE(f));
        }
    };

    inline constexpr fold_right_t fold_right{};

    // ----------------------------------------------------------------------
    // fold_right_last
    // ----------------------------------------------------------------------
    HPX_CXX_CORE_EXPORT struct fold_right_last_t final
      : hpx::detail::tag_parallel_algorithm<fold_right_last_t>
    {
    private:
        template <typename I, typename S, typename F>
        friend constexpr auto tag_fallback_invoke(
            fold_right_last_t, I first, S last, F f)
            requires hpx::is_indirectly_binary_right_foldable<F,
                typename std::iterator_traits<I>::value_type, I>
        {
            return hpx::ranges::detail::fold_right_last_impl(
                HPX_MOVE(first), HPX_MOVE(last), HPX_MOVE(f));
        }

        template <typename R, typename F>
        friend constexpr auto tag_fallback_invoke(fold_right_last_t, R&& r, F f)
            requires hpx::is_indirectly_binary_right_foldable<F,
                typename std::iterator_traits<typename hpx::traits::
                        range_traits<R>::iterator_type>::value_type,
                typename hpx::traits::range_traits<R>::iterator_type>
        {
            return hpx::ranges::detail::fold_right_last_impl(
                hpx::util::begin(r), hpx::util::end(r), HPX_MOVE(f));
        }
    };

    inline constexpr fold_right_last_t fold_right_last{};

}    // namespace hpx::ranges

#endif
