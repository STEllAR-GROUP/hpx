//  Copyright (c) 2026 Hartmut Kaiser
//  Copyright (c) 2026 Mamidi Surya Teja
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/functional/traits/is_invocable.hpp>
#include <hpx/iterator_support/boost_iterator_categories.hpp>
#include <hpx/modules/type_support.hpp>

#include <concepts>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_CORE_EXPORT template <typename F, typename T, typename I,
        typename U>
    concept is_indirect_binary_left_foldable_impl =
        std::movable<T> && std::movable<U> && std::convertible_to<T, U> &&
        std::invocable<F&, U, std::iter_reference_t<I>> &&
        std::assignable_from<U&,
            std::invoke_result_t<F&, U, std::iter_reference_t<I>>>;

    HPX_CXX_CORE_EXPORT template <typename F, typename T, typename I>
    concept is_indirectly_binary_left_foldable =
        std::copy_constructible<F> && std::indirectly_readable<I> &&
        std::invocable<F&, T, std::iter_reference_t<I>> &&
        std::convertible_to<
            std::invoke_result_t<F&, T, std::iter_reference_t<I>>,
            std::decay_t<
                std::invoke_result_t<F&, T, std::iter_reference_t<I>>>> &&
        is_indirect_binary_left_foldable_impl<F, T, I,
            std::decay_t<
                std::invoke_result_t<F&, T, std::iter_reference_t<I>>>>;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename F>
        class flipped
        {
            F f;

        public:
            template <typename T, typename U>
                requires hpx::is_invocable_v<F&, U&&, T&&>
            constexpr hpx::util::invoke_result_t<F&, U&&, T&&> operator()(
                T&& t, U&& u)
            {
                return HPX_INVOKE(f, HPX_FORWARD(U, u), HPX_FORWARD(T, t));
            }
        };
    }    // namespace detail

    HPX_CXX_CORE_EXPORT template <typename F, typename T, typename I>
    concept is_indirectly_binary_right_foldable =
        is_indirectly_binary_left_foldable<detail::flipped<F>, T, I>;

}    // namespace hpx
