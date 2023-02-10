//  Copyright (c) 2019 Jan Melech
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx::traits {

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        struct wildcard
        {
            // Excluded hpx::util::unused_type from wildcard conversions
            // due to ambiguity (unused_type has own conversion to every type).
            template <typename T,
                typename Enable = std::enable_if_t<
                    !std::is_lvalue_reference_v<T> &&
                    !std::is_same_v<std::decay_t<T>, hpx::util::unused_type>>>
            operator T&&() const;

            template <typename T,
                typename Enable = std::enable_if_t<
                    std::is_copy_constructible_v<T> &&
                    !std::is_same_v<std::decay_t<T>, hpx::util::unused_type>>>
            operator T&() const;
        };

        template <std::size_t N = 0>
        static constexpr const wildcard _wildcard{};

        ///////////////////////////////////////////////////////////////////////
        template <typename T, std::size_t... I>
        constexpr auto is_brace_constructible(std::index_sequence<I...>,
            T*) noexcept -> decltype(T{_wildcard<I>...}, std::true_type{})
        {
            return {};
        }

        template <std::size_t... I>
        constexpr std::false_type is_brace_constructible(
            std::index_sequence<I...>, ...) noexcept
        {
            return {};
        }
    }    // namespace detail

    template <typename T, std::size_t N>
    constexpr auto is_brace_constructible() noexcept
        -> decltype(detail::is_brace_constructible(
            std::make_index_sequence<N>{}, static_cast<T*>(nullptr)))
    {
        return {};
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        template <typename T, typename U>
        struct is_paren_constructible;

        template <typename T, std::size_t... I>
        struct is_paren_constructible<T, std::index_sequence<I...>>
          : std::is_constructible<T, decltype(_wildcard<I>)...>
        {
        };
    }    // namespace detail

    template <typename T, std::size_t N>
    constexpr auto is_paren_constructible() noexcept
        -> detail::is_paren_constructible<T, std::make_index_sequence<N>>
    {
        return {};
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        template <std::size_t N>
        using size = std::integral_constant<std::size_t, N>;

        template <typename T,
            typename Enable =
                std::enable_if_t<std::is_class_v<T> && std::is_empty_v<T>>>
        constexpr size<0> arity() noexcept
        {
            return {};
        }

#if !defined(HPX_HAVE_CXX20_PAREN_INITIALIZATION_OF_AGGREGATES)
#define MAKE_ARITY_FUNC(count)                                                 \
    template <typename T,                                                      \
        typename Enable =                                                      \
            std::enable_if_t<traits::is_brace_constructible<T, count>() &&     \
                !traits::is_brace_constructible<T, count + 1>() &&             \
                !traits::is_paren_constructible<T, count>()>>                  \
    constexpr size<count> arity() noexcept                                     \
    {                                                                          \
        return {};                                                             \
    }
#else
#define MAKE_ARITY_FUNC(count)                                                 \
    template <typename T,                                                      \
        typename Enable =                                                      \
            std::enable_if_t<traits::is_brace_constructible<T, count>() &&     \
                !traits::is_brace_constructible<T, count + 1>() &&             \
                traits::is_paren_constructible<T, count>()>>                   \
    constexpr size<count> arity() noexcept                                     \
    {                                                                          \
        return {};                                                             \
    }
#endif

        MAKE_ARITY_FUNC(1)
        MAKE_ARITY_FUNC(2)
        MAKE_ARITY_FUNC(3)
        MAKE_ARITY_FUNC(4)
        MAKE_ARITY_FUNC(5)
        MAKE_ARITY_FUNC(6)
        MAKE_ARITY_FUNC(7)
        MAKE_ARITY_FUNC(8)
        MAKE_ARITY_FUNC(9)
        MAKE_ARITY_FUNC(10)
        MAKE_ARITY_FUNC(11)
        MAKE_ARITY_FUNC(12)
        MAKE_ARITY_FUNC(13)
        MAKE_ARITY_FUNC(14)
        MAKE_ARITY_FUNC(15)

#undef MAKE_ARITY_FUNC
    }    // namespace detail
}    // namespace hpx::traits
