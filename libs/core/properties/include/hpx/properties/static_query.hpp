// Copyright (c) 2021 Hartmut Kaiser
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/type_support/always_void.hpp>

#include <type_traits>

namespace hpx { namespace experimental {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        template <typename T, typename Property, typename = void>
        struct static_query : std::false_type
        {
        };

        template <typename T, typename Property>
        struct static_query<T, Property,
            util::always_void_t<decltype(Property::template static_query_v<T>)>>
          : std::true_type
        {
            using result_type = decltype(Property::template static_query_v<T>);

            static constexpr result_type property_value() noexcept(
                noexcept(Property::template static_query_v<T>))
            {
                return Property::template static_query_v<T>;
            }
        };

        template <typename T, typename Property, typename = void>
        struct static_query_value : std::false_type
        {
        };

        template <typename T, typename Property>
        struct static_query_value<T, Property,
            util::always_void_t<decltype(
                Property::template static_query_v<T> == Property::value())>>
          : std::integral_constant<bool,
                Property::template static_query_v<T> == Property::value()>
        {
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Property, typename Enable = void>
    struct static_query
      : detail::static_query<std::decay_t<T>, std::decay_t<Property>>
    {
    };

    template <typename T, typename Property>
    using static_query_t = typename static_query<T, Property>::type;

    template <typename T, typename Property>
    HPX_INLINE_CONSTEXPR_VARIABLE bool static_query_v =
        static_query<T, Property>::value;

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Property, typename Enable = void>
    struct static_query_value
      : detail::static_query_value<std::decay_t<T>, std::decay_t<Property>>
    {
    };

    template <typename T, typename Property>
    using static_query_value_t = typename static_query_value<T, Property>::type;

    template <typename T, typename Property>
    HPX_INLINE_CONSTEXPR_VARIABLE bool static_query_value_v =
        static_query_value<T, Property>::value;

}}    // namespace hpx::experimental
