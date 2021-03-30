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

    namespace detail {

        template <typename T, typename Property, typename = void>
        struct is_applicable_property : std::false_type
        {
        };

        template <typename T, typename Property>
        struct is_applicable_property<T, Property,
            util::always_void_t<std::enable_if_t<
                Property::template is_applicable_property_v<T>>>>
          : std::true_type
        {
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Property, typename Enable = void>
    struct is_applicable_property
      : detail::is_applicable_property<std::decay_t<T>, std::decay_t<Property>>
    {
    };

    template <typename T, typename Property>
    HPX_INLINE_CONSTEXPR_VARIABLE bool is_applicable_property_v =
        is_applicable_property<T, Property>::value;

}}    // namespace hpx::experimental
