//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2003-2007 Jonathan Turkanis

// See http://www.boost.org/libs/iostreams for documentation.

#pragma once

#include <hpx/config.hpp>

#include <type_traits>

namespace hpx::iostream {

    HPX_CXX_CORE_EXPORT template <typename T>
    struct operations;

    namespace detail {

        HPX_CXX_CORE_EXPORT struct custom_tag
        {
        };

        HPX_CXX_CORE_EXPORT template <typename T>
        struct is_custom
          : std::negation<std::is_base_of<custom_tag, operations<T>>>
        {
        };

        HPX_CXX_CORE_EXPORT template <typename T>
        inline constexpr bool is_custom_v = is_custom<T>::value;
    }    // namespace detail

    HPX_CXX_CORE_EXPORT template <typename T>
    struct operations : detail::custom_tag
    {
    };
}    // namespace hpx::iostream
