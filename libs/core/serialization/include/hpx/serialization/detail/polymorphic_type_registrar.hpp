//  Copyright (c) 2026 Ujjwal Shekhar
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/serialization/macros.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

namespace hpx::serialization::detail {

    template <typename... Ts>
    struct register_types;

    template <>
    struct register_types<>
    {
        static void ensure_instantiated() {}
    };

    template <typename First, typename... Rest>
    struct register_types<First, Rest...>
    {
        // Force instantiation
        static inline auto& instance =
            hpx::experimental::serialization::register_class<First>::instance();

        static void ensure_instantiated()
        {
            (void) instance;    // Force instantiation, prevent linker strip
            register_types<Rest...>::ensure_instantiated();
        }
    };

    // THE USER API: A static object that triggers the chain
    template <typename... Ts>
    struct enable_user_types
    {
        enable_user_types()
        {
            // Force instantiation of the registration chain
            register_types<Ts...>::ensure_instantiated();
        }
    };

}    // namespace hpx::serialization::detail