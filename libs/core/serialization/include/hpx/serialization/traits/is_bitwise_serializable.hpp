//  Copyright (c) 2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <type_traits>

namespace hpx { namespace traits {

    template <typename T>
    struct is_bitwise_serializable : std::is_arithmetic<T>
    {
    };
}}    // namespace hpx::traits

#define HPX_IS_BITWISE_SERIALIZABLE(T)                                         \
    namespace hpx { namespace traits {                                         \
            template <>                                                        \
            struct is_bitwise_serializable<T> : std::true_type                 \
            {                                                                  \
            };                                                                 \
        }                                                                      \
    }                                                                          \
    /**/
