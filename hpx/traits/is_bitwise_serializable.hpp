//  Copyright (c) 2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_IS_BITWISE_SERIALIZABLE_HPP
#define HPX_TRAITS_IS_BITWISE_SERIALIZABLE_HPP

#include <hpx/config.hpp>

#include <type_traits>

namespace hpx { namespace traits
{
    template <typename T>
    struct is_bitwise_serializable
      : std::is_arithmetic<T>
    {};
}}

#define HPX_IS_BITWISE_SERIALIZABLE(T)                                        \
namespace hpx { namespace traits {                                            \
    template <>                                                               \
    struct is_bitwise_serializable< T >                                       \
      : std::true_type                                                        \
    {};                                                                       \
}}                                                                            \
/**/

#endif /*HPX_TRAITS_IS_BITWISE_SERIALIZABLE_HPP*/
