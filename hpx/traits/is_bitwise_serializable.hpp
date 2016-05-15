//  Copyright (c) 2014 Thomas Heller
//
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
#ifdef HPX_HAVE_CXX11_STD_IS_TRIVIALLY_COPYABLE
      : std::is_trivially_copyable<T>
#else
      : std::is_arithmetic<T>
#endif
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
