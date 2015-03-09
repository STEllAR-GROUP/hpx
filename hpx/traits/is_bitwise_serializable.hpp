//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_IS_BITWISE_SERIALIZABLE_HPP
#define HPX_TRAITS_IS_BITWISE_SERIALIZABLE_HPP

#include <boost/mpl/bool.hpp>
#include <boost/type_traits/is_arithmetic.hpp>

namespace hpx { namespace traits {
    template <typename T>
    struct is_bitwise_serializable
      : boost::is_arithmetic<T>
    {};
}}

#define HPX_IS_BITWISE_SERIALIZABLE(T)                                          \
namespace hpx { namespace traits {                                              \
    template <>                                                                 \
    struct is_bitwise_serializable<T>                                           \
      : boost::mpl::true_                                                       \
    {};                                                                         \
}}                                                                              \
/**/

#endif
