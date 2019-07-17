//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_LAZY_CONDITIONAL_HPP
#define HPX_UTIL_LAZY_CONDITIONAL_HPP

#include <type_traits>

namespace hpx { namespace util
{
    template <bool Enable, typename C1, typename C2>
    struct lazy_conditional
      : std::conditional<Enable, C1, C2>::type
    {
    };
}}

#endif
