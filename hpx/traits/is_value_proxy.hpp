///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_TRAITS_IS_VALUE_PROXY_HPP
#define HPX_TRAITS_IS_VALUE_PROXY_HPP

#include <type_traits>

namespace hpx { namespace traits {
    template <typename T>
    struct is_value_proxy
      : std::false_type
    {};
}}

#endif
