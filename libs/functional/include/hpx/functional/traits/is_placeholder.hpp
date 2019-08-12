//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// make inspect happy: hpxinspect:nodeprecatedname:boost::is_placeholder

#ifndef HPX_TRAITS_IS_PLACEHOLDER_HPP
#define HPX_TRAITS_IS_PLACEHOLDER_HPP

#include <hpx/config.hpp>

#include <boost/bind/arg.hpp>

#include <functional>
#include <type_traits>

namespace hpx { namespace traits {
    template <typename T>
    struct is_placeholder
      : std::integral_constant<int,
            std::is_placeholder<T>::value != 0 ?
                std::is_placeholder<T>::value :
                boost::is_placeholder<T>::value>
    {
    };

    template <typename T>
    struct is_placeholder<T const> : is_placeholder<T>
    {
    };
}}    // namespace hpx::traits

#endif /*HPX_TRAITS_IS_PLACEHOLDER_HPP*/
