//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_IS_BIND_EXPRESSION_HPP
#define HPX_TRAITS_IS_BIND_EXPRESSION_HPP

#include <hpx/config.hpp>

#include <functional>

namespace hpx { namespace traits {
    template <typename T>
    struct is_bind_expression : std::is_bind_expression<T>
    {
    };

    template <typename T>
    struct is_bind_expression<T const> : is_bind_expression<T>
    {
    };
}}    // namespace hpx::traits

#endif /*HPX_TRAITS_IS_BIND_EXPRESSION_HPP*/
