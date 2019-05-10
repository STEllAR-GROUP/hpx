//  Copyright (c) 2013-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_IS_BIND_EXPRESSION_HPP
#define HPX_TRAITS_IS_BIND_EXPRESSION_HPP

#include <hpx/config.hpp>

#ifdef HPX_HAVE_CXX11_STD_IS_BIND_EXPRESSION
#include <functional>
#endif
#include <type_traits>

namespace hpx { namespace traits
{
    template <typename T>
    struct is_bind_expression
#ifdef HPX_HAVE_CXX11_STD_IS_BIND_EXPRESSION
      : std::is_bind_expression<T>
#else
      : std::false_type
#endif
    {};

    template <typename T>
    struct is_bind_expression<T const>
      : is_bind_expression<T>
    {};
}}

#endif /*HPX_TRAITS_IS_BIND_EXPRESSION_HPP*/
