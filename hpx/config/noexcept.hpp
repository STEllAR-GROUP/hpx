//  Copyright (c) 2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_CONFIG_NOEXCEPT_HPP
#define HPX_CONFIG_NOEXCEPT_HPP

#include <hpx/config/defines.hpp>

#ifdef HPX_HAVE_CXX11_NOEXCEPT
#   define HPX_NOEXCEPT noexcept
#   define HPX_NOEXCEPT_OR_NOTHROW noexcept
#   define HPX_NOEXCEPT_IF(Predicate) noexcept((Predicate))
#   define HPX_NOEXCEPT_EXPR(Expression) noexcept((Expression))
#else
#   define HPX_NOEXCEPT
#   define HPX_NOEXCEPT_OR_NOTHROW throw()
#   define HPX_NOEXCEPT_IF(Predicate)
#   define HPX_NOEXCEPT_EXPR(Expression) false
#endif

#endif
