//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_CONFIG_CONSTEXPR_HPP
#define HPX_CONFIG_CONSTEXPR_HPP

#include <hpx/config/defines.hpp>

#if defined(HPX_HAVE_CXX11_CONSTEXPR) && !defined(HPX_MSVC_NVCC)
#   define HPX_CONSTEXPR constexpr
#   define HPX_CONSTEXPR_OR_CONST constexpr
#else
#   define HPX_CONSTEXPR
#   define HPX_CONSTEXPR_OR_CONST const
#endif

#ifdef HPX_HAVE_CXX14_CONSTEXPR
#   define HPX_CXX14_CONSTEXPR constexpr
#else
#   define HPX_CXX14_CONSTEXPR
#endif

#define HPX_STATIC_CONSTEXPR static HPX_CONSTEXPR_OR_CONST

#endif
