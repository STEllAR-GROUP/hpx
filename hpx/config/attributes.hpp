//  Copyright (c) 2017 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_CONFIG_ATTRIBUTES_HPP
#define HPX_CONFIG_ATTRIBUTES_HPP

#include <hpx/config/defines.hpp>

// handle [[fallthrough]]
#if defined(HPX_HAVE_CXX17_FALLTHROUGH_ATTRIBUTE)
#   define HPX_FALLTHROUGH [[fallthrough]]
#else
#   define HPX_FALLTHROUGH
#endif

// handle [[deprecated]]
#if defined(HPX_HAVE_DEPRECATION_WARNINGS)
#  define HPX_DEPRECATED_MSG \
   "This functionality is deprecated and will be removed in the future."
#  if defined(HPX_HAVE_CXX14_DEPRECATED_ATTRIBUTE)
#    define HPX_DEPRECATED(x) [[deprecated(x)]]
#  elif defined(HPX_MSVC)
#    define HPX_DEPRECATED(x) __declspec(deprecated(x))
#  elif defined(__GNUC__)
#    define HPX_DEPRECATED(x) __attribute__((__deprecated__(x)))
#  endif
#endif

#if !defined(HPX_DEPRECATED)
#  define HPX_DEPRECATED(x)  /**/
#endif

#endif
