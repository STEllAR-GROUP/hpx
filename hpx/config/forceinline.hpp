//  Copyright (c) 2012-2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_CONFIG_FORCEINLINE_HPP
#define HPX_CONFIG_FORCEINLINE_HPP

#include <hpx/config/compiler_specific.hpp>

#if !defined(HPX_FORCEINLINE)
#   if defined(__CUDACC__)
#       define HPX_FORCEINLINE inline
#   elif defined(HPX_MSVC)
#       define HPX_FORCEINLINE __forceinline
#   elif defined(__GNUC__)
#       define HPX_FORCEINLINE inline __attribute__ ((__always_inline__))
#   else
#       define HPX_FORCEINLINE inline
#   endif
#endif


#endif
