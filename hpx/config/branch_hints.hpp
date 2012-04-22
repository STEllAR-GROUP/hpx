////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (C) 2007, 2008 Tim Blechmann
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_50B9885A_AAD3_48C5_814A_EBCD47C858AC)
#define HPX_50B9885A_AAD3_48C5_814A_EBCD47C858AC

#if defined(__GNUC__)
  #define HPX_LIKELY(expr)    __builtin_expect(static_cast<long int>(expr), true)
  #define HPX_UNLIKELY(expr)  __builtin_expect(static_cast<long int>(expr), false)
#else
  #define HPX_LIKELY(expr)    expr
  #define HPX_UNLIKELY(expr)  expr
#endif

#endif // HPX_50B9885A_AAD3_48C5_814A_EBCD47C858AC

