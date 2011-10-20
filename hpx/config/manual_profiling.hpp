////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_8877B5FB_1967_43B9_AF98_1A01F162B725)
#define HPX_8877B5FB_1967_43B9_AF98_1A01F162B725

#if defined(__GNUC__)
  #define HPX_SUPER_PURE  __attribute__((const))
  #define HPX_PURE        __attribute__((pure))
  #define HPX_HOT         __attribute__((hot))
  #define HPX_COLD        __attribute__((cold))
#else
  #define HPX_SUPER_PURE
  #define HPX_PURE
  #define HPX_HOT
  #define HPX_COLD
#endif

#endif // HPX_8877B5FB_1967_43B9_AF98_1A01F162B725

