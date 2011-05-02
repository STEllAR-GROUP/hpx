////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(SHEOL_8877B5FB_1967_43B9_AF98_1A01F162B725)
#define SHEOL_8877B5FB_1967_43B9_AF98_1A01F162B725

#if defined(__GNUC__)
  #define SHEOL_SUPER_PURE  __attribute__((const)) 
  #define SHEOL_PURE        __attribute__((pure)) 
  #define SHEOL_HOT         __attribute__((hot)) 
  #define SHEOL_COLD        __attribute__((cold)) 
#else
  #define SHEOL_SUPER_PURE
  #define SHEOL_PURE
  #define SHEOL_HOT
  #define SHEOL_COLD
#endif

#endif // SHEOL_8877B5FB_1967_43B9_AF98_1A01F162B725

