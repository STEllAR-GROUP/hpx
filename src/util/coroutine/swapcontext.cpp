//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#if defined(__GNUC__)

#if defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__)
#include "swapcontext32.ipp"
#elif defined(__x86_64__) || defined(__amd64__)
#include "swapcontext64.ipp"
#else
#error Unsupported platform 
#endif

#endif
