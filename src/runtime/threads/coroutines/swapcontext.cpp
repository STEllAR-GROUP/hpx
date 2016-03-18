//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if !defined(HPX_HAVE_GENERIC_CONTEXT_COROUTINES)

#if (defined(__linux) || defined(linux) || defined(__linux__)) \
 && !defined(__bgq__) && !defined(__powerpc__)

#if defined(__x86_64__) || defined(__amd64__)
#include "swapcontext64.ipp"
#elif defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__)
#include "swapcontext32.ipp"
#else
#error Unsupported platform
#endif

#endif

#endif
