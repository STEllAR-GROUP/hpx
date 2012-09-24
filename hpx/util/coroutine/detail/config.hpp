//  Copyright (c) 2007 Robert Perricone
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COROUTINE_B9FD4B38_8972_488B_82BA_BA01C5FDA5FF
#define HPX_COROUTINE_B9FD4B38_8972_488B_82BA_BA01C5FDA5FF

#if defined(BOOST_WINDOWS)
#  define HPX_COROUTINE_SYMBOL_EXPORT      __declspec(dllexport)
#  define HPX_COROUTINE_SYMBOL_IMPORT      __declspec(dllimport)
#elif defined(HPX_COROUTINE_GCC_HAVE_VISIBILITY)
#  define HPX_COROUTINE_SYMBOL_EXPORT      __attribute__((visibility("default")))
#  define HPX_COROUTINE_SYMBOL_IMPORT      __attribute__((visibility("default")))
#else
#  define HPX_COROUTINE_SYMBOL_EXPORT      /* empty */
#  define HPX_COROUTINE_SYMBOL_IMPORT      /* empty */
#endif

#if defined(HPX_COROUTINE_EXPORTS)
#  define HPX_COROUTINE_EXPORT       HPX_COROUTINE_SYMBOL_EXPORT
#else
#  define HPX_COROUTINE_EXPORT       HPX_COROUTINE_SYMBOL_IMPORT
#endif

#if !defined(HPX_COROUTINE_NUM_HEAPS)
#  define HPX_COROUTINE_NUM_HEAPS    7
#endif

#endif

