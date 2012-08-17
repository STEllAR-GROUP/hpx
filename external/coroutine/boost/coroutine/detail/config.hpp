//  Copyright (c) 2007 Robert Perricone
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_COROUTINE_B9FD4B38_8972_488B_82BA_BA01C5FDA5FF
#define BOOST_COROUTINE_B9FD4B38_8972_488B_82BA_BA01C5FDA5FF

#if defined(BOOST_WINDOWS)
#  define BOOST_COROUTINE_SYMBOL_EXPORT      __declspec(dllexport)
#  define BOOST_COROUTINE_SYMBOL_IMPORT      __declspec(dllimport)
#elif defined(BOOST_COROUTINE_GCC_HAVE_VISIBILITY)
#  define BOOST_COROUTINE_SYMBOL_EXPORT      __attribute__((visibility("default")))
#  define BOOST_COROUTINE_SYMBOL_IMPORT      __attribute__((visibility("default")))
#else
#  define BOOST_COROUTINE_SYMBOL_EXPORT      /* empty */
#  define BOOST_COROUTINE_SYMBOL_IMPORT      /* empty */
#endif

#if defined(BOOST_COROUTINE_EXPORTS)
#  define BOOST_COROUTINE_EXPORT       BOOST_COROUTINE_SYMBOL_EXPORT
#else
#  define BOOST_COROUTINE_EXPORT       BOOST_COROUTINE_SYMBOL_IMPORT
#endif

#if !defined(BOOST_COROUTINE_NUM_HEAPS)
#  define BOOST_COROUTINE_NUM_HEAPS    7
#endif

#endif

