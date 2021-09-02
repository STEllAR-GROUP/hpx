//  Copyright (c) 2007-2019 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config/defines.hpp>

#if defined(DOXYGEN)
/// Marks a class or function to be exported from HPXLocal or imported if it is
/// consumed.
#define HPX_LOCAL_EXPORT
#else

// clang-format off
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#if !defined(HPX_MODULE_STATIC_LINKING)
# define HPX_SYMBOL_EXPORT      __declspec(dllexport)
# define HPX_SYMBOL_IMPORT      __declspec(dllimport)
# define HPX_SYMBOL_INTERNAL    /* empty */
#endif
#elif defined(__NVCC__) || defined(__CUDACC__)
# define HPX_SYMBOL_EXPORT      /* empty */
# define HPX_SYMBOL_IMPORT      /* empty */
# define HPX_SYMBOL_INTERNAL    /* empty */
#elif defined(HPX_HAVE_ELF_HIDDEN_VISIBILITY)
# define HPX_SYMBOL_EXPORT      __attribute__((visibility("default")))
# define HPX_SYMBOL_IMPORT      __attribute__((visibility("default")))
# define HPX_SYMBOL_INTERNAL    __attribute__((visibility("hidden")))
#endif

// make sure we have reasonable defaults
#if !defined(HPX_SYMBOL_EXPORT)
# define HPX_SYMBOL_EXPORT      /* empty */
#endif
#if !defined(HPX_SYMBOL_IMPORT)
# define HPX_SYMBOL_IMPORT      /* empty */
#endif
#if !defined(HPX_SYMBOL_INTERNAL)
# define HPX_SYMBOL_INTERNAL    /* empty */
#endif

///////////////////////////////////////////////////////////////////////////////
#if defined(HPX_LOCAL_EXPORTS)
# define  HPX_LOCAL_EXPORT        HPX_SYMBOL_EXPORT
#else
# define  HPX_LOCAL_EXPORT        HPX_SYMBOL_IMPORT
#endif

///////////////////////////////////////////////////////////////////////////////
// helper macro for symbols which have to be exported from the runtime and all
// components
# define HPX_ALWAYS_EXPORT       HPX_SYMBOL_EXPORT
# define HPX_ALWAYS_IMPORT       HPX_SYMBOL_IMPORT
// clang-format on
#endif
