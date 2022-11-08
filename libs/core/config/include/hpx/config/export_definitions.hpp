//  Copyright (c) 2007-2019 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/defines.hpp>

#if defined(DOXYGEN)
/// Marks a class or function to be exported from HPX or imported if it is
/// consumed.
#define HPX_EXPORT
#else

// clang-format off
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#if !defined(HPX_MODULE_STATIC_LINKING) && !defined(HPX_HAVE_STATIC_LINKING)
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
#if defined(HPX_CORE_EXPORTS)
# define  HPX_CORE_EXPORT        HPX_SYMBOL_EXPORT
#else
# define  HPX_CORE_EXPORT        HPX_SYMBOL_IMPORT
#endif

///////////////////////////////////////////////////////////////////////////////
#if defined(HPX_EXPORTS) || defined(HPX_FULL_EXPORTS)
# define  HPX_EXPORT             HPX_SYMBOL_EXPORT
#else
# define  HPX_EXPORT             HPX_SYMBOL_IMPORT
#endif

///////////////////////////////////////////////////////////////////////////////
// define the export/import helper macros to be used for component modules
#if defined(HPX_COMPONENT_EXPORTS)
# define  HPX_COMPONENT_EXPORT   HPX_SYMBOL_EXPORT
#else
# define  HPX_COMPONENT_EXPORT   HPX_SYMBOL_IMPORT
#endif

///////////////////////////////////////////////////////////////////////////////
// define the export/import helper macros to be used for component modules
#if defined(HPX_LIBRARY_EXPORTS)
# define  HPX_LIBRARY_EXPORT     HPX_SYMBOL_EXPORT
#else
# define  HPX_LIBRARY_EXPORT     HPX_SYMBOL_IMPORT
#endif

///////////////////////////////////////////////////////////////////////////////
// helper macro for symbols which have to be exported from the runtime and all
// components
#if defined(HPX_CORE_EXPORTS) || \
    defined(HPX_FULL_EXPORTS) || defined(HPX_EXPORTS) || \
    defined(HPX_COMPONENT_EXPORTS) || defined(HPX_APPLICATION_EXPORTS) || \
    defined(HPX_LIBRARY_EXPORTS)
# define HPX_ALWAYS_EXPORT       HPX_SYMBOL_EXPORT
# define HPX_ALWAYS_IMPORT       HPX_SYMBOL_IMPORT
#else
# define HPX_ALWAYS_EXPORT       HPX_SYMBOL_IMPORT
# define HPX_ALWAYS_IMPORT       HPX_SYMBOL_IMPORT
#endif
#endif
// clang-format on
