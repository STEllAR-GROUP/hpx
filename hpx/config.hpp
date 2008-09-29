//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_CONFIG_MAR_24_2008_0943AM)
#define HPX_CONFIG_MAR_24_2008_0943AM

#include <hpx/config/export_definitions.hpp>

///////////////////////////////////////////////////////////////////////////////
/// This is the default ip/port number used by the global address resolver
#define HPX_PORT 7810
#define HPX_NAME_RESOLVER_PORT 7911

///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_DEBUG) && defined(_DEBUG)
#  define HPX_DEBUG 1
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines if the Intel Thread Building Blocks library will be used
#if !defined(HPX_USE_TBB)
#  define HPX_USE_TBB 0
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the maximum number of constructor arguments a component 
/// can take
#if !defined(HPX_COMPONENT_ARGUMENT_LIMIT)
#  define HPX_COMPONENT_ARGUMENT_LIMIT 4
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the maximum number of arguments an action can take
#if !defined(HPX_ACTION_ARGUMENT_LIMIT)
#  define HPX_ACTION_ARGUMENT_LIMIT 4
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the number kept alive of outgoing (parcel-) connections
#if !defined(HPX_MAX_CONNECTION_CACHE_SIZE)
#  define HPX_MAX_CONNECTION_CACHE_SIZE 64
#endif

///////////////////////////////////////////////////////////////////////////////
// This defines the default installation location, should be set by build system
#if !defined(HPX_PREFIX)
#  define HPX_PREFIX "."
#endif

///////////////////////////////////////////////////////////////////////////////
//  Characters used 
//    - to delimit several HPX ini paths
//    - used as file extensions for shared libraries
//    - used as path delimiters
#ifdef BOOST_WINDOWS  // windows
#  define HPX_INI_PATH_DELIMITER            ";"
#  define HPX_SHARED_LIB_EXTENSION          ".dll"
#  define HPX_PATH_DELIMITERS               "\\/"
#else                 // unix like
#  define HPX_INI_PATH_DELIMITER            ":"
#  define HPX_PATH_DELIMITERS               "/"
#  ifdef _APPLE       // apple
#    define HPX_SHARED_LIB_EXTENSION        ".dylib"
#  else               // linux & co
#    define HPX_SHARED_LIB_EXTENSION        ".so"
#  endif
#endif

///////////////////////////////////////////////////////////////////////////////
// on Windows the debug versions of the libraries have mangled entry points 
#if !defined(BOOST_WINDOWS)
#  define HPX_MANGLE_COMPONENT_NAME(n)      BOOST_PP_CAT(libhpx_component_, n)
#  define HPX_MANGLE_COMPONENT_NAME_STR(n)  "libhpx_component_" + n
#elif defined(_DEBUG)
#  define HPX_MANGLE_COMPONENT_NAME(n)      BOOST_PP_CAT(n, d)
#  define HPX_MANGLE_COMPONENT_NAME_STR(n)  n + "d"
#else
#  define HPX_MANGLE_COMPONENT_NAME(n)      n
#  define HPX_MANGLE_COMPONENT_NAME_STR(n)  n
#endif

///////////////////////////////////////////////////////////////////////////////
#include <hpx/config/defaults.hpp>

#endif


