//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_CONFIG_MAR_24_2008_0943AM)
#define HPX_CONFIG_MAR_24_2008_0943AM

#include <string>
#include <hpx/config/export_definitions.hpp>

///////////////////////////////////////////////////////////////////////////////
/// This is the default ip/port number used by the global address resolver
#define HPX_PORT 7910
#define HPX_NAME_RESOLVER_ADDRESS   "127.0.0.1"
#define HPX_NAME_RESOLVER_PORT      7911

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
/// This defines the maximum number of arguments a component action can take
#if !defined(HPX_ACTION_ARGUMENT_LIMIT)
#  define HPX_ACTION_ARGUMENT_LIMIT 5
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the maximum number of arguments a plain action can take
#if !defined(HPX_PLAIN_ACTION_ARGUMENT_LIMIT)
#  define HPX_PLAIN_ACTION_ARGUMENT_LIMIT 5
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the number of outgoing (parcel-) connections kept alive 
#if !defined(HPX_MAX_PARCEL_CONNECTION_CACHE_SIZE)
#  define HPX_MAX_PARCEL_CONNECTION_CACHE_SIZE 64
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the number of outgoing (agas-) connections kept alive 
/// This generally shouldn't be much larger than the number of OS threads used
/// in the runtime system.
#if !defined(HPX_MAX_AGAS_CONNECTION_CACHE_SIZE)
#  define HPX_MAX_AGAS_CONNECTION_CACHE_SIZE 8
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the number of AGAS address translations kept in the local 
/// cache. This is just the initial siye which maz be adjusted depending on the 
/// load of the system, etc.
#if !defined(HPX_INITIAL_AGAS_CACHE_SIZE)
#  define HPX_INITIAL_AGAS_CACHE_SIZE 128
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines whether to use the portable binary archives for parcel 
/// serialization
#if !defined(HPX_USE_PORTABLE_ARCHIVES)
#  define HPX_USE_PORTABLE_ARCHIVES 1
#endif

///////////////////////////////////////////////////////////////////////////////
// This defines the default installation location, should be set by build system
#if !defined(HPX_PREFIX)
#  define HPX_PREFIX "."
#endif

///////////////////////////////////////////////////////////////////////////////
// This defines the maximum number of connect retries to the AGAS service 
// allowing for some leeway during startup of the localities 
#if !defined(HPX_MAX_AGAS_RETRIES)
#define HPX_MAX_AGAS_RETRIES 10
#endif

#if !defined(HPX_AGAS_RETRIES_SLEEP)
#define HPX_AGAS_RETRIES_SLEEP 100    // [ms]
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
#  ifdef __APPLE__    // apple
#    define HPX_SHARED_LIB_EXTENSION        ".dylib"
#  else               // linux & co
#    define HPX_SHARED_LIB_EXTENSION        ".so"
#  endif
#endif

///////////////////////////////////////////////////////////////////////////////
// on Windows the debug versions of the libraries have mangled entry points 
#if !defined(BOOST_WINDOWS)
#  define HPX_MANGLE_COMPONENT_NAME_PREFIX  libhpx_component_
#  define HPX_MANGLE_COMPONENT_NAME(n)                                        \
    BOOST_PP_CAT(HPX_MANGLE_COMPONENT_NAME_PREFIX, n)                         \
    /**/
#  define HPX_MANGLE_COMPONENT_NAME_STR(n)                                    \
    BOOST_PP_STRINGIZE(HPX_MANGLE_COMPONENT_NAME_PREFIX) + n                  \
    /**/
#  define HPX_MANGLE_NAME(n)                BOOST_PP_CAT(lib, n)
#elif defined(_DEBUG)
#  define HPX_MANGLE_COMPONENT_NAME(n)      BOOST_PP_CAT(n, d)
#  define HPX_MANGLE_COMPONENT_NAME_STR(n)  n + "d"
#  define HPX_MANGLE_NAME(n)                BOOST_PP_CAT(n, d)
#else
#  define HPX_MANGLE_COMPONENT_NAME(n)      n
#  define HPX_MANGLE_COMPONENT_NAME_STR(n)  n
#  define HPX_MANGLE_NAME(n)                n
#endif

///////////////////////////////////////////////////////////////////////////////
#include <hpx/config/defaults.hpp>

#endif


