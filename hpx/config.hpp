//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_CONFIG_MAR_24_2008_0943AM)
#define HPX_CONFIG_MAR_24_2008_0943AM

#include <boost/config.hpp>
#include <string>
#include <hpx/config/export_definitions.hpp>

///////////////////////////////////////////////////////////////////////////////
// Make sure DEBUG macro is defined consistently across platforms
#if defined(_DEBUG) && !defined(DEBUG)
#define DEBUG
#endif

///////////////////////////////////////////////////////////////////////////////
/// This is the default ip/port number used by the global address resolver
#define HPX_PORT                    7910
#define HPX_NAME_RESOLVER_ADDRESS   "127.0.0.1"
#define HPX_NAME_RESOLVER_PORT      7911
#define HPX_RANDOM_PORT_MIN         26001
#define HPX_RANDOM_PORT_MAX         26132

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
/// This defines the maximum number of possible runtime instances in one 
/// executable
#if !defined(HPX_RUNTIME_INSTANCE_LIMIT)
#  define HPX_RUNTIME_INSTANCE_LIMIT 2
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the maximum number of arguments a component action can take
#if !defined(HPX_ACTION_ARGUMENT_LIMIT)
#  define HPX_ACTION_ARGUMENT_LIMIT 7
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the maximum number of arguments a component action can take
#if !defined(HPX_WAIT_ARGUMENT_LIMIT)
#  define HPX_WAIT_ARGUMENT_LIMIT 10
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the maximum number of arguments a component constructor can 
/// take
#if !defined(HPX_COMPONENT_CREATE_ARG_MAX)
#  define HPX_COMPONENT_CREATE_ARG_MAX 3
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
/// This defines the initial global reference count associated with any created 
/// object.
#if !defined(HPX_INITIAL_GLOBALCREDIT)
#  define HPX_INITIAL_GLOBALCREDIT 255
#endif

///////////////////////////////////////////////////////////////////////////////
// This defines the default installation location, should be set by build system
#if !defined(HPX_PREFIX)
#  define HPX_PREFIX "."
#endif

///////////////////////////////////////////////////////////////////////////////
// This defines the maximum number of connect retries to the AGAS service 
// allowing for some leeway during startup of the localities 
#if !defined(HPX_MAX_NETWORK_RETRIES)
#define HPX_MAX_NETWORK_RETRIES 100
#endif

#if !defined(HPX_NETWORK_RETRIES_SLEEP)
#define HPX_NETWORK_RETRIES_SLEEP 100
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
#if !defined(HPX_LIBRARY_STRING)
#define HPX_LIBRARY_STRING                                                    \
        BOOST_PP_STRINGIZE(HPX_MANGLE_NAME(HPX_COMPONENT_NAME))               \
        HPX_SHARED_LIB_EXTENSION                                              \
    /**/
#endif

#if !defined(HPX_APPLICATION_STRING) && defined(HPX_APPLICATION_NAME)
#define HPX_APPLICATION_STRING BOOST_PP_STRINGIZE(HPX_APPLICATION_NAME)
#endif

///////////////////////////////////////////////////////////////////////////////
#if defined(BOOST_WINDOWS)
#define snprintf _snprintf
#if !defined(HPX_EMULATE_SWAP_CONTEXT)
#define HPX_EMULATE_SWAP_CONTEXT 0
#endif
#endif

///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_USE_ITT)
#define HPX_USE_ITT 0
#endif

///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_IDLE_LOOP_COUNT_MAX)
#define HPX_IDLE_LOOP_COUNT_MAX 20000
#endif

///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_LOCK_LIMIT)
#define HPX_LOCK_LIMIT 28
#endif

///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_USE_ABP_SCHEDULER)
#define HPX_USE_ABP_SCHEDULER 0
#endif

///////////////////////////////////////////////////////////////////////////////
#include <hpx/config/defaults.hpp>

#endif


