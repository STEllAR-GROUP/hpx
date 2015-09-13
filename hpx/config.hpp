//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_CONFIG_MAR_24_2008_0943AM)
#define HPX_CONFIG_MAR_24_2008_0943AM

// We need to detect if user code include boost/config.hpp before
// including hpx/config.hpp
// Everything else might lead to hard compile errors and possible very subtile bugs.
#if defined(BOOST_CONFIG_HPP)
#error Boost.Config was included before the hpx config header. This might lead to subtile failures and compile errors. Please include <hpx/config.hpp> before any other boost header
#endif

#include <hpx/config/defines.hpp>
#include <hpx/config/version.hpp>
#include <hpx/config/compiler_specific.hpp>
#include <hpx/config/branch_hints.hpp>
#include <hpx/config/manual_profiling.hpp>
#include <hpx/config/forceinline.hpp>
#include <hpx/config/constexpr.hpp>

#include <boost/version.hpp>

#if BOOST_VERSION == 105400
#include <cstdint> // Boost.Atomic has trouble finding [u]intptr_t
#endif

#if BOOST_VERSION < 105600
#include <boost/exception/detail/attribute_noreturn.hpp>
#endif

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>

#if defined(_MSC_VER)
// On Windows, make sure winsock.h is not included even if windows.h is
// included before winsock2.h
#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#endif

///////////////////////////////////////////////////////////////////////////////
// Make sure DEBUG macro is defined consistently across platforms
#if defined(_DEBUG) && !defined(DEBUG)
#  define DEBUG
#endif

///////////////////////////////////////////////////////////////////////////////
#if defined(DEBUG) && !defined(HPX_DEBUG)
#  define HPX_DEBUG
#endif

#if defined(HPX_DEBUG)
#  define HPX_BUILD_TYPE debug
#else
#  define HPX_BUILD_TYPE release
#endif

///////////////////////////////////////////////////////////////////////////////
/// This is the default ip/port number used by the parcel subsystem
#define HPX_INITIAL_IP_PORT         7910
#define HPX_CONNECTING_IP_PORT      7909
#define HPX_INITIAL_IP_ADDRESS      "127.0.0.1"

///////////////////////////////////////////////////////////////////////////////
/// This defines the maximum number of possible runtime instances in one
/// executable
#if !defined(HPX_RUNTIME_INSTANCE_LIMIT)
#  define HPX_RUNTIME_INSTANCE_LIMIT 1
#endif

///////////////////////////////////////////////////////////////////////////////
// Properly handle all preprocessing limits
#if !defined(HPX_LIMIT)
#  define HPX_LIMIT 5
#elif (HPX_LIMIT < 5)
#  error "HPX_LIMIT is too low, it must be at least 5"
#endif

// make sure Fusion sizes are adjusted appropriately as well
#if HPX_LIMIT > 5 && !defined(FUSION_MAX_VECTOR_SIZE)
#  define FUSION_MAX_VECTOR_SIZE 20
#endif

// make sure boost::result_of is adjusted appropriately as well
#if HPX_LIMIT > 5 && !defined(BOOST_RESULT_OF_NUM_ARGS)
#  define BOOST_RESULT_OF_NUM_ARGS 20
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the type of the parcelport to be used during application
/// bootstrap. This value can be changed at runtime by the configuration
/// parameter:
///
///   hpx.parcel.bootstrap = ...
///
/// (or by setting the corresponding environment variable HPX_PARCEL_BOOTSTRAP).
#if !defined(HPX_PARCEL_BOOTSTRAP)
#  define HPX_PARCEL_BOOTSTRAP "tcp"
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the number of outgoing (parcel-) connections kept alive (to
/// all other localities). This value can be changed at runtime by setting
/// the configuration parameter:
///
///   hpx.parcel.max_connections = ...
///
/// (or by setting the corresponding environment variable
/// HPX_PARCEL_MAX_CONNECTIONS).
#if !defined(HPX_PARCEL_MAX_CONNECTIONS)
#  define HPX_PARCEL_MAX_CONNECTIONS 512
#endif

/// This defines the number of outgoing ipc (parcel-) connections kept alive
/// (to each of the other localities on the same node). This value can be changed
/// at runtime by setting the configuration parameter:
///
///   hpx.parcel.ipc.data_buffer_cache_size = ...
///
/// (or by setting the corresponding environment variable
/// HPX_PARCEL_IPC_DATA_BUFFER_CACHE_SIZE).
#if !defined(HPX_PARCEL_IPC_DATA_BUFFER_CACHE_SIZE)
#  define HPX_PARCEL_IPC_DATA_BUFFER_CACHE_SIZE 512
#endif

/// This defines the number of MPI requests in flight
/// This value can be changed at runtime by setting the configuration parameter:
///
///   hpx.parcel.mpi.max_requests = ...
///
/// (or by setting the corresponding environment variable
/// HPX_PARCEL_MPI_MAX_REQUESTS).
#if !defined(HPX_PARCEL_MPI_MAX_REQUESTS)
#  define HPX_PARCEL_MPI_MAX_REQUESTS 2147483647
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the number of outgoing (parcel-) connections kept alive (to
/// each of the other localities). This value can be changed at runtime by
/// setting the configuration parameter:
///
///   hpx.parcel.max_connections_per_locality = ...
///
/// (or by setting the corresponding environment variable
/// HPX_PARCEL_MAX_CONNECTIONS_PER_LOCALITY).
#if !defined(HPX_PARCEL_MAX_CONNECTIONS_PER_LOCALITY)
#  define HPX_PARCEL_MAX_CONNECTIONS_PER_LOCALITY 4
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the maximally allowed message size for messages transferred
/// between localities. This value can be changed at runtime by
/// setting the configuration parameter:
///
///   hpx.parcel.max_message_size = ...
///
/// (or by setting the corresponding environment variable
/// HPX_PARCEL_MAX_MESSAGE_SIZE).
#if !defined(HPX_PARCEL_MAX_MESSAGE_SIZE)
#  define HPX_PARCEL_MAX_MESSAGE_SIZE 1000000000
#endif

/// This defines the maximally allowed outbound  message size for coalescing
/// messages transferred between localities. This value can be changed at
/// runtime by setting the configuration parameter:
///
///   hpx.parcel.max_outbound_message_size = ...
///
/// (or by setting the corresponding environment variable
/// HPX_PARCEL_MAX_OUTBOUND_MESSAGE_SIZE).
#if !defined(HPX_PARCEL_MAX_OUTBOUND_MESSAGE_SIZE)
#  define HPX_PARCEL_MAX_OUTBOUND_MESSAGE_SIZE 1000000
#endif

///////////////////////////////////////////////////////////////////////////////
// This defines the number of bytes of overhead it takes to serialize a
// parcel.
#if !defined(HPX_PARCEL_SERIALIZATION_OVERHEAD)
#   define HPX_PARCEL_SERIALIZATION_OVERHEAD 512
#endif

/// This defines the number of AGAS address translations kept in the local
/// cache on a per OS-thread basis (system wide used OS threads).
#if !defined(HPX_AGAS_LOCAL_CACHE_SIZE_PER_THREAD)
#  define HPX_AGAS_LOCAL_CACHE_SIZE_PER_THREAD 32
#endif

/// This defines the number of AGAS address translations kept in the local
/// cache. This is just the initial size which may be adjusted depending on the
/// load of the system, etc. It must be a minimum of 3 for AGAS v3
/// bootstrapping.
/// The actual number of local cache entries used is determined by
///
///  max(HPX_INITIAL_AGAS_LOCAL_CACHE_SIZE,
///      HPX_AGAS_LOCAL_CACHE_SIZE_PER_THREAD * num_nodes)
///
#if !defined(HPX_INITIAL_AGAS_LOCAL_CACHE_SIZE)
#  define HPX_INITIAL_AGAS_LOCAL_CACHE_SIZE 256
#endif

///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_INITIAL_AGAS_MAX_PENDING_REFCNT_REQUESTS)
#  define HPX_INITIAL_AGAS_MAX_PENDING_REFCNT_REQUESTS 4096
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the initial global reference count associated with any created
/// object.
#if !defined(HPX_GLOBALCREDIT_INITIAL)
#  define HPX_GLOBALCREDIT_INITIAL 0x80000000ll     // 2 ^ 31, i.e. 2 ^ 0b11111
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the default number of OS-threads created for the different
/// internal thread pools
#if !defined(HPX_NUM_IO_POOL_SIZE)
#  define HPX_NUM_IO_POOL_SIZE 2
#endif
#if !defined(HPX_NUM_PARCEL_POOL_SIZE)
#  define HPX_NUM_PARCEL_POOL_SIZE 2
#endif
#if !defined(HPX_NUM_TIMER_POOL_SIZE)
#  define HPX_NUM_TIMER_POOL_SIZE 2
#endif

///////////////////////////////////////////////////////////////////////////////
/// By default, enable minimal thread deadlock detection in debug builds only.
#if !defined(HPX_THREAD_MINIMAL_DEADLOCK_DETECTION)
#  if defined(HPX_DEBUG)
#    define HPX_THREAD_MINIMAL_DEADLOCK_DETECTION
#  endif
#endif

///////////////////////////////////////////////////////////////////////////////
/// By default, enable storing the parent thread information in debug builds
/// only.
#if !defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
#  if defined(HPX_DEBUG)
#    define HPX_HAVE_THREAD_PARENT_REFERENCE
#  endif
#endif

///////////////////////////////////////////////////////////////////////////////
/// By default, enable storing the thread phase in debug builds only.
#if !defined(HPX_HAVE_THREAD_PHASE_INFORMATION)
#  if defined(HPX_DEBUG)
#    define HPX_HAVE_THREAD_PHASE_INFORMATION
#  endif
#endif

///////////////////////////////////////////////////////////////////////////////
/// By default, enable storing the thread description in debug builds only.
#if !defined(HPX_HAVE_THREAD_DESCRIPTION)
#  if defined(HPX_DEBUG)
#    define HPX_HAVE_THREAD_DESCRIPTION
#  endif
#endif

///////////////////////////////////////////////////////////////////////////////
/// By default, enable storing the target address of the data the thread is
/// accessing in debug builds only.
#if !defined(HPX_HAVE_THREAD_TARGET_ADDRESS)
#  if defined(HPX_DEBUG)
#    define HPX_HAVE_THREAD_TARGET_ADDRESS
#  endif
#endif

///////////////////////////////////////////////////////////////////////////////
/// By default we do not maintain stack back-traces on suspension. This is a
/// pure debugging aid to be able to see in the debugger where a suspended
/// thread got stuck.
#if defined(HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION) && \
  !defined(HPX_HAVE_STACKTRACES)
#  error HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION reqires HPX_HAVE_STACKTRACES to be defined!
#endif

/// By default we capture only 5 levels of stack back trace on suspension
#if !defined(HPX_THREAD_BACKTRACE_ON_SUSPENSION_DEPTH)
#  define HPX_THREAD_BACKTRACE_ON_SUSPENSION_DEPTH 5
#endif

///////////////////////////////////////////////////////////////////////////////
// This defines the maximum number of connect retries to the AGAS service
// allowing for some leeway during startup of the localities
#if !defined(HPX_MAX_NETWORK_RETRIES)
#  define HPX_MAX_NETWORK_RETRIES 1000
#endif

#if !defined(HPX_NETWORK_RETRIES_SLEEP)
#  define HPX_NETWORK_RETRIES_SLEEP 100
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
#  elif defined(HPX_HAVE_STATIC_LINKING)
#    define HPX_SHARED_LIB_EXTENSION        ".a"
#  else  // linux & co
#    define HPX_SHARED_LIB_EXTENSION        ".so"
#  endif
#endif

///////////////////////////////////////////////////////////////////////////////
#if !defined(BOOST_WINDOWS)
#  if defined(HPX_DEBUG)
#    define HPX_MAKE_DLL_STRING(n)  "lib" + n + "d" + HPX_SHARED_LIB_EXTENSION
#  else
#    define HPX_MAKE_DLL_STRING(n)  "lib" + n + HPX_SHARED_LIB_EXTENSION
#  endif
#elif defined(HPX_DEBUG)
#  define HPX_MAKE_DLL_STRING(n)   n + "d" + HPX_SHARED_LIB_EXTENSION
#else
#  define HPX_MAKE_DLL_STRING(n)   n + HPX_SHARED_LIB_EXTENSION
#endif

#if defined(HPX_DEBUG)
#  define HPX_MANGLE_NAME(n)     BOOST_PP_CAT(n, d)
#  define HPX_MANGLE_STRING(n)   n + "d"
#else
#  define HPX_MANGLE_NAME(n)     n
#  define HPX_MANGLE_STRING(n)   n
#endif

///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_COMPONENT_NAME)
#  define HPX_COMPONENT_NAME hpx
#endif

#if !defined(HPX_COMPONENT_STRING)
#  define HPX_COMPONENT_STRING BOOST_PP_STRINGIZE(HPX_COMPONENT_NAME)
#endif

///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_PLUGIN_NAME)
#  define HPX_PLUGIN_NAME hpx
#endif

#if !defined(HPX_PLUGIN_STRING)
#  define HPX_PLUGIN_STRING BOOST_PP_STRINGIZE(HPX_PLUGIN_NAME)
#endif

#if !defined(HPX_PLUGIN_PLUGIN_PREFIX)
#  define HPX_PLUGIN_PLUGIN_PREFIX HPX_MANGLE_NAME(HPX_PLUGIN_NAME)
#endif

///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_PLUGIN_COMPONENT_PREFIX)
#  define HPX_PLUGIN_COMPONENT_PREFIX HPX_MANGLE_NAME(HPX_COMPONENT_NAME)
#endif

///////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_APPLICATION_STRING)
#  if defined(HPX_APPLICATION_NAME)
#    define HPX_APPLICATION_STRING BOOST_PP_STRINGIZE(HPX_APPLICATION_NAME)
#  else
#    define HPX_APPLICATION_STRING "unknown HPX application"
#  endif
#endif

///////////////////////////////////////////////////////////////////////////////
#if defined(BOOST_WINDOWS) && _MSC_VER < 1900
#  define snprintf _snprintf
#endif

///////////////////////////////////////////////////////////////////////////////
// Count number of empty (no HPX thread available) thread manager loop executions
#if !defined(HPX_IDLE_LOOP_COUNT_MAX)
#  define HPX_IDLE_LOOP_COUNT_MAX 200000
#endif

///////////////////////////////////////////////////////////////////////////////
// Count number of busy thread manager loop executions before forcefully
// cleaning up terminated thread objects
#if !defined(HPX_BUSY_LOOP_COUNT_MAX)
#  define HPX_BUSY_LOOP_COUNT_MAX 200000
#endif

///////////////////////////////////////////////////////////////////////////////
// Count number of terminated threads before forcefully cleaning up all of
// them. Note: terminated threads are cleaned up either when this number is
// reached for a particular thread queue or if the HPX_BUSY_LOOP_COUNT_MAX is
// reached, which will clean up the terminated threads for _all_ thread queues.
#if !defined(HPX_MAX_TERMINATED_THREADS)
#  define HPX_MAX_TERMINATED_THREADS 1000
#endif

///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_WRAPPER_HEAP_STEP)
#  define HPX_WRAPPER_HEAP_STEP 0xFFFFU
#endif

#if !defined(HPX_INITIAL_GID_RANGE)
#  define HPX_INITIAL_GID_RANGE 0xFFFFU
#endif

///////////////////////////////////////////////////////////////////////////////
// Enable lock verification code which allows to check whether there are locks
// held while HPX-threads are suspended and/or interrupted.
#if !defined(HPX_HAVE_VERIFY_LOCKS)
#  if defined(HPX_DEBUG)
#    define HPX_HAVE_VERIFY_LOCKS
#  endif
#endif

#if !defined(HPX_HAVE_VERIFY_LOCKS_GLOBALY)
#  if defined(HPX_DEBUG)
#    define HPX_HAVE_VERIFY_LOCKS_GLOBALY
#  endif
#endif

///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_SMALL_STACK_SIZE)
#  if defined(BOOST_WINDOWS) && !defined(HPX_HAVE_GENERIC_CONTEXT_COROUTINES)
#    define HPX_SMALL_STACK_SIZE    0x4000        // 16kByte
#  else
#    if defined(HPX_DEBUG)
#      define HPX_SMALL_STACK_SIZE  0x10000       // 64kByte
#    else
#      define HPX_SMALL_STACK_SIZE  0x8000        // 32kByte
#    endif
#  endif
#endif

#if !defined(HPX_MEDIUM_STACK_SIZE)
#  define HPX_MEDIUM_STACK_SIZE   0x0020000       // 128kByte
#endif
#if !defined(HPX_LARGE_STACK_SIZE)
#  define HPX_LARGE_STACK_SIZE    0x0200000       // 2MByte
#endif
#if !defined(HPX_HUGE_STACK_SIZE)
#  define HPX_HUGE_STACK_SIZE     0x2000000       // 32MByte
#endif

///////////////////////////////////////////////////////////////////////////////
// This limits how deep the internal recursion of future continuations will go
// before a new operation is re-spawned.
#if !defined(HPX_CONTINUATION_MAX_RECURSION_DEPTH)
#if defined(HPX_DEBUG)
#define HPX_CONTINUATION_MAX_RECURSION_DEPTH 14
#else
#define HPX_CONTINUATION_MAX_RECURSION_DEPTH 20
#endif
#endif

///////////////////////////////////////////////////////////////////////////////
// Older Boost versions do not have BOOST_NOEXCEPT defined
#if !defined(BOOST_NOEXCEPT)
#  define BOOST_NOEXCEPT
#  define BOOST_NOEXCEPT_IF(Predicate)
#  define BOOST_NOEXCEPT_EXPR(Expression) false
#endif

///////////////////////////////////////////////////////////////////////////////
// Older Boost versions do not have BOOST_NOINLINE defined
#if !defined(BOOST_NOINLINE)
#  if defined(_MSC_VER)
#    define BOOST_NOINLINE __declspec(noinline)
#  else
#    define BOOST_NOINLINE
#  endif
#endif

///////////////////////////////////////////////////////////////////////////////
// Older Boost versions do not have BOOST_NORETURN defined
#if defined(BOOST_NORETURN)
#  define HPX_ATTRIBUTE_NORETURN BOOST_NORETURN
#elif defined(BOOST_ATTRIBUTE_NORETURN)
#  define HPX_ATTRIBUTE_NORETURN BOOST_ATTRIBUTE_NORETURN
#else
#  define HPX_ATTRIBUTE_NORETURN
#endif

///////////////////////////////////////////////////////////////////////////////
// GCC has issues with forceinline and member function pointers
#if defined(HPX_GCC_VERSION)
#  define HPX_MAYBE_FORCEINLINE inline
#else
#  define HPX_MAYBE_FORCEINLINE BOOST_FORCEINLINE
#endif

///////////////////////////////////////////////////////////////////////////////
// Make sure we have support for more than 64 threads for Xeon Phi
#if defined(__MIC__) && !defined(HPX_WITH_MORE_THAN_64_THREADS)
#  define HPX_WITH_MORE_THAN_64_THREADS
#endif
#if defined(__MIC__) && !defined(HPX_HAVE_MAX_CPU_COUNT)
#  define HPX_HAVE_MAX_CPU_COUNT 256
#endif

///////////////////////////////////////////////////////////////////////////////
#define HPX_AGAS_BOOTSTRAP_PREFIX                    0U

#define HPX_AGAS_NS_MSB                              0x0000000000000001ULL

#define HPX_AGAS_PRIMARY_NS_MSB                      0x0000000100000001ULL
#define HPX_AGAS_PRIMARY_NS_LSB                      0x0000000000000001ULL
#define HPX_AGAS_COMPONENT_NS_MSB                    0x0000000100000001ULL
#define HPX_AGAS_COMPONENT_NS_LSB                    0x0000000000000002ULL
#define HPX_AGAS_SYMBOL_NS_MSB                       0x0000000100000001ULL
#define HPX_AGAS_SYMBOL_NS_LSB                       0x0000000000000003ULL
#define HPX_AGAS_LOCALITY_NS_MSB                     0x0000000100000001ULL
#define HPX_AGAS_LOCALITY_NS_LSB                     0x0000000000000004ULL

#if defined(HPX_HAVE_SODIUM)
#  define HPX_ROOT_CERTIFICATE_AUTHORITY_MSB         0x0000000100000001ULL
#  define HPX_ROOT_CERTIFICATE_AUTHORITY_LSB         0x0000000000000005ULL
#  define HPX_SUBORDINATE_CERTIFICATE_AUTHORITY_MSB  0x0000000000000001ULL
// this is made locality specific
#  define HPX_SUBORDINATE_CERTIFICATE_AUTHORITY_LSB  0x0000000000000006ULL
#endif

#if !defined(HPX_NO_DEPRECATED)
#  define HPX_DEPRECATED_MSG \
   "This function is deprecated and will be removed in the future."
#  if defined(_MSC_VER)
#    define HPX_DEPRECATED(x) __declspec(deprecated(x))
#  elif (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 5))
#    define HPX_DEPRECATED(x) __attribute__((__deprecated__(x)))
#  elif (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4))
#    define HPX_DEPRECATED(x) __attribute__((__deprecated__))
#  endif
#  if !defined(HPX_DEPRECATED)
#    define HPX_DEPRECATED(x)  /**/
#  endif
#endif

#endif
