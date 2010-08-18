/*
    Copyright(c) 2005-2010 Intel Corporation.  All Rights Reserved.

    The source code contained or described herein and all documents related
    to the source code ("Material") are owned by Intel Corporation or its
    suppliers or licensors.  Title to the Material remains with Intel
    Corporation or its suppliers and licensors.  The Material is protected
    by worldwide copyright laws and treaty provisions.  No part of the
    Material may be used, copied, reproduced, modified, published, uploaded,
    posted, transmitted, distributed, or disclosed in any way without
    Intel's prior express written permission.

    No license under any patent, copyright, trade secret or other
    intellectual property right is granted to or conferred upon you by
    disclosure or delivery of the Materials, either expressly, by
    implication, inducement, estoppel or otherwise.  Any license under such
    intellectual property rights must be express and approved by Intel in
    writing.
*/
#ifndef _ITTNOTIFY_H_
#define _ITTNOTIFY_H_
/**
 * @file
 * @brief Public User API functions and types
 * @mainpage
 * Ability to control the collection during runtime. User API can be inserted into the user application.
 * Commands include:
 *   - Collection control
 *   - Marking
 *   - Thread manipulation
 *   - User-defined synchronization primitives
 *
 * The User API provides ability to control the collection, set marks at the execution of specific user code and
 * specify custom synchronization primitives implemented without standard system APIs.
 *
 * Use case: User inserts API calls to the desired places in her code. The code is then compiled and
 * linked with static part of User API library. User can recompile the code with specific macro defined
 * to enable API calls.  If this macro is not defined there is no run-time overhead and no need to  link
 * with static part of User API library. During  runtime the static library loads and initializes the dynamic part.
 * In case of instrumentation-based collection, only a stub library is loaded; otherwise a proxy library is loaded,
 * which calls the collector.
 *  
 * User API set is native (C/C++) only (no MRTE support). As a mitigation can use JNI or C/C++ function
 * call from managed code where needed. If the collector causes significant overhead or data storage, then
 * pausing analysis should reduce the overhead to minimal levels.
 */

/** @cond exclude_from_documentation */
#ifndef ITT_OS_WIN
#  define ITT_OS_WIN   1
#endif /* ITT_OS_WIN */

#ifndef ITT_OS_LINUX
#  define ITT_OS_LINUX 2
#endif /* ITT_OS_LINUX */

#ifndef ITT_OS_MAC
#  define ITT_OS_MAC   3
#endif /* ITT_OS_MAC */

#ifndef ITT_OS
#  if defined WIN32 || defined _WIN32
#    define ITT_OS ITT_OS_WIN
#  elif defined( __APPLE__ ) && defined( __MACH__ )
#    define ITT_OS ITT_OS_MAC
#  else
#    define ITT_OS ITT_OS_LINUX
#  endif
#endif /* ITT_OS */

#ifndef ITT_PLATFORM_WIN
#  define ITT_PLATFORM_WIN 1
#endif /* ITT_PLATFORM_WIN */

#ifndef ITT_PLATFORM_POSIX
#  define ITT_PLATFORM_POSIX 2
#endif /* ITT_PLATFORM_POSIX */

#ifndef ITT_PLATFORM
#  if ITT_OS==ITT_OS_WIN
#    define ITT_PLATFORM ITT_PLATFORM_WIN
#  else
#    define ITT_PLATFORM ITT_PLATFORM_POSIX
#  endif /* _WIN32 */
#endif /* ITT_PLATFORM */

#include <stddef.h>
#include <stdarg.h>
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#include <tchar.h>
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */

#ifndef CDECL
#  if ITT_PLATFORM==ITT_PLATFORM_WIN
#    define CDECL __cdecl
#  else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#    define CDECL /* nothing */
#  endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#endif /* CDECL */

#ifndef STDCALL
#  if ITT_PLATFORM==ITT_PLATFORM_WIN
#    define STDCALL __stdcall
#  else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#    define STDCALL /* nothing */
#  endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#endif /* STDCALL */

#define ITTAPI    CDECL
#define LIBITTAPI /* nothing */

#ifdef INTEL_ITTNOTIFY_ENABLE_LEGACY
#  if ITT_PLATFORM==ITT_PLATFORM_WIN
#    pragma message("WARNING!!! Deprecated API is used. Please undefine INTEL_ITTNOTIFY_ENABLE_LEGACY macro")
#  else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#    warning "Deprecated API is used. Please undefine INTEL_ITTNOTIFY_ENABLE_LEGACY macro"
#  endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#  include "legacy/ittnotify.h"
#endif /* INTEL_ITTNOTIFY_ENABLE_LEGACY */

#define ITT_JOIN_AUX(p,n) p##n
#define ITT_JOIN(p,n)     ITT_JOIN_AUX(p,n)

#ifndef INTEL_ITTNOTIFY_PREFIX
#  define INTEL_ITTNOTIFY_PREFIX __itt_
#endif /* INTEL_ITTNOTIFY_PREFIX */
#ifndef INTEL_ITTNOTIFY_POSTFIX
#  define INTEL_ITTNOTIFY_POSTFIX _ptr_
#endif /* INTEL_ITTNOTIFY_POSTFIX */

#define ITTNOTIFY_NAME_AUX(n) ITT_JOIN(INTEL_ITTNOTIFY_PREFIX,n)
#define ITTNOTIFY_NAME(n)     ITTNOTIFY_NAME_AUX(ITT_JOIN(n,INTEL_ITTNOTIFY_POSTFIX))

#define ITTNOTIFY_VOID(n) (!ITTNOTIFY_NAME(n)) ? (void)0 : ITTNOTIFY_NAME(n)
#define ITTNOTIFY_DATA(n) (!ITTNOTIFY_NAME(n)) ?       0 : ITTNOTIFY_NAME(n)

#if !defined(ITT_EXPORT)
#define ITT_EXPORT
#endif

#ifdef ITT_STUB
#undef ITT_STUB
#endif
#ifdef ITT_STUBV
#undef ITT_STUBV
#endif
#define ITT_STUBV(api,type,name,args,params)                      \
    typedef type (api* ITT_JOIN(ITTNOTIFY_NAME(name),_t)) args;   \
    extern ITT_EXPORT ITT_JOIN(ITTNOTIFY_NAME(name),_t) ITTNOTIFY_NAME(name);
#define ITT_STUB ITT_STUBV

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
/** @endcond */

/**
 * @defgroup public Public API
 * @{
 * @}
 */

/**
 * @defgroup control Collection Control
 * @ingroup public
 * General behavior: application continues to run, but no profiling information is being collected
 *
 * Pausing occurs not only for the current thread but for all process as well as spawned processes
 * - Intel(R) Parallel Inspector:
 *   - Does not analyze or report errors that involve memory access.
 *   - Other errors are reported as usual. Pausing data collection in
 *     Intel(R) Parallel Inspector only pauses tracing and analyzing
 *     memory access. It does not pause tracing or analyzing threading APIs.
 *   .
 * - Intel(R) Parallel Amplifier:
 *   - Does continue to record when new threads are started.
 *   .
 * - Other effects:
 *   - Possible reduction of runtime overhead.
 *   .
 * @{
 */
/** @brief Pause collection */
void ITTAPI __itt_pause(void);
/** @brief Resume collection */
void ITTAPI __itt_resume(void);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, pause,  (void), ())
ITT_STUBV(ITTAPI, void, resume, (void), ())
#define __itt_pause      ITTNOTIFY_VOID(pause)
#define __itt_pause_ptr  ITTNOTIFY_NAME(pause)
#define __itt_resume     ITTNOTIFY_VOID(resume)
#define __itt_resume_ptr ITTNOTIFY_NAME(resume)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_pause()
#define __itt_pause_ptr  0
#define __itt_resume()
#define __itt_resume_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_pause_ptr  0
#define __itt_resume_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */
/** @} control group */

/**
 * @defgroup threads Threads
 * @ingroup public
 * Threads name group
 * @{
 */
/**
 * @brief Sets thread name using char or Unicode string
 * @param[in] name - name of thread
 */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
void ITTAPI __itt_thread_set_nameA(const char    *name);
void ITTAPI __itt_thread_set_nameW(const wchar_t *name);
#if defined(UNICODE) || defined(_UNICODE)
#  define __itt_thread_set_name     __itt_thread_set_nameW
#  define __itt_thread_set_name_ptr __itt_thread_set_nameW_ptr
#else /* UNICODE */
#  define __itt_thread_set_name     __itt_thread_set_nameA
#  define __itt_thread_set_name_ptr __itt_thread_set_nameA_ptr
#endif /* UNICODE */
#else  /* ITT_PLATFORM==ITT_PLATFORM_WIN */
void ITTAPI __itt_thread_set_name(const char *name);
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUBV(ITTAPI, void, thread_set_nameA, (const char    *name), (name))
ITT_STUBV(ITTAPI, void, thread_set_nameW, (const wchar_t *name), (name))
#else  /* ITT_PLATFORM==ITT_PLATFORM_WIN */
ITT_STUBV(ITTAPI, void, thread_set_name,  (const char    *name), (name))
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_thread_set_nameA     ITTNOTIFY_VOID(thread_set_nameA)
#define __itt_thread_set_nameA_ptr ITTNOTIFY_NAME(thread_set_nameA)
#define __itt_thread_set_nameW     ITTNOTIFY_VOID(thread_set_nameW)
#define __itt_thread_set_nameW_ptr ITTNOTIFY_NAME(thread_set_nameW)
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_thread_set_name     ITTNOTIFY_VOID(thread_set_name)
#define __itt_thread_set_name_ptr ITTNOTIFY_NAME(thread_set_name)
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#else  /* INTEL_NO_ITTNOTIFY_API */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_thread_set_nameA(name)
#define __itt_thread_set_nameA_ptr 0
#define __itt_thread_set_nameW(name)
#define __itt_thread_set_nameW_ptr 0
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_thread_set_name(name)
#define __itt_thread_set_name_ptr 0
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_thread_set_nameA_ptr 0
#define __itt_thread_set_nameW_ptr 0
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_thread_set_name_ptr 0
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Mark current thread as ignored from this point on, for the duration of its existence.
 */
void ITTAPI __itt_thread_ignore(void);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, thread_ignore, (void), ())
#define __itt_thread_ignore     ITTNOTIFY_VOID(thread_ignore)
#define __itt_thread_ignore_ptr ITTNOTIFY_NAME(thread_ignore)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_thread_ignore()
#define __itt_thread_ignore_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_thread_ignore_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */
/** @} threads group */

/**
 * @defgroup sync Synchronization
 * @ingroup public
 * Synchronization group
 * @{
 */
/**
 * @hideinitializer
 * @brief possible value of attribute argument for sync object type
 */
#define __itt_attr_barrier 1

/**
 * @hideinitializer
 * @brief possible value of attribute argument for sync object type
 */
#define __itt_attr_mutex   2

/**
 * @brief Register the creation of a sync object using char or Unicode string
 * @param[in] addr      - pointer to the sync object. You should use a real pointer to your object
 *                        to make sure that the values don't clash with other object addresses
 * @param[in] objtype   - null-terminated object type string. If NULL is passed, the object will
 *                        be assumed to be of generic "User Synchronization" type
 * @param[in] objname   - null-terminated object name string. If NULL, no name will be assigned
 *                        to the object -- you can use the __itt_sync_rename call later to assign
 *                        the name
 * @param[in] attribute - one of [#__itt_attr_barrier, #__itt_attr_mutex] values which defines the
 *                        exact semantics of how prepare/acquired/releasing calls work.
 */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
void ITTAPI __itt_sync_createA(void *addr, const char    *objtype, const char    *objname, int attribute);
void ITTAPI __itt_sync_createW(void *addr, const wchar_t *objtype, const wchar_t *objname, int attribute);
#if defined(UNICODE) || defined(_UNICODE)
#  define __itt_sync_create     __itt_sync_createW
#  define __itt_sync_create_ptr __itt_sync_createW_ptr
#else /* UNICODE */
#  define __itt_sync_create     __itt_sync_createA
#  define __itt_sync_create_ptr __itt_sync_createA_ptr
#endif /* UNICODE */
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
void ITTAPI __itt_sync_create (void *addr, const char *objtype, const char *objname, int attribute);
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUBV(ITTAPI, void, sync_createA, (void *addr, const char    *objtype, const char    *objname, int attribute), (addr, objtype, objname, attribute))
ITT_STUBV(ITTAPI, void, sync_createW, (void *addr, const wchar_t *objtype, const wchar_t *objname, int attribute), (addr, objtype, objname, attribute))
#else  /* ITT_PLATFORM==ITT_PLATFORM_WIN */
ITT_STUBV(ITTAPI, void, sync_create,  (void *addr, const char* objtype, const char* objname, int attribute), (addr, objtype, objname, attribute))
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_sync_createA     ITTNOTIFY_VOID(sync_createA)
#define __itt_sync_createA_ptr ITTNOTIFY_NAME(sync_createA)
#define __itt_sync_createW     ITTNOTIFY_VOID(sync_createW)
#define __itt_sync_createW_ptr ITTNOTIFY_NAME(sync_createW)
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_sync_create     ITTNOTIFY_VOID(sync_create)
#define __itt_sync_create_ptr ITTNOTIFY_NAME(sync_create)
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#else  /* INTEL_NO_ITTNOTIFY_API */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_sync_createA(addr, objtype, objname, attribute)
#define __itt_sync_createA_ptr 0
#define __itt_sync_createW(addr, objtype, objname, attribute)
#define __itt_sync_createW_ptr 0
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_sync_create(addr, objtype, objname, attribute)
#define __itt_sync_create_ptr 0
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_sync_createA_ptr 0
#define __itt_sync_createW_ptr 0
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_sync_create_ptr 0
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Assign a name to a sync object using char or Unicode string.
 *
 * Sometimes you cannot assign the name to a sync object in the __itt_sync_set_name() call because it
 * is not yet known there. In this case you should use the rename call which allows to assign the
 * name after the creation has been registered. The renaming can be done multiple times. All waits
 * after a new name has been assigned will be attributed to the sync object with this name.
 * @param[in] addr - pointer to the sync object
 * @param[in] name - null-terminated object name string
 */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
void ITTAPI __itt_sync_renameA(void *addr, const char    *name);
void ITTAPI __itt_sync_renameW(void *addr, const wchar_t *name);
#if defined(UNICODE) || defined(_UNICODE)
#  define __itt_sync_rename     __itt_sync_renameW
#  define __itt_sync_rename_ptr __itt_sync_renameW_ptr
#else /* UNICODE */
#  define __itt_sync_rename     __itt_sync_renameA
#  define __itt_sync_rename_ptr __itt_sync_renameA_ptr
#endif /* UNICODE */
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
void ITTAPI __itt_sync_rename(void *addr, const char *name);
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUBV(ITTAPI, void, sync_renameA, (void *addr, const char    *name), (addr, name))
ITT_STUBV(ITTAPI, void, sync_renameW, (void *addr, const wchar_t *name), (addr, name))
#else  /* ITT_PLATFORM==ITT_PLATFORM_WIN */
ITT_STUBV(ITTAPI, void, sync_rename,  (void *addr, const char    *name), (addr, name))
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_sync_renameA     ITTNOTIFY_VOID(sync_renameA)
#define __itt_sync_renameA_ptr ITTNOTIFY_NAME(sync_renameA)
#define __itt_sync_renameW     ITTNOTIFY_VOID(sync_renameW)
#define __itt_sync_renameW_ptr ITTNOTIFY_NAME(sync_renameW)
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_sync_rename     ITTNOTIFY_VOID(sync_rename)
#define __itt_sync_rename_ptr ITTNOTIFY_NAME(sync_rename)
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#else  /* INTEL_NO_ITTNOTIFY_API */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_sync_renameA(addr, name)
#define __itt_sync_renameA_ptr 0
#define __itt_sync_renameW(addr, name)
#define __itt_sync_renameW_ptr 0
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_sync_rename(addr, name)
#define __itt_sync_rename_ptr 0
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_sync_renameA_ptr 0
#define __itt_sync_renameW_ptr 0
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_sync_rename_ptr 0
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Is called when sync object is destroyed (needed to track lifetime of objects)
 */
void ITTAPI __itt_sync_destroy(void *addr);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, sync_destroy, (void *addr), (addr))
#define __itt_sync_destroy     ITTNOTIFY_VOID(sync_destroy)
#define __itt_sync_destroy_ptr ITTNOTIFY_NAME(sync_destroy)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_sync_destroy(addr)
#define __itt_sync_destroy_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_sync_destroy_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/*****************************************************************//**
 * @name group of functions is used for performance measurement tools
 *********************************************************************/
/** @{ */
/**
 * @brief Enter spin loop on user-defined sync object
 */
void ITTAPI __itt_sync_prepare(void* addr);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, sync_prepare, (void *addr), (addr))
#define __itt_sync_prepare     ITTNOTIFY_VOID(sync_prepare)
#define __itt_sync_prepare_ptr ITTNOTIFY_NAME(sync_prepare)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_sync_prepare(addr)
#define __itt_sync_prepare_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_sync_prepare_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Quit spin loop without acquiring spin object
 */
void ITTAPI __itt_sync_cancel(void *addr);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, sync_cancel, (void *addr), (addr))
#define __itt_sync_cancel     ITTNOTIFY_VOID(sync_cancel)
#define __itt_sync_cancel_ptr ITTNOTIFY_NAME(sync_cancel)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_sync_cancel(addr)
#define __itt_sync_cancel_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_sync_cancel_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Successful spin loop completion (sync object acquired)
 */
void ITTAPI __itt_sync_acquired(void *addr);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, sync_acquired, (void *addr), (addr))
#define __itt_sync_acquired     ITTNOTIFY_VOID(sync_acquired)
#define __itt_sync_acquired_ptr ITTNOTIFY_NAME(sync_acquired)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_sync_acquired(addr)
#define __itt_sync_acquired_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_sync_acquired_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Start sync object releasing code. Is called before the lock release call.
 */
void ITTAPI __itt_sync_releasing(void* addr);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, sync_releasing, (void *addr), (addr))
#define __itt_sync_releasing     ITTNOTIFY_VOID(sync_releasing)
#define __itt_sync_releasing_ptr ITTNOTIFY_NAME(sync_releasing)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_sync_releasing(addr)
#define __itt_sync_releasing_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_sync_releasing_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */
/** @} */

/**
 * @brief Start sync object released code. Is called after the lock release call.
 */
void ITTAPI __itt_sync_released(void* addr);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, sync_released, (void *addr), (addr))
#define __itt_sync_released     ITTNOTIFY_VOID(sync_released)
#define __itt_sync_released_ptr ITTNOTIFY_NAME(sync_released)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_sync_released(addr)
#define __itt_sync_released_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_sync_released_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */
/** @} */

/**************************************************************//**
 * @name group of functions is used for correctness checking tools
 ******************************************************************/
/** @{ */
/**
 * @brief Fast synchronization which does no require spinning.
 * - This special function is to be used by TBB and OpenMP libraries only when they know
 *   there is no spin but they need to suppress TC warnings about shared variable modifications.
 * - It only has corresponding pointers in static library and does not have corresponding function
 *   in dynamic library.
 * @see void __itt_sync_prepare(void* addr);
 */
void ITTAPI __itt_fsync_prepare(void* addr);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, fsync_prepare, (void *addr), (addr))
#define __itt_fsync_prepare     ITTNOTIFY_VOID(fsync_prepare)
#define __itt_fsync_prepare_ptr ITTNOTIFY_NAME(fsync_prepare)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_fsync_prepare(addr)
#define __itt_fsync_prepare_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_fsync_prepare_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Fast synchronization which does no require spinning.
 * - This special function is to be used by TBB and OpenMP libraries only when they know
 *   there is no spin but they need to suppress TC warnings about shared variable modifications.
 * - It only has corresponding pointers in static library and does not have corresponding function
 *   in dynamic library.
 * @see void __itt_sync_cancel(void *addr);
 */
void ITTAPI __itt_fsync_cancel(void *addr);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, fsync_cancel, (void *addr), (addr))
#define __itt_fsync_cancel     ITTNOTIFY_VOID(fsync_cancel)
#define __itt_fsync_cancel_ptr ITTNOTIFY_NAME(fsync_cancel)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_fsync_cancel(addr)
#define __itt_fsync_cancel_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_fsync_cancel_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Fast synchronization which does no require spinning.
 * - This special function is to be used by TBB and OpenMP libraries only when they know
 *   there is no spin but they need to suppress TC warnings about shared variable modifications.
 * - It only has corresponding pointers in static library and does not have corresponding function
 *   in dynamic library.
 * @see void __itt_sync_acquired(void *addr);
 */
void ITTAPI __itt_fsync_acquired(void *addr);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, fsync_acquired, (void *addr), (addr))
#define __itt_fsync_acquired     ITTNOTIFY_VOID(fsync_acquired)
#define __itt_fsync_acquired_ptr ITTNOTIFY_NAME(fsync_acquired)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_fsync_acquired(addr)
#define __itt_fsync_acquired_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_fsync_acquired_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Fast synchronization which does no require spinning.
 * - This special function is to be used by TBB and OpenMP libraries only when they know
 *   there is no spin but they need to suppress TC warnings about shared variable modifications.
 * - It only has corresponding pointers in static library and does not have corresponding function
 *   in dynamic library.
 * @see void __itt_sync_releasing(void* addr);
 */
void ITTAPI __itt_fsync_releasing(void* addr);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, fsync_releasing, (void *addr), (addr))
#define __itt_fsync_releasing     ITTNOTIFY_VOID(fsync_releasing)
#define __itt_fsync_releasing_ptr ITTNOTIFY_NAME(fsync_releasing)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_fsync_releasing(addr)
#define __itt_fsync_releasing_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_fsync_releasing_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */
/** @} */
/** @} sync group */

/**
 * @defgroup model Modeling by Intel(R) Parallel Advisor
 * @ingroup public
 * This is the subset of itt used for modeling by Intel(R) Parallel Advisor.
 * This API is called ONLY using annotate.h, by "Annotation" macros
 * the user places in their sources during the parallelism modeling steps.
 *
 * The requirements, constraints, design and implementation
 * for this interface are covered in:
 * Shared%20Documents/Design%20Documents/AdvisorAnnotations.doc
 *
 * site_begin/end and task_begin/end take the address of handle variables,
 * which are writeable by the API.  Handles must be 0 initialized prior
 * to the first call to begin, or may cause a run-time failure.
 * The handles are initialized in a multi-thread safe way by the API if
 * the handle is 0.  The commonly expected idiom is one static handle to
 * identify a site or task.  If a site or task of the same name has already
 * been started during this collection, the same handle MAY be returned,
 * but is not required to be - it is unspecified if data merging is done
 * based on name.  These routines also take an instance variable.  Like
 * the lexical instance, these must be 0 initialized.  Unlike the lexical
 * instance, this is used to track a single dynamic instance.
 *
 * API used by the Intel(R) Parallel Advisor to describe potential concurrency
 * and related activities. User-added source annotations expand to calls
 * to these procedures to enable modeling of a hypothetical concurrent
 * execution serially.
 * @{
 */
typedef void* __itt_model_site;             /*!< @brief handle for lexical site     */
typedef void* __itt_model_site_instance;    /*!< @brief handle for dynamic instance */
typedef void* __itt_model_task;             /*!< @brief handle for lexical site     */
typedef void* __itt_model_task_instance;    /*!< @brief handle for dynamic instance */

/**
 * @enum __itt_model_disable
 * @brief Enumerator for the disable methods
 */
typedef enum {
    __itt_model_disable_observation,
    __itt_model_disable_collection
} __itt_model_disable;

/**
 * @brief ANNOTATE_SITE_BEGIN/ANNOTATE_SITE_END support.
 *
 * site_begin/end model a potential concurrency site.
 * site instances may be recursively nested with themselves.
 * site_end exits the most recently started but unended site for the current
 * thread.  The handle passed to end may be used to validate structure.
 * Instances of a site encountered on different threads concurrently
 * are considered completely distinct. If the site name for two different
 * lexical sites match, it is unspecified whether they are treated as the
 * same or different for data presentation.
 */
void ITTAPI __itt_model_site_begin(__itt_model_site *site, __itt_model_site_instance *instance, const char *name);
void ITTAPI __itt_model_site_end  (__itt_model_site *site, __itt_model_site_instance *instance);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, model_site_begin,  (__itt_model_site *site, __itt_model_site_instance *instance, const char *name), (site, instance, name))
ITT_STUBV(ITTAPI, void, model_site_end,    (__itt_model_site *site, __itt_model_site_instance *instance), (site, instance))
#define __itt_model_site_begin      ITTNOTIFY_VOID(model_site_begin)
#define __itt_model_site_begin_ptr  ITTNOTIFY_NAME(model_site_begin)
#define __itt_model_site_end        ITTNOTIFY_VOID(model_site_end)
#define __itt_model_site_end_ptr    ITTNOTIFY_NAME(model_site_end)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_model_site_begin(site, instance, name)
#define __itt_model_site_begin_ptr  0
#define __itt_model_site_end(site, instance)
#define __itt_model_site_end_ptr    0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_model_site_begin_ptr  0
#define __itt_model_site_end_ptr    0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief ANNOTATE_TASK_BEGIN/ANNOTATE_TASK_END support
 *
 * task_begin/end model a potential task, which is contained within the most
 * closely enclosing dynamic site.  task_end exits the most recently started
 * but unended task.  The handle passed to end may be used to validate
 * structure.  It is unspecified if bad dynamic nesting is detected.  If it
 * is, it should be encoded in the resulting data collection.  The collector
 * should not fail due to construct nesting issues, nor attempt to directly
 * indicate the problem.
 */
void ITTAPI __itt_model_task_begin(__itt_model_task *task, __itt_model_task_instance *instance, const char *name);
void ITTAPI __itt_model_task_end  (__itt_model_task *task, __itt_model_task_instance *instance);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, model_task_begin,  (__itt_model_task *task, __itt_model_task_instance *instance, const char *name), (task, instance, name))
ITT_STUBV(ITTAPI, void, model_task_end,    (__itt_model_task *task, __itt_model_task_instance *instance), (task, instance))
#define __itt_model_task_begin      ITTNOTIFY_VOID(model_task_begin)
#define __itt_model_task_begin_ptr  ITTNOTIFY_NAME(model_task_begin)
#define __itt_model_task_end        ITTNOTIFY_VOID(model_task_end)
#define __itt_model_task_end_ptr    ITTNOTIFY_NAME(model_task_end)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_model_task_begin(task, instance, name)
#define __itt_model_task_begin_ptr  0
#define __itt_model_task_end(task, instance)
#define __itt_model_task_end_ptr    0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_model_task_begin_ptr  0
#define __itt_model_task_end_ptr    0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief ANNOTATE_LOCK_ACQUIRE/ANNOTATE_LOCK_RELEASE support
 *
 * lock_acquire/release model a potential lock for both lockset and
 * performance modeling.  Each unique address is modeled as a separate
 * lock, with invalid addresses being valid lock IDs.  Specifically:
 * no storage is accessed by the API at the specified address - it is only
 * used for lock identification.  Lock acquires may be self-nested and are
 * unlocked by a corresponding number of releases.
 * (These closely correspond to __itt_sync_acquired/__itt_sync_releasing,
 * but may not have identical semantics.)
 */
void ITTAPI __itt_model_lock_acquire(void *lock);
void ITTAPI __itt_model_lock_release(void *lock);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, model_lock_acquire, (void *lock), (lock))
ITT_STUBV(ITTAPI, void, model_lock_release, (void *lock), (lock))
#define __itt_model_lock_acquire     ITTNOTIFY_VOID(model_lock_acquire)
#define __itt_model_lock_acquire_ptr ITTNOTIFY_NAME(model_lock_acquire)
#define __itt_model_lock_release     ITTNOTIFY_VOID(model_lock_release)
#define __itt_model_lock_release_ptr ITTNOTIFY_NAME(model_lock_release)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_model_lock_acquire(lock)
#define __itt_model_lock_acquire_ptr 0
#define __itt_model_lock_release(lock)
#define __itt_model_lock_release_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_model_lock_acquire_ptr 0
#define __itt_model_lock_release_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief ANNOTATE_RECORD_ALLOCATION/ANNOTATE_RECORD_DEALLOCATION support
 *
 * record_allocation/deallocation describe user-defined memory allocator
 * behavior, which may be required for correctness modeling to understand
 * when storage is not expected to be actually reused across threads.
 */
void ITTAPI __itt_model_record_allocation  (void *addr, size_t size);
void ITTAPI __itt_model_record_deallocation(void *addr);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, model_record_allocation,   (void *addr, size_t size), (addr, size))
ITT_STUBV(ITTAPI, void, model_record_deallocation, (void *addr),              (addr))
#define __itt_model_record_allocation       ITTNOTIFY_VOID(model_record_allocation)
#define __itt_model_record_allocation_ptr   ITTNOTIFY_NAME(model_record_allocation)
#define __itt_model_record_deallocation     ITTNOTIFY_VOID(model_record_deallocation)
#define __itt_model_record_deallocation_ptr ITTNOTIFY_NAME(model_record_deallocation)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_model_record_allocation(addr, size)
#define __itt_model_record_allocation_ptr   0
#define __itt_model_record_deallocation(addr)
#define __itt_model_record_deallocation_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_model_record_allocation_ptr   0
#define __itt_model_record_deallocation_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief ANNOTATE_INDUCTION_USES support
 *
 * Note particular storage is inductive through the end of the current site
 */
void ITTAPI __itt_model_induction_uses(void* addr, size_t size);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, model_induction_uses, (void *addr, size_t size), (addr, size))
#define __itt_model_induction_uses     ITTNOTIFY_VOID(model_induction_uses)
#define __itt_model_induction_uses_ptr ITTNOTIFY_NAME(model_induction_uses)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_model_induction_uses(addr, size)
#define __itt_model_induction_uses_ptr   0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_model_induction_uses_ptr   0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief ANNOTATE_REDUCTION_USES support
 *
 * Note particular storage is used for reduction through the end
 * of the current site
 */
void ITTAPI __itt_model_reduction_uses(void* addr, size_t size);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, model_reduction_uses, (void *addr, size_t size), (addr, size))
#define __itt_model_reduction_uses     ITTNOTIFY_VOID(model_reduction_uses)
#define __itt_model_reduction_uses_ptr ITTNOTIFY_NAME(model_reduction_uses)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_model_reduction_uses(addr, size)
#define __itt_model_reduction_uses_ptr   0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_model_reduction_uses_ptr   0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief ANNOTATE_OBSERVE_USES support
 *
 * Have correctness modeling record observations about uses of storage
 * through the end of the current site
 */
void ITTAPI __itt_model_observe_uses(void* addr, size_t size);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, model_observe_uses, (void *addr, size_t size), (addr, size))
#define __itt_model_observe_uses     ITTNOTIFY_VOID(model_observe_uses)
#define __itt_model_observe_uses_ptr ITTNOTIFY_NAME(model_observe_uses)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_model_observe_uses(addr, size)
#define __itt_model_observe_uses_ptr   0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_model_observe_uses_ptr   0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief ANNOTATE_CLEAR_USES support
 *
 * Clear the special handling of a piece of storage related to induction,
 * reduction or observe_uses
 */
void ITTAPI __itt_model_clear_uses(void* addr);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, model_clear_uses, (void *addr), (addr))
#define __itt_model_clear_uses     ITTNOTIFY_VOID(model_clear_uses)
#define __itt_model_clear_uses_ptr ITTNOTIFY_NAME(model_clear_uses)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_model_clear_uses(addr)
#define __itt_model_clear_uses_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_model_clear_uses_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief ANNOTATE_DISABLE_*_PUSH/ANNOTATE_DISABLE_*_POP support
 *
 * disable_push/disable_pop push and pop disabling based on a parameter.
 * Disabling observations stops processing of memory references during
 * correctness modeling, and all annotations that occur in the disabled
 * region.  This allows description of code that is expected to be handled
 * specially during conversion to parallelism or that is not recognized
 * by tools (e.g. some kinds of synchronization operations.)
 * This mechanism causes all annotations in the disabled region, other
 * than disable_push and disable_pop, to be ignored.  (For example, this
 * might validly be used to disable an entire parallel site and the contained
 * tasks and locking in it for data collection purposes.)
 * The disable for collection is a more expensive operation, but reduces
 * collector overhead significantly.  This applies to BOTH correctness data
 * collection and performance data collection.  For example, a site
 * containing a task might only enable data collection for the first 10
 * iterations.  Both performance and correctness data should reflect this,
 * and the program should run as close to full speed as possible when
 * collection is disabled.
 */
void ITTAPI __itt_model_disable_push(__itt_model_disable x);
void ITTAPI __itt_model_disable_pop(void);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, model_disable_push, (__itt_model_disable x), (x))
ITT_STUBV(ITTAPI, void, model_disable_pop,  (void),                  ())
#define __itt_model_disable_push     ITTNOTIFY_VOID(model_disable_push)
#define __itt_model_disable_push_ptr ITTNOTIFY_NAME(model_disable_push)
#define __itt_model_disable_pop      ITTNOTIFY_VOID(model_disable_pop)
#define __itt_model_disable_pop_ptr  ITTNOTIFY_NAME(model_disable_pop)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_model_disable_push(x)
#define __itt_model_disable_push_ptr 0
#define __itt_model_disable_pop()
#define __itt_model_disable_pop_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_model_disable_push_ptr 0
#define __itt_model_disable_pop_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */
/** @} model group */

/**
 * @defgroup frames Frames
 * @ingroup public
 * Frames group
 * @{
 */
/**
 * @brief opaque structure for frame identification
 */
typedef struct __itt_frame_t *__itt_frame;

/**
 * @brief Create a global frame with given domain
 */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
__itt_frame ITTAPI __itt_frame_createA(const char    *domain);
__itt_frame ITTAPI __itt_frame_createW(const wchar_t *domain);
#if defined(UNICODE) || defined(_UNICODE)
#  define __itt_frame_create     __itt_frame_createW
#  define __itt_frame_create_ptr __itt_frame_createW_ptr
#else /* UNICODE */
#  define __itt_frame_create     __itt_frame_createA
#  define __itt_frame_create_ptr __itt_frame_createA_ptr
#endif /* UNICODE */
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
__itt_frame ITTAPI __itt_frame_create(const char *domain);
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUB(ITTAPI, __itt_frame, frame_createA, (const char    *domain), (domain))
ITT_STUB(ITTAPI, __itt_frame, frame_createW, (const wchar_t *domain), (domain))
#else  /* ITT_PLATFORM==ITT_PLATFORM_WIN */
ITT_STUB(ITTAPI, __itt_frame, frame_create,  (const char *domain), (domain))
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_frame_createA     ITTNOTIFY_DATA(frame_createA)
#define __itt_frame_createA_ptr ITTNOTIFY_NAME(frame_createA)
#define __itt_frame_createW     ITTNOTIFY_DATA(frame_createW)
#define __itt_frame_createW_ptr ITTNOTIFY_NAME(frame_createW)
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_frame_create     ITTNOTIFY_DATA(frame_create)
#define __itt_frame_create_ptr ITTNOTIFY_NAME(frame_create)
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#else  /* INTEL_NO_ITTNOTIFY_API */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_frame_createA(domain)
#define __itt_frame_createA_ptr 0
#define __itt_frame_createW(domain)
#define __itt_frame_createW_ptr 0
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_frame_create(domain)
#define __itt_frame_create_ptr  0
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_frame_createA_ptr 0
#define __itt_frame_createW_ptr 0
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_frame_create_ptr  0
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/** @brief Record an frame begin occurrence. */
void ITTAPI __itt_frame_begin(__itt_frame frame);
/** @brief Record an frame end occurrence. */
void ITTAPI __itt_frame_end  (__itt_frame frame);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, frame_begin, (__itt_frame frame), (frame))
ITT_STUBV(ITTAPI, void, frame_end,   (__itt_frame frame), (frame))
#define __itt_frame_begin     ITTNOTIFY_VOID(frame_begin)
#define __itt_frame_begin_ptr ITTNOTIFY_NAME(frame_begin)
#define __itt_frame_end       ITTNOTIFY_VOID(frame_end)
#define __itt_frame_end_ptr   ITTNOTIFY_NAME(frame_end)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_frame_begin(frame)
#define __itt_frame_begin_ptr 0
#define __itt_frame_end(frame)
#define __itt_frame_end_ptr   0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_frame_begin_ptr 0
#define __itt_frame_end_ptr   0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */
/** @} frames group */

/**
 * @defgroup events Events
 * @ingroup public
 * Events group
 * @{
 */
/** @brief user event type */
typedef int __itt_event;

/**
 * @brief Create an event notification
 * @note name or namelen being null/name and namelen not matching, user event feature not enabled
 * @return non-zero event identifier upon success and __itt_err otherwise
 */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
__itt_event LIBITTAPI __itt_event_createA(const char    *name, int namelen);
__itt_event LIBITTAPI __itt_event_createW(const wchar_t *name, int namelen);
#if defined(UNICODE) || defined(_UNICODE)
#  define __itt_event_create     __itt_event_createW
#  define __itt_event_create_ptr __itt_event_createW_ptr
#else
#  define __itt_event_create     __itt_event_createA
#  define __itt_event_create_ptr __itt_event_createA_ptr
#endif /* UNICODE */
#else  /* ITT_PLATFORM==ITT_PLATFORM_WIN */
__itt_event LIBITTAPI __itt_event_create(const char *name, int namelen);
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUB(LIBITTAPI, __itt_event, event_createA, (const char    *name, int namelen), (name, namelen))
ITT_STUB(LIBITTAPI, __itt_event, event_createW, (const wchar_t *name, int namelen), (name, namelen))
#else  /* ITT_PLATFORM==ITT_PLATFORM_WIN */
ITT_STUB(LIBITTAPI, __itt_event, event_create,  (const char *name, int namelen), (name, namelen))
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_event_createA     ITTNOTIFY_DATA(event_createA)
#define __itt_event_createA_ptr ITTNOTIFY_NAME(event_createA)
#define __itt_event_createW     ITTNOTIFY_DATA(event_createW)
#define __itt_event_createW_ptr ITTNOTIFY_NAME(event_createW)
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_event_create      ITTNOTIFY_DATA(event_create)
#define __itt_event_create_ptr  ITTNOTIFY_NAME(event_create)
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#else  /* INTEL_NO_ITTNOTIFY_API */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_event_createA(name, namelen) (__itt_event)0
#define __itt_event_createA_ptr 0
#define __itt_event_createW(name, namelen) (__itt_event)0
#define __itt_event_createW_ptr 0
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_event_create(name, namelen)  (__itt_event)0
#define __itt_event_create_ptr  0
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_event_createA_ptr 0
#define __itt_event_createW_ptr 0
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_event_create_ptr  0
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Record an event occurrence.
 * @return __itt_err upon failure (invalid event id/user event feature not enabled)
 */
int LIBITTAPI __itt_event_start(__itt_event event);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUB(LIBITTAPI, int, event_start, (__itt_event event), (event))
#define __itt_event_start     ITTNOTIFY_DATA(event_start)
#define __itt_event_start_ptr ITTNOTIFY_NAME(event_start)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_event_start(event) (int)0
#define __itt_event_start_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_event_start_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Record an event end occurrence.
 * @note It is optional if events do not have durations.
 * @return __itt_err upon failure (invalid event id/user event feature not enabled)
 */
int LIBITTAPI __itt_event_end(__itt_event event);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUB(LIBITTAPI, int, event_end, (__itt_event event), (event))
#define __itt_event_end     ITTNOTIFY_DATA(event_end)
#define __itt_event_end_ptr ITTNOTIFY_NAME(event_end)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_event_end(event) (int)0
#define __itt_event_end_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_event_end_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */
/** @} events group */

/**
 * @defgroup heap Heap
 * @ingroup public
 * Heap group
 * @{
 */

typedef void* __itt_heap_function;

/**
 * @brief Create an identification for heap function
 * @return non-zero identifier or NULL
 */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
__itt_heap_function ITTAPI __itt_heap_function_createA(const char*    name, const char*    domain);
__itt_heap_function ITTAPI __itt_heap_function_createW(const wchar_t* name, const wchar_t* domain);
#if defined(UNICODE) || defined(_UNICODE)
#  define __itt_heap_function_create     __itt_heap_function_createW
#  define __itt_heap_function_create_ptr __itt_heap_function_createW_ptr
#else
#  define __itt_heap_function_create     __itt_heap_function_createA
#  define __itt_heap_function_create_ptr __itt_heap_function_createA_ptr
#endif /* UNICODE */
#else  /* ITT_PLATFORM==ITT_PLATFORM_WIN */
__itt_heap_function ITTAPI __itt_heap_function_create(const char* name, const char* domain);
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUB(ITTAPI, __itt_heap_function, heap_function_createA, (const char*    name, const char*    domain), (name, domain))
ITT_STUB(ITTAPI, __itt_heap_function, heap_function_createW, (const wchar_t* name, const wchar_t* domain), (name, domain))
#else  /* ITT_PLATFORM==ITT_PLATFORM_WIN */
ITT_STUB(ITTAPI, __itt_heap_function, heap_function_create,  (const char*    name, const char*    domain), (name, domain))
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_heap_function_createA     ITTNOTIFY_DATA(heap_function_createA)
#define __itt_heap_function_createA_ptr ITTNOTIFY_NAME(heap_function_createA)
#define __itt_heap_function_createW     ITTNOTIFY_DATA(heap_function_createW)
#define __itt_heap_function_createW_ptr ITTNOTIFY_NAME(heap_function_createW)
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_heap_function_create      ITTNOTIFY_DATA(heap_function_create)
#define __itt_heap_function_create_ptr  ITTNOTIFY_NAME(heap_function_create)
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#else  /* INTEL_NO_ITTNOTIFY_API */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_heap_function_createA(name, domain) (__itt_heap_function)0
#define __itt_heap_function_createA_ptr 0
#define __itt_heap_function_createW(name, domain) (__itt_heap_function)0
#define __itt_heap_function_createW_ptr 0
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_heap_function_create(name, domain)  (__itt_heap_function)0
#define __itt_heap_function_create_ptr  0
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_heap_function_createA_ptr 0
#define __itt_heap_function_createW_ptr 0
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_heap_function_create_ptr  0
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Record an allocation begin occurrence.
 */
void ITTAPI __itt_heap_allocate_begin(__itt_heap_function h, size_t size, int initialized);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, heap_allocate_begin, (__itt_heap_function h, size_t size, int initialized), (h, size, initialized))
#define __itt_heap_allocate_begin     ITTNOTIFY_VOID(heap_allocate_begin)
#define __itt_heap_allocate_begin_ptr ITTNOTIFY_NAME(heap_allocate_begin)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_heap_allocate_begin(h, size, initialized)
#define __itt_heap_allocate_begin_ptr   0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_heap_allocate_begin_ptr   0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Record an allocation end occurrence.
 */
void ITTAPI __itt_heap_allocate_end(__itt_heap_function h, void** addr, size_t size, int initialized);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, heap_allocate_end, (__itt_heap_function h, void** addr, size_t size, int initialized), (h, addr, size, initialized))
#define __itt_heap_allocate_end     ITTNOTIFY_VOID(heap_allocate_end)
#define __itt_heap_allocate_end_ptr ITTNOTIFY_NAME(heap_allocate_end)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_heap_allocate_end(h, addr, size, initialized)
#define __itt_heap_allocate_end_ptr   0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_heap_allocate_end_ptr   0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Record an free begin occurrence.
 */
void ITTAPI __itt_heap_free_begin(__itt_heap_function h, void* addr);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, heap_free_begin, (__itt_heap_function h, void* addr), (h, addr))
#define __itt_heap_free_begin     ITTNOTIFY_VOID(heap_free_begin)
#define __itt_heap_free_begin_ptr ITTNOTIFY_NAME(heap_free_begin)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_heap_free_begin(h, addr)
#define __itt_heap_free_begin_ptr   0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_heap_free_begin_ptr   0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Record an free end occurrence.
 */
void ITTAPI __itt_heap_free_end(__itt_heap_function h, void* addr);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, heap_free_end, (__itt_heap_function h, void* addr), (h, addr))
#define __itt_heap_free_end     ITTNOTIFY_VOID(heap_free_end)
#define __itt_heap_free_end_ptr ITTNOTIFY_NAME(heap_free_end)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_heap_free_end(h, addr)
#define __itt_heap_free_end_ptr   0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_heap_free_end_ptr   0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Record an reallocation begin occurrence.
 */
void ITTAPI __itt_heap_reallocate_begin(__itt_heap_function h, void* addr, size_t new_size, int initialized);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, heap_reallocate_begin, (__itt_heap_function h, void* addr, size_t new_size, int initialized), (h, addr, new_size, initialized))
#define __itt_heap_reallocate_begin     ITTNOTIFY_VOID(heap_reallocate_begin)
#define __itt_heap_reallocate_begin_ptr ITTNOTIFY_NAME(heap_reallocate_begin)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_heap_reallocate_begin(h, addr, new_size, initialized)
#define __itt_heap_reallocate_begin_ptr   0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_heap_reallocate_begin_ptr   0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Record an reallocation end occurrence.
 */
void ITTAPI __itt_heap_reallocate_end(__itt_heap_function h, void* addr, void** new_addr, size_t new_size, int initialized);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, heap_reallocate_end, (__itt_heap_function h, void* addr, void** new_addr, size_t new_size, int initialized), (h, addr, new_addr, new_size, initialized))
#define __itt_heap_reallocate_end     ITTNOTIFY_VOID(heap_reallocate_end)
#define __itt_heap_reallocate_end_ptr ITTNOTIFY_NAME(heap_reallocate_end)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_heap_reallocate_end(h, addr, new_addr, new_size, initialized)
#define __itt_heap_reallocate_end_ptr   0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_heap_reallocate_end_ptr   0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/** @brief internal access begin */
void ITTAPI __itt_heap_internal_access_begin(void);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, heap_internal_access_begin,  (void), ())
#define __itt_heap_internal_access_begin      ITTNOTIFY_VOID(heap_internal_access_begin)
#define __itt_heap_internal_access_begin_ptr  ITTNOTIFY_NAME(heap_internal_access_begin)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_heap_internal_access_begin()
#define __itt_heap_internal_access_begin_ptr  0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_heap_internal_access_begin_ptr  0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/** @brief internal access end */
void ITTAPI __itt_heap_internal_access_end(void);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, heap_internal_access_end, (void), ())
#define __itt_heap_internal_access_end     ITTNOTIFY_VOID(heap_internal_access_end)
#define __itt_heap_internal_access_end_ptr ITTNOTIFY_NAME(heap_internal_access_end)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_heap_internal_access_end()
#define __itt_heap_internal_access_end_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_heap_internal_access_end_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */
/** @} heap group */

/** @cond exclude_from_documentation */
#ifdef __cplusplus
}
#endif /* __cplusplus */
/** @endcond */

#endif /* _ITTNOTIFY_H_ */
