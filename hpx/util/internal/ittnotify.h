/*
    Copyright 2005-2010 Intel Corporation.  All Rights Reserved.

    This file is part of Threading Building Blocks.

    Threading Building Blocks is free software; you can redistribute it
    and/or modify it under the terms of the GNU General Public License
    version 2 as published by the Free Software Foundation.

    Threading Building Blocks is distributed in the hope that it will be
    useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Threading Building Blocks; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

    As a special exception, you may use this file as part of a free software
    library without restriction.  Specifically, if other files instantiate
    templates or use macros or inline functions from this file, or you compile
    this file and link it with other files to produce an executable, this
    file does not by itself cause the resulting executable to be covered by
    the GNU General Public License.  This exception does not however
    invalidate any other reasons why the executable file might be covered by
    the GNU General Public License.
*/

#ifndef _INTERNAL_ITTNOTIFY_H_
#define _INTERNAL_ITTNOTIFY_H_
/**
 * @file
 * @brief Internal User API functions and types
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

#ifndef ITTAPI
#define ITTAPI    CDECL
#endif
#ifndef LIBITTAPI
#define LIBITTAPI /* nothing */
#endif

#define ITT_JOIN_AUX(p,n) p##n
#define ITT_JOIN(p,n)     ITT_JOIN_AUX(p,n)

#ifndef INTEL_ITTNOTIFY_PREFIX
#  define INTEL_ITTNOTIFY_PREFIX __itt_
#endif /* INTEL_ITTNOTIFY_PREFIX */
#ifndef INTEL_ITTNOTIFY_POSTFIX
#  define INTEL_ITTNOTIFY_POSTFIX _ptr_
#endif /* INTEL_ITTNOTIFY_POSTFIX */

#ifndef ITTNOTIFY_NAME
#define ITTNOTIFY_NAME_AUX(n) ITT_JOIN(INTEL_ITTNOTIFY_PREFIX,n)
#define ITTNOTIFY_NAME(n)     ITTNOTIFY_NAME_AUX(ITT_JOIN(n,INTEL_ITTNOTIFY_POSTFIX))
#endif

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
 * @defgroup internal Internal API
 * @{
 * @}
 */

/**
 * @defgroup makrs Marks
 * @ingroup internal
 * Marks group
 * @warning Internal API:
 *   - It is not shipped to outside of Intel
 *   - It is delivered to internal Intel teams using e-mail or SVN access only
 * @{
 */
/** @brief user mark type */
typedef int __itt_mark_type;

/**
 * @brief Creates a user mark type with the specified name using char or Unicode string.
 * @param[in] name - name of mark to create
 * @return Returns a handle to the mark type
 */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
__itt_mark_type ITTAPI __itt_mark_createA(const char    *name);
__itt_mark_type ITTAPI __itt_mark_createW(const wchar_t *name);
#ifdef UNICODE
#  define __itt_mark_create     __itt_mark_createW
#  define __itt_mark_create_ptr __itt_mark_createW_ptr
#else /* UNICODE */
#  define __itt_mark_create     __itt_mark_createA
#  define __itt_mark_create_ptr __itt_mark_createA_ptr
#endif /* UNICODE */
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
__itt_mark_type ITTAPI __itt_mark_create(const char *name);
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUB(ITTAPI, __itt_mark_type, mark_createA, (const char    *name), (name))
ITT_STUB(ITTAPI, __itt_mark_type, mark_createW, (const wchar_t *name), (name))
#else  /* ITT_PLATFORM==ITT_PLATFORM_WIN */
ITT_STUB(ITTAPI, __itt_mark_type, mark_create,  (const char *name), (name))
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_mark_createA     ITTNOTIFY_DATA(mark_createA)
#define __itt_mark_createA_ptr ITTNOTIFY_NAME(mark_createA)
#define __itt_mark_createW     ITTNOTIFY_DATA(mark_createW)
#define __itt_mark_createW_ptr ITTNOTIFY_NAME(mark_createW)
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_mark_create      ITTNOTIFY_DATA(mark_create)
#define __itt_mark_create_ptr  ITTNOTIFY_NAME(mark_create)
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#else  /* INTEL_NO_ITTNOTIFY_API */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_mark_createA(name) (__itt_mark_type)0
#define __itt_mark_createA_ptr 0
#define __itt_mark_createW(name) (__itt_mark_type)0
#define __itt_mark_createW_ptr 0
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_mark_create(name)  (__itt_mark_type)0
#define __itt_mark_create_ptr  0
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_mark_createA_ptr 0
#define __itt_mark_createW_ptr 0
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_mark_create_ptr  0
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Creates a "discrete" user mark type of the specified type and an optional parameter using char or Unicode string.
 *
 * - The mark of "discrete" type is placed to collection results in case of success. It appears in overtime view(s) as a special tick sign.
 * - The call is "synchronous" - function returns after mark is actually added to results.
 * - This function is useful, for example, to mark different phases of application
 *   (beginning of the next mark automatically meand end of current region).
 * - Can be used together with "continuous" marks (see below) at the same collection session
 * @param[in] mt - mark, created by __itt_mark_create(const char* name) function
 * @param[in] parameter - string parameter of mark
 * @return Returns zero value in case of success, non-zero value otherwise.
 */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
int ITTAPI __itt_markA(__itt_mark_type mt, const char    *parameter);
int ITTAPI __itt_markW(__itt_mark_type mt, const wchar_t *parameter);
#ifdef UNICODE
#  define __itt_mark     __itt_markW
#  define __itt_mark_ptr __itt_markW_ptr
#else /* UNICODE  */
#  define __itt_mark     __itt_markA
#  define __itt_mark_ptr __itt_markA_ptr
#endif /* UNICODE */
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
int ITTAPI __itt_mark(__itt_mark_type mt, const char *parameter);
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUB(ITTAPI, int, markA, (__itt_mark_type mt, const char    *parameter), (mt, parameter))
ITT_STUB(ITTAPI, int, markW, (__itt_mark_type mt, const wchar_t *parameter), (mt, parameter))
#else  /* ITT_PLATFORM==ITT_PLATFORM_WIN */
ITT_STUB(ITTAPI, int, mark,  (__itt_mark_type mt, const char *parameter), (mt, parameter))
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_markA     ITTNOTIFY_DATA(markA)
#define __itt_markA_ptr ITTNOTIFY_NAME(markA)
#define __itt_markW     ITTNOTIFY_DATA(markW)
#define __itt_markW_ptr ITTNOTIFY_NAME(markW)
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_mark      ITTNOTIFY_DATA(mark)
#define __itt_mark_ptr  ITTNOTIFY_NAME(mark)
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#else  /* INTEL_NO_ITTNOTIFY_API */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_markA(mt, parameter) (int)0
#define __itt_markA_ptr 0
#define __itt_markW(mt, parameter) (int)0
#define __itt_markW_ptr 0
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_mark(mt, parameter)  (int)0
#define __itt_mark_ptr  0
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_markA_ptr 0
#define __itt_markW_ptr 0
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_mark_ptr  0
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Use this if necessary to create a "discrete" user event type (mark) for process
 * rather then for one thread
 * @see int __itt_mark(__itt_mark_type mt, const char* parameter);
 */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
int ITTAPI __itt_mark_globalA(__itt_mark_type mt, const char    *parameter);
int ITTAPI __itt_mark_globalW(__itt_mark_type mt, const wchar_t *parameter);
#ifdef UNICODE
#  define __itt_mark_global     __itt_mark_globalW
#  define __itt_mark_global_ptr __itt_mark_globalW_ptr
#else /* UNICODE  */
#  define __itt_mark_global     __itt_mark_globalA
#  define __itt_mark_global_ptr __itt_mark_globalA_ptr
#endif /* UNICODE */
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
int ITTAPI __itt_mark_global(__itt_mark_type mt, const char *parameter);
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUB(ITTAPI, int, mark_globalA, (__itt_mark_type mt, const char    *parameter), (mt, parameter))
ITT_STUB(ITTAPI, int, mark_globalW, (__itt_mark_type mt, const wchar_t *parameter), (mt, parameter))
#else  /* ITT_PLATFORM==ITT_PLATFORM_WIN */
ITT_STUB(ITTAPI, int, mark_global,  (__itt_mark_type mt, const char *parameter), (mt, parameter))
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_mark_globalA     ITTNOTIFY_DATA(mark_globalA)
#define __itt_mark_globalA_ptr ITTNOTIFY_NAME(mark_globalA)
#define __itt_mark_globalW     ITTNOTIFY_DATA(mark_globalW)
#define __itt_mark_globalW_ptr ITTNOTIFY_NAME(mark_globalW)
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_mark_global      ITTNOTIFY_DATA(mark_global)
#define __itt_mark_global_ptr  ITTNOTIFY_NAME(mark_global)
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#else  /* INTEL_NO_ITTNOTIFY_API */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_mark_globalA(mt, parameter) (int)0
#define __itt_mark_globalA_ptr 0
#define __itt_mark_globalW(mt, parameter) (int)0
#define __itt_mark_globalW_ptr 0
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_mark_global(mt, parameter)  (int)0
#define __itt_mark_global_ptr  0
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_mark_globalA_ptr 0
#define __itt_mark_globalW_ptr 0
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_mark_global_ptr  0
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Creates an "end" point for "continuous" mark with specified name.
 *
 * - Returns zero value in case of success, non-zero value otherwise.
 *   Also returns non-zero value when preceding "begin" point for the
 *   mark with the same name failed to be created or not created.
 * - The mark of "continuous" type is placed to collection results in
 *   case of success. It appears in overtime view(s) as a special tick
 *   sign (different from "discrete" mark) together with line from
 *   corresponding "begin" mark to "end" mark.
 * @note Continuous marks can overlap and be nested inside each other.
 * Discrete mark can be nested inside marked region
 * @param[in] mt - mark, created by __itt_mark_create(const char* name) function
 * @return Returns zero value in case of success, non-zero value otherwise.
 */
int ITTAPI __itt_mark_off(__itt_mark_type mt);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUB(ITTAPI, int, mark_off, (__itt_mark_type mt), (mt))
#define __itt_mark_off     ITTNOTIFY_DATA(mark_off)
#define __itt_mark_off_ptr ITTNOTIFY_NAME(mark_off)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_mark_off(mt) (int)0
#define __itt_mark_off_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_mark_off_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Use this if necessary to create an "end" point for mark of process
 * @see int __itt_mark_off(__itt_mark_type mt);
 */
int ITTAPI __itt_mark_global_off(__itt_mark_type mt);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUB(ITTAPI, int, mark_global_off, (__itt_mark_type mt), (mt))
#define __itt_mark_global_off     ITTNOTIFY_DATA(mark_global_off)
#define __itt_mark_global_off_ptr ITTNOTIFY_NAME(mark_global_off)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_mark_global_off(mt) (int)0
#define __itt_mark_global_off_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_mark_global_off_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */
/** @} marks group */

/**
 * @defgroup counters Counters
 * @ingroup internal
 * Counters group
 * @{
 */
/**
 * @brief opaque structure for counter identification
 */
typedef struct ___itt_counter *__itt_counter;

/**
 * @brief Create a counter with given name/domain for the calling thread
 *
 * After __itt_counter_create() is called, __itt_counter_inc() / __itt_counter_inc_delta() can be used
 * to increment the counter on any thread
 */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
__itt_counter ITTAPI __itt_counter_createA(const char    *name, const char    *domain);
__itt_counter ITTAPI __itt_counter_createW(const wchar_t *name, const wchar_t *domain);
#ifdef UNICODE
#  define __itt_counter_create     __itt_counter_createW
#  define __itt_counter_create_ptr __itt_counter_createW_ptr
#else /* UNICODE */
#  define __itt_counter_create     __itt_counter_createA
#  define __itt_counter_create_ptr __itt_counter_createA_ptr
#endif /* UNICODE */
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
__itt_counter ITTAPI __itt_counter_create(const char *name, const char *domain);
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#if ITT_PLATFORM==ITT_PLATFORM_WIN
ITT_STUB(ITTAPI, __itt_counter, counter_createA, (const char    *name, const char    *domain), (name, domain))
ITT_STUB(ITTAPI, __itt_counter, counter_createW, (const wchar_t *name, const wchar_t *domain), (name, domain))
#else  /* ITT_PLATFORM==ITT_PLATFORM_WIN */
ITT_STUB(ITTAPI, __itt_counter, counter_create,  (const char *name, const char *domain), (name, domain))
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_counter_createA     ITTNOTIFY_DATA(counter_createA)
#define __itt_counter_createA_ptr ITTNOTIFY_NAME(counter_createA)
#define __itt_counter_createW     ITTNOTIFY_DATA(counter_createW)
#define __itt_counter_createW_ptr ITTNOTIFY_NAME(counter_createW)
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_counter_create     ITTNOTIFY_DATA(counter_create)
#define __itt_counter_create_ptr ITTNOTIFY_NAME(counter_create)
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#else  /* INTEL_NO_ITTNOTIFY_API */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_counter_createA(name, domain)
#define __itt_counter_createA_ptr 0
#define __itt_counter_createW(name, domain)
#define __itt_counter_createW_ptr 0
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_counter_create(name, domain)
#define __itt_counter_create_ptr  0
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#if ITT_PLATFORM==ITT_PLATFORM_WIN
#define __itt_counter_createA_ptr 0
#define __itt_counter_createW_ptr 0
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#define __itt_counter_create_ptr  0
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Destroy the counter identified by the pointer previously returned by __itt_counter_create()
 */
void ITTAPI __itt_counter_destroy(__itt_counter id);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, counter_destroy, (__itt_counter id), (id))
#define __itt_counter_destroy     ITTNOTIFY_VOID(counter_destroy)
#define __itt_counter_destroy_ptr ITTNOTIFY_NAME(counter_destroy)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_counter_destroy(id)
#define __itt_counter_destroy_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_counter_destroy_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Increment the counter value
 */
void ITTAPI __itt_counter_inc(__itt_counter id);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, counter_inc, (__itt_counter id), (id))
#define __itt_counter_inc     ITTNOTIFY_VOID(counter_inc)
#define __itt_counter_inc_ptr ITTNOTIFY_NAME(counter_inc)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_counter_inc(id)
#define __itt_counter_inc_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_counter_inc_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Increment the counter value with x
 */
void ITTAPI __itt_counter_inc_delta(__itt_counter id, unsigned long long value);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, counter_inc_delta, (__itt_counter id, unsigned long long value), (id, value))
#define __itt_counter_inc_delta     ITTNOTIFY_VOID(counter_inc_delta)
#define __itt_counter_inc_delta_ptr ITTNOTIFY_NAME(counter_inc_delta)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_counter_inc_delta(id, value)
#define __itt_counter_inc_delta_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_counter_inc_delta_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */
/** @} counters group */

/**
 * @defgroup stitch Stack Stitching
 * @ingroup internal
 * Stack Stitching group
 * @{
 */
/**
 * @brief opaque structure for counter identification
 */
typedef struct ___itt_caller *__itt_caller;

/**
 * @brief Create the stitch point e.g. a point in call stack where other stacks should be stitched to.
 * The function returns a unique identifier which is used to match the cut points with corresponding stitch points.
 */
__itt_caller ITTAPI __itt_stack_caller_create(void);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUB(ITTAPI, __itt_caller, stack_caller_create, (void), ())
#define __itt_stack_caller_create     ITTNOTIFY_DATA(stack_caller_create)
#define __itt_stack_caller_create_ptr ITTNOTIFY_NAME(stack_caller_create)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_stack_caller_create() (__itt_caller)0
#define __itt_stack_caller_create_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_stack_caller_create_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Destroy the inforamtion about stitch point identified by the pointer previously returned by __itt_stack_caller_create()
 */
void ITTAPI __itt_stack_caller_destroy(__itt_caller id);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, stack_caller_destroy, (__itt_caller id), (id))
#define __itt_stack_caller_destroy     ITTNOTIFY_VOID(stack_caller_destroy)
#define __itt_stack_caller_destroy_ptr ITTNOTIFY_NAME(stack_caller_destroy)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_stack_caller_destroy(id)
#define __itt_stack_caller_destroy_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_stack_caller_destroy_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief Sets the cut point. Stack from each event which occurs after this call will be cut
 * at the same stack level the function was called and stitched to the corresponding stitch point.
 */
void ITTAPI __itt_stack_callee_enter(__itt_caller id);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, stack_callee_enter, (__itt_caller id), (id))
#define __itt_stack_callee_enter     ITTNOTIFY_VOID(stack_callee_enter)
#define __itt_stack_callee_enter_ptr ITTNOTIFY_NAME(stack_callee_enter)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_stack_callee_enter(id)
#define __itt_stack_callee_enter_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_stack_callee_enter_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/**
 * @brief This function eliminates the cut point which was set by latest __itt_stack_callee_enter().
 */
void ITTAPI __itt_stack_callee_leave(__itt_caller id);

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
ITT_STUBV(ITTAPI, void, stack_callee_leave, (__itt_caller id), (id))
#define __itt_stack_callee_leave     ITTNOTIFY_VOID(stack_callee_leave)
#define __itt_stack_callee_leave_ptr ITTNOTIFY_NAME(stack_callee_leave)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_stack_callee_leave(id)
#define __itt_stack_callee_leave_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_stack_callee_leave_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/** @} stitch group */

/* ***************************************************************************************************************************** */

/** @cond exclude_from_documentation */
typedef enum __itt_error_code {
    __itt_error_success       = 0, /*!< no error */
    __itt_error_no_module     = 1, /*!< module can't be loaded */
    /* %1$s -- library name; win: %2$d -- system error code; unx: %2$s -- system error message. */
    __itt_error_no_symbol     = 2, /*!< symbol not found */
    /* %1$s -- library name, %2$s -- symbol name. */
    __itt_error_unknown_group = 3, /*!< unknown group specified */
    /* %1$s -- env var name, %2$s -- group name. */
    __itt_error_cant_read_env = 4, /*!< GetEnvironmentVariable() failed */
    /* %1$s -- env var name, %2$d -- system error. */
    __itt_error_env_too_long  = 5, /*!< variable value too long */
    /* %1$s -- env var name, %2$d -- actual length of the var, %3$d -- max allowed length. */
    __itt_error_system        = 6  /*!< pthread_mutexattr_init or pthread_mutex_init failed */
    /* %1$s -- function name, %2$d -- errno. */
} __itt_error_code;

typedef void (__itt_error_notification_t)(__itt_error_code code, va_list);
__itt_error_notification_t* __itt_set_error_handler(__itt_error_notification_t*);

const char* ITTAPI __itt_api_version(void);
/** @endcond */

/** @cond exclude_from_documentation */
#ifndef INTEL_NO_MACRO_BODY
#ifndef INTEL_NO_ITTNOTIFY_API
#define __itt_error_handler ITT_JOIN(INTEL_ITTNOTIFY_PREFIX, error_handler)
void __itt_error_handler(__itt_error_code code, va_list args);
extern const int ITTNOTIFY_NAME(err);
#define __itt_err ITTNOTIFY_NAME(err)
ITT_STUB(ITTAPI, const char*, api_version, (void), ())
#define __itt_api_version     ITTNOTIFY_DATA(api_version)
#define __itt_api_version_ptr ITTNOTIFY_NAME(api_version)
#else  /* INTEL_NO_ITTNOTIFY_API */
#define __itt_api_version()   (const char*)0
#define __itt_api_version_ptr 0
#endif /* INTEL_NO_ITTNOTIFY_API */
#else  /* INTEL_NO_MACRO_BODY */
#define __itt_api_version_ptr 0
#endif /* INTEL_NO_MACRO_BODY */
/** @endcond */

/** @cond exclude_from_documentation */
#ifdef __cplusplus
}
#endif /* __cplusplus */
/** @endcond */

#endif /* _INTERNAL_ITTNOTIFY_H_ */
