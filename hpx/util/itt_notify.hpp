//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ITT_NOTIFY_AUG_17_2010_1237PM)
#define HPX_UTIL_ITT_NOTIFY_AUG_17_2010_1237PM

#include <hpx/config.hpp>

///////////////////////////////////////////////////////////////////////////////
#if defined(HPX_USE_ITT)

#define ITT_EXPORT HPX_EXPORT

#include <hpx/util/tools_api/ittnotify.h>
#include <hpx/util/tools_api/internal/ittnotify.h>

///////////////////////////////////////////////////////////////////////////////
#define HPX_ITT_SYNC_CREATE(obj, type, name)                                  \
    if (__itt_sync_create_ptr)                                                \
        __itt_sync_create(obj, type, name, __itt_attr_mutex)                  \
    /**/
#define HPX_ITT_SYNC(name, obj)                                               \
    if (__itt_ ## name ## _ptr) {                                             \
        __itt_ ## name ## _ptr(                                               \
            const_cast<void*>(static_cast<volatile void*>(obj)));             \
    }                                                                         \
    /**/
#define HPX_ITT_SYNC_SET_NAME(obj, type, name)                                \
    if (__itt_sync_set_name_ptr) {                                            \
        __itt_sync_set_name_ptr(                                              \
            const_cast<void*>(static_cast<volatile void*>(obj)),              \
                type, name, __itt_attr_mutex);                                \
    }                                                                         \
    /**/

#define HPX_ITT_THREAD_SET_NAME(name)                                         \
    if (__itt_thread_set_name_ptr) __itt_thread_set_name_ptr(name)            \
    /**/

// #define HPX_ITT_STACK_CREATE(obj)             obj = __itt_stack_caller_create()
// #define HPX_ITT_STACK(name, obj)              __itt_stack_ ## name(obj)

namespace hpx { namespace util
{
    HPX_EXPORT void init_itt_api();
    HPX_EXPORT void deinit_itt_api();
}}

#else

#define HPX_ITT_SYNC_CREATE(obj, type, name)    ((void)0)
#define HPX_ITT_SYNC(name, obj)                 ((void)0)
#define HPX_ITT_SYNC_SET_NAME(obj, type, name)  ((void)0)

#define HPX_ITT_THREAD_SET_NAME(name)           ((void)0)
// #define HPX_ITT_SYNC_CREATE(obj, type, name)  ((void)0)
// #define HPX_ITT_STACK_CREATE(obj)             ((void)0)
// #define HPX_ITT_STACK(name, obj)              ((void)0)

namespace hpx { namespace util
{
    inline void init_itt_api() {}
    inline void deinit_itt_api() {}
}}

#endif

#define HPX_ITT_SYNC_PREPARE(obj)           HPX_ITT_SYNC(sync_prepare, obj)
#define HPX_ITT_SYNC_CANCEL(obj)            HPX_ITT_SYNC(sync_cancel, obj)
#define HPX_ITT_SYNC_ACQUIRED(obj)          HPX_ITT_SYNC(sync_acquired, obj)
#define HPX_ITT_SYNC_RELEASING(obj)         HPX_ITT_SYNC(sync_releasing, obj)
#define HPX_ITT_SYNC_RELEASED(obj)          HPX_ITT_SYNC(sync_released, obj)
#define HPX_ITT_SYNC_DESTROY(obj)           HPX_ITT_SYNC(sync_destroy, obj)

// #define HPX_ITT_STACK_ENTER(obj)              HPX_ITT_STACK(callee_enter, obj)
// #define HPX_ITT_STACK_LEAVE(obj)              HPX_ITT_STACK(callee_leave, obj)
// #define HPX_ITT_STACK_DESTROY(obj)            HPX_ITT_STACK(caller_destroy, obj)

#endif
