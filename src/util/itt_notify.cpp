//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/itt_notify.hpp>

#if HPX_USE_ITT != 0

#include <ittnotify.h>
#include <legacy/ittnotify.h>
#include <hpx/util/internal/ittnotify.h>

///////////////////////////////////////////////////////////////////////////////
// decide whether to use the ITT notify API if it's available
bool use_ittnotify_api = false;

///////////////////////////////////////////////////////////////////////////////
#define HPX_INTERNAL_ITT_SYNC_CREATE(obj, type, name)                         \
    if (use_ittnotify_api && __itt_sync_createA_ptr) {                        \
        __itt_sync_createA_ptr(                                               \
            const_cast<void*>(static_cast<volatile void*>(obj)),              \
                type, name, __itt_attr_mutex);                                \
    }                                                                         \
    /**/
#define HPX_INTERNAL_ITT_SYNC(fname, obj)                                     \
    if (use_ittnotify_api && __itt_ ## fname ## _ptr) {                       \
        __itt_ ## fname ## _ptr(                                              \
            const_cast<void*>(static_cast<volatile void*>(obj)));             \
    }                                                                         \
    /**/
#define HPX_INTERNAL_ITT_SYNC_RENAME(obj, name)                               \
    if (use_ittnotify_api && __itt_sync_renameA_ptr) {                        \
        __itt_sync_renameA_ptr(                                               \
            const_cast<void*>(static_cast<volatile void*>(obj)), name);       \
    }                                                                         \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_INTERNAL_ITT_STACK_CREATE()                                       \
    (use_ittnotify_api && __itt_stack_caller_create_ptr) ?                    \
        __itt_stack_caller_create_ptr() : (__itt_caller)0                     \
    /**/
#define HPX_INTERNAL_ITT_STACK_ENTER(ctx)                                     \
    if (use_ittnotify_api && __itt_stack_callee_enter_ptr)                    \
        __itt_stack_callee_enter_ptr(ctx);                                    \
    /**/
#define HPX_INTERNAL_ITT_STACK_LEAVE(ctx)                                     \
    if (use_ittnotify_api && __itt_stack_callee_leave_ptr)                    \
        __itt_stack_callee_leave_ptr(ctx);                                    \
    /**/
#define HPX_INTERNAL_ITT_STACK_DESTROY(ctx)                                   \
    if (use_ittnotify_api && __itt_stack_caller_destroy_ptr && ctx != (__itt_caller)0) \
        __itt_stack_caller_destroy_ptr(ctx);                                  \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_INTERNAL_ITT_FRAME_CREATE(name)                                   \
    (use_ittnotify_api && __itt_frame_create_ptr) ?                           \
        __itt_frame_create_ptr(name) : (__itt_frame)0                         \
    /**/
#define HPX_INTERNAL_ITT_FRAME_BEGIN(frame)                                   \
    if (use_ittnotify_api && __itt_frame_begin_ptr)                           \
        __itt_frame_begin_ptr(frame);                                         \
    /**/
#define HPX_INTERNAL_ITT_FRAME_END(frame)                                     \
    if (use_ittnotify_api && __itt_frame_end_ptr)                             \
        __itt_frame_end_ptr(frame);                                           \
    /**/
#define HPX_INTERNAL_ITT_FRAME_DESTROY(frame) ((void)0)

///////////////////////////////////////////////////////////////////////////////
#define HPX_INTERNAL_ITT_MARK_CREATE(name)                                    \
    (use_ittnotify_api && __itt_mark_create_ptr) ?                            \
        __itt_mark_create_ptr(name) : 0                                       \
    /**/
#define HPX_INTERNAL_ITT_MARK_OFF(mark)                                       \
    if (use_ittnotify_api && __itt_mark_off_ptr) __itt_mark_off_ptr(mark);    \
    /**/
#define HPX_INTERNAL_ITT_MARK(mark, parameter)                                \
    if (use_ittnotify_api && __itt_mark_ptr) __itt_mark_ptr(mark, parameter); \
    /**/

#define HPX_INTERNAL_ITT_THREAD_SET_NAME(name)                                \
    if (use_ittnotify_api && __itt_thread_set_name_ptr)                       \
        __itt_thread_set_name_ptr(name);                                      \
    /**/
#define HPX_INTERNAL_ITT_THREAD_IGNORE()                                      \
    if (use_ittnotify_api && __itt_thread_ignore_ptr)                         \
        __itt_thread_ignore_ptr();                                            \
    /**/

///////////////////////////////////////////////////////////////////////////////
#if defined(BOOST_MSVC) \
    || defined(__BORLANDC__) \
    || (defined(__MWERKS__) && defined(_WIN32) && (__MWERKS__ >= 0x3000)) \
    || (defined(__ICL) && defined(_MSC_EXTENSIONS) && (_MSC_VER >= 1200))

#pragma comment(lib, "libittnotify.lib")
#endif

///////////////////////////////////////////////////////////////////////////////
#define HPX_INTERNAL_ITT_SYNC_PREPARE(obj)           HPX_INTERNAL_ITT_SYNC(sync_prepare, obj)
#define HPX_INTERNAL_ITT_SYNC_CANCEL(obj)            HPX_INTERNAL_ITT_SYNC(sync_cancel, obj)
#define HPX_INTERNAL_ITT_SYNC_ACQUIRED(obj)          HPX_INTERNAL_ITT_SYNC(sync_acquired, obj)
#define HPX_INTERNAL_ITT_SYNC_RELEASING(obj)         HPX_INTERNAL_ITT_SYNC(sync_releasing, obj)
#define HPX_INTERNAL_ITT_SYNC_RELEASED(obj)          ((void)0) //HPX_INTERNAL_ITT_SYNC(sync_released, obj)
#define HPX_INTERNAL_ITT_SYNC_DESTROY(obj)           HPX_INTERNAL_ITT_SYNC(sync_destroy, obj)

///////////////////////////////////////////////////////////////////////////////
void itt_sync_create(void *addr, const char* objtype, const char* objname)
{
    HPX_INTERNAL_ITT_SYNC_CREATE(addr, objtype, objname);
}

void itt_sync_rename(void* addr, const char* objname)
{
    HPX_INTERNAL_ITT_SYNC_RENAME(addr, objname);
}

void itt_sync_prepare(void* addr)
{
    HPX_INTERNAL_ITT_SYNC_PREPARE(addr);
}

void itt_sync_acquired(void* addr)
{
    HPX_INTERNAL_ITT_SYNC_ACQUIRED(addr);
}

void itt_sync_cancel(void* addr)
{
    HPX_INTERNAL_ITT_SYNC_CANCEL(addr);
}

void itt_sync_releasing(void* addr)
{
    HPX_INTERNAL_ITT_SYNC_RELEASING(addr);
}

void itt_sync_released(void* addr)
{
    HPX_INTERNAL_ITT_SYNC_RELEASED(addr);
}

void itt_sync_destroy(void* addr)
{
    HPX_INTERNAL_ITT_SYNC_DESTROY(addr);
}

///////////////////////////////////////////////////////////////////////////////
__itt_caller itt_stack_create()
{
    return HPX_INTERNAL_ITT_STACK_CREATE();
}

void itt_stack_enter(__itt_caller ctx)
{
    HPX_INTERNAL_ITT_STACK_ENTER(ctx);
}

void itt_stack_leave(__itt_caller ctx)
{
    HPX_INTERNAL_ITT_STACK_LEAVE(ctx);
}

void itt_stack_destroy(__itt_caller ctx)
{
    HPX_INTERNAL_ITT_STACK_DESTROY(ctx);
}

///////////////////////////////////////////////////////////////////////////////
__itt_frame_t* itt_frame_create(char const* name)
{
    return HPX_INTERNAL_ITT_FRAME_CREATE(name);
}

void itt_frame_begin(__itt_frame_t* frame)
{
    HPX_INTERNAL_ITT_FRAME_BEGIN(frame);
}

void itt_frame_end(__itt_frame_t* frame)
{
    HPX_INTERNAL_ITT_FRAME_END(frame);
}

void itt_frame_destroy(__itt_frame_t* frame)
{
    HPX_INTERNAL_ITT_FRAME_DESTROY(frame);
}

///////////////////////////////////////////////////////////////////////////////
int itt_mark_create(char const* name)
{
    return HPX_INTERNAL_ITT_MARK_CREATE(name);
}

void itt_mark_off(int mark)
{
    HPX_INTERNAL_ITT_MARK_OFF(mark);
}

void itt_mark(int mark, char const* par)
{
    HPX_INTERNAL_ITT_MARK(mark, par);
}

///////////////////////////////////////////////////////////////////////////////
void itt_thread_set_name(char const* name)
{
    HPX_INTERNAL_ITT_THREAD_SET_NAME(name);
}

void itt_thread_ignore()
{
    HPX_INTERNAL_ITT_THREAD_IGNORE();
}

#endif // HPX_USE_ITT

