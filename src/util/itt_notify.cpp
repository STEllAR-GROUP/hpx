//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/itt_notify.hpp>

#if HPX_USE_ITTNOTIFY != 0

#include <ittnotify.h>
#include <legacy/ittnotify.h>
#include <hpx/util/internal/ittnotify.h>

///////////////////////////////////////////////////////////////////////////////
// decide whether to use the ITT notify API if it's available
bool use_ittnotify_api = false;

///////////////////////////////////////////////////////////////////////////////
#define HPX_INTERNAL_ITT_SYNC_CREATE(obj, type, name)                         \
    if (use_ittnotify_api && __itt_sync_create_ptr) {                         \
        __itt_sync_create_ptr(                                                \
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
    if (use_ittnotify_api && __itt_sync_rename_ptr) {                         \
        __itt_sync_rename_ptr(                                                \
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
#define HPX_INTERNAL_ITT_FRAME_BEGIN(domain, id)                              \
    if (use_ittnotify_api && __itt_frame_begin_v3_ptr)                        \
        __itt_frame_begin_v3_ptr(domain, id);                                 \
    /**/
#define HPX_INTERNAL_ITT_FRAME_END(domain, id)                                \
    if (use_ittnotify_api && __itt_frame_end_v3_ptr)                          \
        __itt_frame_end_v3_ptr(domain, id);                                   \
    /**/

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

///////////////////////////////////////////////////////////////////////////////
#define HPX_INTERNAL_ITT_THREAD_SET_NAME(name)                                \
    if (use_ittnotify_api && __itt_thread_set_name_ptr)                       \
        __itt_thread_set_name_ptr(name);                                      \
    /**/
#define HPX_INTERNAL_ITT_THREAD_IGNORE()                                      \
    if (use_ittnotify_api && __itt_thread_ignore_ptr)                         \
        __itt_thread_ignore_ptr();                                            \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_INTERNAL_ITT_TASK_BEGIN(domain, name)                             \
    if (use_ittnotify_api && __itt_task_begin_ptr)                            \
        __itt_task_begin_ptr(domain, __itt_null, __itt_null, name);           \
    /**/
#define HPX_INTERNAL_ITT_TASK_END(domain)                                     \
    if (use_ittnotify_api && __itt_task_end_ptr)                              \
        __itt_task_end_ptr(domain);                                           \
    /**/

#define HPX_INTERNAL_ITT_DOMAIN_CREATE(name)                                  \
    (use_ittnotify_api && __itt_domain_create_ptr) ?                          \
        __itt_domain_create_ptr(name) : 0                                     \
    /**/

#define HPX_INTERNAL_ITT_STRING_HANDLE_CREATE(name)                           \
    (use_ittnotify_api && __itt_string_handle_create_ptr) ?                   \
        __itt_string_handle_create_ptr(name) : 0                              \
    /**/

#define HPX_INTERNAL_ITT_MAKE_ID(id, addr, extra)                             \
    if (use_ittnotify_api) id = __itt_id_make(addr, extra);                   \
    /**/

#define HPX_INTERNAL_ITT_ID_CREATE(domain, id)                                \
    if (use_ittnotify_api && __itt_id_create_ptr)                             \
        __itt_id_create_ptr(domain, id);                                      \
    /**/
#define HPX_INTERNAL_ITT_ID_DESTROY(id) delete id

///////////////////////////////////////////////////////////////////////////////
#define HPX_INTERNAL_ITT_HEAP_FUNCTION_CREATE(name, domain)                   \
    (use_ittnotify_api && __itt_heap_function_create_ptr) ?                   \
        __itt_heap_function_create_ptr(name, domain) : 0                      \
    /**/

#define HPX_INTERNAL_HEAP_ALLOCATE_BEGIN(f, size, init)                       \
    if (use_ittnotify_api && __itt_heap_allocate_begin_ptr)                   \
        __itt_heap_allocate_begin_ptr(f, size, init);                         \
    /**/
#define HPX_INTERNAL_HEAP_ALLOCATE_END(f, addr, size, init)                   \
    if (use_ittnotify_api && __itt_heap_allocate_end_ptr)                     \
        __itt_heap_allocate_end_ptr(f, addr, size, init);                     \
    /**/

#define HPX_INTERNAL_HEAP_FREE_BEGIN(f, addr)                                 \
    if (use_ittnotify_api && __itt_heap_free_begin_ptr)                       \
        __itt_heap_free_begin_ptr(f, addr);                                   \
    /**/
#define HPX_INTERNAL_HEAP_FREE_END(f, addr)                                   \
    if (use_ittnotify_api && __itt_heap_free_end_ptr)                         \
        __itt_heap_free_end_ptr(f, addr);                                     \
    /**/

#define HPX_INTERNAL_HEAP_REALLOCATE_BEGIN(f, addr, size, init)               \
    if (use_ittnotify_api && __itt_heap_reallocate_begin_ptr)                 \
        __itt_heap_reallocate_begin_ptr(f, addr, size, init);                 \
    /**/
#define HPX_INTERNAL_HEAP_REALLOCATE_END(f, addr, new_addr, size, init)       \
    if (use_ittnotify_api && __itt_heap_reallocate_end_ptr)                   \
        __itt_heap_reallocate_end_ptr(f, addr, new_addr, size, init);         \
    /**/

#define HPX_INTERNAL_INTERNAL_ACCESS_BEGIN()                                  \
    if (use_ittnotify_api && __itt_heap_internal_access_begin_ptr)            \
        __itt_heap_internal_access_begin_ptr();                               \
    /**/
#define HPX_INTERNAL_INTERNAL_ACCESS_END()                                    \
    if (use_ittnotify_api && __itt_heap_internal_access_end_ptr)              \
        __itt_heap_internal_access_end_ptr();                                 \
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
void itt_frame_begin(___itt_domain const* domain, ___itt_id* id)
{
    HPX_INTERNAL_ITT_FRAME_BEGIN(domain, id);
}

void itt_frame_end(___itt_domain const* domain, ___itt_id* id)
{
    HPX_INTERNAL_ITT_FRAME_END(domain, id);
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

///////////////////////////////////////////////////////////////////////////////
void itt_task_begin(___itt_domain const* domain, ___itt_string_handle* name)
{
    HPX_INTERNAL_ITT_TASK_BEGIN(domain, name);
}

void itt_task_end(___itt_domain const* domain)
{
    HPX_INTERNAL_ITT_TASK_END(domain);
}

___itt_domain* itt_domain_create(char const* name)
{
    return HPX_INTERNAL_ITT_DOMAIN_CREATE(name);
}

___itt_string_handle* itt_task_handle_create(char const* name)
{
    return HPX_INTERNAL_ITT_STRING_HANDLE_CREATE(name);
}

___itt_id* itt_make_id(void* addr, unsigned long extra)
{
    ___itt_id* id = new ___itt_id;
    HPX_INTERNAL_ITT_MAKE_ID(*id, addr, extra);
    return id;
}

void itt_id_create(___itt_domain const* domain, ___itt_id* id)
{
    HPX_INTERNAL_ITT_ID_CREATE(domain, *id);
}

void itt_id_destroy(___itt_id* id)
{
    HPX_INTERNAL_ITT_ID_DESTROY(id);
}

///////////////////////////////////////////////////////////////////////////////
__itt_heap_function itt_heap_function_create(const char* name, const char* domain)
{
    return HPX_INTERNAL_ITT_HEAP_FUNCTION_CREATE(name, domain);
}

void itt_heap_allocate_begin(__itt_heap_function f, std::size_t size, int init)
{
    HPX_INTERNAL_HEAP_ALLOCATE_BEGIN(f, size, init);
}

void itt_heap_allocate_end(__itt_heap_function f, void** addr, std::size_t size, int init)
{
    HPX_INTERNAL_HEAP_ALLOCATE_END(f, addr, size, init);
}

void itt_heap_free_begin(__itt_heap_function f, void* addr)
{
    HPX_INTERNAL_HEAP_FREE_BEGIN(f, addr);
}

void itt_heap_free_end(__itt_heap_function f, void* addr)
{
    HPX_INTERNAL_HEAP_FREE_END(f, addr);
}

void itt_heap_reallocate_begin(__itt_heap_function f, void* addr, std::size_t new_size, int init)
{
    HPX_INTERNAL_HEAP_REALLOCATE_BEGIN(f, addr, new_size, init);
}

void itt_heap_reallocate_end(__itt_heap_function f, void* addr, void** new_addr, std::size_t new_size, int init)
{
    HPX_INTERNAL_HEAP_REALLOCATE_END(f, addr, new_addr, new_size, init);
}

void itt_heap_internal_access_begin()
{
    HPX_INTERNAL_INTERNAL_ACCESS_BEGIN();
}

void itt_heap_internal_access_end()
{
    HPX_INTERNAL_INTERNAL_ACCESS_END();
}

#endif // HPX_USE_ITTNOTIFY

