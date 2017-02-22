//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/util/itt_notify.hpp>
#include <hpx/util/thread_description.hpp>

#if HPX_HAVE_ITTNOTIFY != 0

#define INTEL_ITTNOTIFY_API_PRIVATE

#include <ittnotify.h>
#include <legacy/ittnotify.h>

#include <cstddef>
#include <cstdint>

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
        __itt_stack_caller_create_ptr() : (__itt_caller)nullptr               \
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
    if (use_ittnotify_api && __itt_stack_caller_destroy_ptr &&                \
            ctx != (__itt_caller)nullptr)                                     \
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
#define HPX_INTERNAL_ITT_TASK_BEGIN_ID(domain, id, name)                      \
    if (use_ittnotify_api && __itt_task_begin_ptr)                            \
        __itt_task_begin_ptr(domain, id, __itt_null, name);                   \
    /**/
#define HPX_INTERNAL_ITT_TASK_END(domain)                                     \
    if (use_ittnotify_api && __itt_task_end_ptr)                              \
        __itt_task_end_ptr(domain);                                           \
    /**/

#define HPX_INTERNAL_ITT_DOMAIN_CREATE(name)                                  \
    (use_ittnotify_api && __itt_domain_create_ptr) ?                          \
        __itt_domain_create_ptr(name) : nullptr                               \
    /**/

#define HPX_INTERNAL_ITT_STRING_HANDLE_CREATE(name)                           \
    (use_ittnotify_api && __itt_string_handle_create_ptr) ?                   \
        __itt_string_handle_create_ptr(name) : nullptr                        \
    /**/

#define HPX_INTERNAL_ITT_MAKE_ID(addr, extra)                                 \
    use_ittnotify_api ? __itt_id_make(addr, extra) : __itt_null               \
    /**/

#define HPX_INTERNAL_ITT_ID_CREATE(domain, id)                                \
    if (use_ittnotify_api && __itt_id_create_ptr)                             \
        __itt_id_create_ptr(domain, id);                                      \
    /**/
#define HPX_INTERNAL_ITT_ID_DESTROY(id) delete id

///////////////////////////////////////////////////////////////////////////////
#define HPX_INTERNAL_ITT_HEAP_FUNCTION_CREATE(name, domain)                   \
    (use_ittnotify_api && __itt_heap_function_create_ptr) ?                   \
        __itt_heap_function_create_ptr(name, domain) : nullptr                \
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
#if defined(__itt_counter_create_typed_ptr) && defined(__itt_counter_set_value_ptr)
#define HPX_INTERNAL_COUNTER_CREATE(name, domain)                             \
    (use_ittnotify_api && __itt_counter_create_ptr) ?                         \
        __itt_counter_create_ptr(name, domain) : (__itt_counter)nullptr       \
    /**/
#define HPX_INTERNAL_COUNTER_CREATE_TYPED(name, domain, type)                 \
    (use_ittnotify_api && __itt_counter_create_typed_ptr) ?                   \
        __itt_counter_create_typed_ptr(name, domain, type) :                  \
            (__itt_counter)nullptr                                            \
    /**/
#define HPX_INTERNAL_COUNTER_SET_VALUE(id, value_ptr)                         \
    if (use_ittnotify_api && __itt_counter_set_value_ptr)                     \
        __itt_counter_set_value_ptr(id, value_ptr);                           \
    /**/
#define HPX_INTERNAL_COUNTER_DESTROY(id)                                      \
    if (use_ittnotify_api && __itt_counter_destroy_ptr)                       \
        __itt_counter_destroy_ptr(id);                                        \
    /**/
#else
// older itt-notify implementations don't support the typed counter API
#define HPX_INTERNAL_COUNTER_CREATE(name, domain)                             \
    (__itt_counter)nullptr                                                    \
    /**/
#define HPX_INTERNAL_COUNTER_CREATE_TYPED(name, domain, type)                 \
    (__itt_counter)nullptr                                                    \
    /**/
#define HPX_INTERNAL_COUNTER_SET_VALUE(id, value_ptr)                         \
    /**/
#define HPX_INTERNAL_COUNTER_DESTROY(id)                                      \
    /**/
#endif

///////////////////////////////////////////////////////////////////////////////
#define HPX_INTERNAL_EVENT_CREATE(name, len)                                  \
    (use_ittnotify_api && __itt_event_create_ptr) ?                           \
        __itt_event_create_ptr(name, len) : 0;                                \
    /**/
#define HPX_INTERNAL_EVENT_START(e)                                           \
    (use_ittnotify_api && __itt_event_start_ptr) ?                            \
        __itt_event_start_ptr(e) : 0                                          \
    /**/
#define HPX_INTERNAL_EVENT_END(e)                                             \
    (use_ittnotify_api && __itt_event_end_ptr) ?                              \
        __itt_event_end_ptr(e) : 0                                            \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_INTERNAL_METADATA_ADD(domain, id, key, type, count, data)         \
    if (use_ittnotify_api && __itt_metadata_add_ptr)                          \
        __itt_metadata_add_ptr(domain, id, key, type, count, data)            \
    /**/
#define HPX_INTERNAL_METADATA_STR_ADD(domain, id, key, data)                  \
    if (use_ittnotify_api && __itt_metadata_str_add_ptr)                      \
        __itt_metadata_str_add_ptr(domain, id, key, data, 0)                  \
    /**/

///////////////////////////////////////////////////////////////////////////////
#if defined(HPX_MSVC) \
    || defined(__BORLANDC__) \
    || (defined(__MWERKS__) && defined(_WIN32) && (__MWERKS__ >= 0x3000)) \
    || (defined(__ICL) && defined(_MSC_EXTENSIONS) && (HPX_MSVC >= 1200))

#pragma comment(lib, "libittnotify.lib")
#endif

///////////////////////////////////////////////////////////////////////////////
#define HPX_INTERNAL_ITT_SYNC_PREPARE(obj)\
           HPX_INTERNAL_ITT_SYNC(sync_prepare, obj)
#define HPX_INTERNAL_ITT_SYNC_CANCEL(obj)\
            HPX_INTERNAL_ITT_SYNC(sync_cancel, obj)
#define HPX_INTERNAL_ITT_SYNC_ACQUIRED(obj)\
          HPX_INTERNAL_ITT_SYNC(sync_acquired, obj)
#define HPX_INTERNAL_ITT_SYNC_RELEASING(obj)\
         HPX_INTERNAL_ITT_SYNC(sync_releasing, obj)
#define HPX_INTERNAL_ITT_SYNC_RELEASED(obj)\
          ((void)0) //HPX_INTERNAL_ITT_SYNC(sync_released, obj)
#define HPX_INTERNAL_ITT_SYNC_DESTROY(obj)\
           HPX_INTERNAL_ITT_SYNC(sync_destroy, obj)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace itt
{
    domain::domain(char const* name)
      : domain_(HPX_ITT_DOMAIN_CREATE(name))
    {
        if (domain_) domain_->flags = 1;    // enable domain
    }

    task::task(domain const& domain, util::thread_description const& name)
      : domain_(domain), id_(0), sh_()
    {
        if (name.kind() == util::thread_description::data_type_description)
        {
            sh_ = name.get_description_itt();
        }
        else
        {
            sh_ = HPX_ITT_STRING_HANDLE_CREATE("address");
        }

        id_ = HPX_ITT_MAKE_ID(domain_.domain_,
            reinterpret_cast<std::size_t>(sh_.handle_));

        HPX_ITT_TASK_BEGIN_ID(domain_.domain_, id_, sh_.handle_);

        if (name.kind() == util::thread_description::data_type_address)
        {
            add_metadata(sh_.handle_, name.get_address());
        }
    }

    task::task(domain const& domain, string_handle const& name)
      : domain_(domain), id_(0), sh_(name)
    {
        id_ = HPX_ITT_MAKE_ID(domain_.domain_,
            reinterpret_cast<std::size_t>(sh_.handle_));

        HPX_ITT_TASK_BEGIN_ID(domain_.domain_, id_, sh_.handle_);
    }
}}}

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

void itt_task_begin(___itt_domain const* domain, ___itt_id* id,
    ___itt_string_handle* name)
{
    HPX_INTERNAL_ITT_TASK_BEGIN_ID(domain, *id, name);
}

void itt_task_end(___itt_domain const* domain)
{
    HPX_INTERNAL_ITT_TASK_END(domain);
}

___itt_domain* itt_domain_create(char const* name)
{
    return HPX_INTERNAL_ITT_DOMAIN_CREATE(name);
}

___itt_string_handle* itt_string_handle_create(char const* name)
{
    return HPX_INTERNAL_ITT_STRING_HANDLE_CREATE(name);
}

___itt_id* itt_make_id(void* addr, std::size_t extra)
{
    return new ___itt_id(HPX_INTERNAL_ITT_MAKE_ID(addr, extra));
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

void itt_heap_allocate_end(__itt_heap_function f, void** addr,
    std::size_t size, int init)
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

void itt_heap_reallocate_begin(__itt_heap_function f, void* addr,
    std::size_t new_size, int init)
{
    HPX_INTERNAL_HEAP_REALLOCATE_BEGIN(f, addr, new_size, init);
}

void itt_heap_reallocate_end(__itt_heap_function f, void* addr,
    void** new_addr, std::size_t new_size, int init)
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

///////////////////////////////////////////////////////////////////////////////
__itt_counter itt_counter_create(char const* name, char const* domain)
{
    return HPX_INTERNAL_COUNTER_CREATE(name, domain);
}

__itt_counter itt_counter_create_typed(char const* name, char const* domain,
    int type)
{
    return HPX_INTERNAL_COUNTER_CREATE_TYPED(name, domain,
        (__itt_metadata_type)type);
}

void itt_counter_destroy(__itt_counter id)
{
    HPX_INTERNAL_COUNTER_DESTROY(id);
}

void itt_counter_set_value(__itt_counter id, void *value_ptr)
{
    HPX_INTERNAL_COUNTER_SET_VALUE(id, value_ptr);
}

///////////////////////////////////////////////////////////////////////////////
__itt_event itt_event_create(char const *name, int namelen)
{
    return HPX_INTERNAL_EVENT_CREATE(name, namelen);
}

int itt_event_start(__itt_event evnt)
{
    return HPX_INTERNAL_EVENT_START(evnt);
}

int itt_event_end(__itt_event evnt)
{
    return HPX_INTERNAL_EVENT_END(evnt);
}

///////////////////////////////////////////////////////////////////////////////
void itt_metadata_add(___itt_domain* domain, ___itt_id* id,
    ___itt_string_handle* key, std::uint64_t const& data)
{
    HPX_INTERNAL_METADATA_ADD(domain, *id, key, __itt_metadata_u64, 1,
        const_cast<std::uint64_t*>(&data));
}

void itt_metadata_add(___itt_domain* domain, ___itt_id* id,
    ___itt_string_handle* key, double const& data)
{
    HPX_INTERNAL_METADATA_ADD(domain, *id, key, __itt_metadata_double, 1,
        const_cast<double*>(&data));
}

void itt_metadata_add(___itt_domain* domain, ___itt_id* id,
    ___itt_string_handle* key, char const* data)
{
    HPX_INTERNAL_METADATA_STR_ADD(domain, *id, key, data);
}

void itt_metadata_add(___itt_domain* domain, ___itt_id* id,
    ___itt_string_handle* key, void const* data)
{
    HPX_INTERNAL_METADATA_ADD(domain, *id, key, __itt_metadata_unknown, 1,
        const_cast<void*>(data));
}

#endif // HPX_HAVE_ITTNOTIFY

