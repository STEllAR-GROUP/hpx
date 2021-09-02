//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>

struct ___itt_caller;
struct ___itt_string_handle;
struct ___itt_domain;
struct ___itt_id;
using __itt_heap_function = void*;
struct ___itt_counter;

///////////////////////////////////////////////////////////////////////////////
#define HPX_ITT_SYNC_CREATE(obj, type, name) itt_sync_create(obj, type, name)
#define HPX_ITT_SYNC_RENAME(obj, name) itt_sync_rename(obj, name)
#define HPX_ITT_SYNC_PREPARE(obj) itt_sync_prepare(obj)
#define HPX_ITT_SYNC_CANCEL(obj) itt_sync_cancel(obj)
#define HPX_ITT_SYNC_ACQUIRED(obj) itt_sync_acquired(obj)
#define HPX_ITT_SYNC_RELEASING(obj) itt_sync_releasing(obj)
#define HPX_ITT_SYNC_RELEASED(obj) itt_sync_released(obj)
#define HPX_ITT_SYNC_DESTROY(obj) itt_sync_destroy(obj)

#define HPX_ITT_STACK_CREATE(ctx) ctx = itt_stack_create()
#define HPX_ITT_STACK_CALLEE_ENTER(ctx) itt_stack_enter(ctx)
#define HPX_ITT_STACK_CALLEE_LEAVE(ctx) itt_stack_leave(ctx)
#define HPX_ITT_STACK_DESTROY(ctx) itt_stack_destroy(ctx)

#define HPX_ITT_FRAME_BEGIN(frame, id) itt_frame_begin(frame, id)
#define HPX_ITT_FRAME_END(frame, id) itt_frame_end(frame, id)

#define HPX_ITT_MARK_CREATE(mark, name) mark = itt_mark_create(name)
#define HPX_ITT_MARK_OFF(mark) itt_mark_off(mark)
#define HPX_ITT_MARK(mark, parameter) itt_mark(mark, parameter)

#define HPX_ITT_THREAD_SET_NAME(name) itt_thread_set_name(name)
#define HPX_ITT_THREAD_IGNORE() itt_thread_ignore()

#define HPX_ITT_TASK_BEGIN(domain, name) itt_task_begin(domain, name)
#define HPX_ITT_TASK_BEGIN_ID(domain, id, name) itt_task_begin(domain, id, name)
#define HPX_ITT_TASK_END(domain) itt_task_end(domain)

#define HPX_ITT_DOMAIN_CREATE(name) itt_domain_create(name)
#define HPX_ITT_STRING_HANDLE_CREATE(name) itt_string_handle_create(name)

#define HPX_ITT_MAKE_ID(addr, extra) itt_make_id(addr, extra)
#define HPX_ITT_ID_CREATE(domain, id) itt_id_create(domain, id)
#define HPX_ITT_ID_DESTROY(id) itt_id_destroy(id)

#define HPX_ITT_HEAP_FUNCTION_CREATE(name, domain)                             \
    itt_heap_function_create(name, domain) /**/
#define HPX_ITT_HEAP_ALLOCATE_BEGIN(f, size, initialized)                      \
    itt_heap_allocate_begin(f, size, initialized) /**/
#define HPX_ITT_HEAP_ALLOCATE_END(f, addr, size, initialized)                  \
    itt_heap_allocate_end(f, addr, size, initialized) /**/
#define HPX_ITT_HEAP_FREE_BEGIN(f, addr) itt_heap_free_begin(f, addr)
#define HPX_ITT_HEAP_FREE_END(f, addr) itt_heap_free_end(f, addr)
#define HPX_ITT_HEAP_REALLOCATE_BEGIN(f, addr, new_size, initialized)          \
    itt_heap_reallocate_begin(f, addr, new_size, initialized) /**/
#define HPX_ITT_HEAP_REALLOCATE_END(f, addr, new_addr, new_size, initialized)  \
    itt_heap_reallocate_end(f, addr, new_addr, new_size, initialized) /**/
#define HPX_ITT_HEAP_INTERNAL_ACCESS_BEGIN() itt_heap_internal_access_begin()
#define HPX_ITT_HEAP_INTERNAL_ACCESS_END() itt_heap_internal_access_end()

#define HPX_ITT_COUNTER_CREATE(name, domain)                                   \
    itt_counter_create(name, domain) /**/
#define HPX_ITT_COUNTER_CREATE_TYPED(name, domain, type)                       \
    itt_counter_create_typed(name, domain, type) /**/
#define HPX_ITT_COUNTER_SET_VALUE(id, value_ptr)                               \
    itt_counter_set_value(id, value_ptr) /**/
#define HPX_ITT_COUNTER_DESTROY(id) itt_counter_destroy(id)

#define HPX_ITT_METADATA_ADD(domain, id, key, data)                            \
    itt_metadata_add(domain, id, key, data) /**/

///////////////////////////////////////////////////////////////////////////////
// decide whether to use the ITT notify API if it's available

#if HPX_HAVE_ITTNOTIFY != 0
HPX_LOCAL_EXPORT extern bool use_ittnotify_api;

///////////////////////////////////////////////////////////////////////////////
HPX_LOCAL_EXPORT void itt_sync_create(
    void* addr, const char* objtype, const char* objname) noexcept;
HPX_LOCAL_EXPORT void itt_sync_rename(void* addr, const char* name) noexcept;
HPX_LOCAL_EXPORT void itt_sync_prepare(void* addr) noexcept;
HPX_LOCAL_EXPORT void itt_sync_acquired(void* addr) noexcept;
HPX_LOCAL_EXPORT void itt_sync_cancel(void* addr) noexcept;
HPX_LOCAL_EXPORT void itt_sync_releasing(void* addr) noexcept;
HPX_LOCAL_EXPORT void itt_sync_released(void* addr) noexcept;
HPX_LOCAL_EXPORT void itt_sync_destroy(void* addr) noexcept;

HPX_LOCAL_EXPORT ___itt_caller* itt_stack_create() noexcept;
HPX_LOCAL_EXPORT void itt_stack_enter(___itt_caller* ctx) noexcept;
HPX_LOCAL_EXPORT void itt_stack_leave(___itt_caller* ctx) noexcept;
HPX_LOCAL_EXPORT void itt_stack_destroy(___itt_caller* ctx) noexcept;

HPX_LOCAL_EXPORT void itt_frame_begin(
    ___itt_domain const* frame, ___itt_id* id) noexcept;
HPX_LOCAL_EXPORT void itt_frame_end(
    ___itt_domain const* frame, ___itt_id* id) noexcept;

HPX_LOCAL_EXPORT int itt_mark_create(char const*) noexcept;
HPX_LOCAL_EXPORT void itt_mark_off(int mark) noexcept;
HPX_LOCAL_EXPORT void itt_mark(int mark, char const*) noexcept;

HPX_LOCAL_EXPORT void itt_thread_set_name(char const*) noexcept;
HPX_LOCAL_EXPORT void itt_thread_ignore() noexcept;

HPX_LOCAL_EXPORT void itt_task_begin(
    ___itt_domain const*, ___itt_string_handle*) noexcept;
HPX_LOCAL_EXPORT void itt_task_begin(
    ___itt_domain const*, ___itt_id*, ___itt_string_handle*) noexcept;
HPX_LOCAL_EXPORT void itt_task_end(___itt_domain const*) noexcept;

HPX_LOCAL_EXPORT ___itt_domain* itt_domain_create(char const*) noexcept;
HPX_LOCAL_EXPORT ___itt_string_handle* itt_string_handle_create(
    char const*) noexcept;

HPX_LOCAL_EXPORT ___itt_id* itt_make_id(void*, std::size_t);
HPX_LOCAL_EXPORT void itt_id_create(
    ___itt_domain const*, ___itt_id* id) noexcept;
HPX_LOCAL_EXPORT void itt_id_destroy(___itt_id* id) noexcept;

HPX_LOCAL_EXPORT __itt_heap_function itt_heap_function_create(
    const char*, const char*) noexcept;
HPX_LOCAL_EXPORT void itt_heap_allocate_begin(
    __itt_heap_function, std::size_t, int) noexcept;
HPX_LOCAL_EXPORT void itt_heap_allocate_end(
    __itt_heap_function, void**, std::size_t, int) noexcept;
HPX_LOCAL_EXPORT void itt_heap_free_begin(__itt_heap_function, void*) noexcept;
HPX_LOCAL_EXPORT void itt_heap_free_end(__itt_heap_function, void*) noexcept;
HPX_LOCAL_EXPORT void itt_heap_reallocate_begin(
    __itt_heap_function, void*, std::size_t, int) noexcept;
HPX_LOCAL_EXPORT void itt_heap_reallocate_end(
    __itt_heap_function, void*, void**, std::size_t, int) noexcept;
HPX_LOCAL_EXPORT void itt_heap_internal_access_begin() noexcept;
HPX_LOCAL_EXPORT void itt_heap_internal_access_end() noexcept;

HPX_LOCAL_EXPORT ___itt_counter* itt_counter_create(
    char const*, char const*) noexcept;
HPX_LOCAL_EXPORT ___itt_counter* itt_counter_create_typed(
    char const*, char const*, int) noexcept;
HPX_LOCAL_EXPORT void itt_counter_destroy(___itt_counter*) noexcept;
HPX_LOCAL_EXPORT void itt_counter_set_value(___itt_counter*, void*) noexcept;

HPX_LOCAL_EXPORT int itt_event_create(char const* name, int namelen) noexcept;
HPX_LOCAL_EXPORT int itt_event_start(int evnt) noexcept;
HPX_LOCAL_EXPORT int itt_event_end(int evnt) noexcept;

HPX_LOCAL_EXPORT void itt_metadata_add(___itt_domain* domain, ___itt_id* id,
    ___itt_string_handle* key, std::uint64_t const& data) noexcept;
HPX_LOCAL_EXPORT void itt_metadata_add(___itt_domain* domain, ___itt_id* id,
    ___itt_string_handle* key, double const& data) noexcept;
HPX_LOCAL_EXPORT void itt_metadata_add(___itt_domain* domain, ___itt_id* id,
    ___itt_string_handle* key, char const* data) noexcept;
HPX_LOCAL_EXPORT void itt_metadata_add(___itt_domain* domain, ___itt_id* id,
    ___itt_string_handle* key, void const* data) noexcept;

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util {

    struct thread_description;
}}    // namespace hpx::util

namespace hpx { namespace util { namespace itt {

    struct stack_context
    {
        stack_context()
          : itt_context_(nullptr)
        {
            HPX_ITT_STACK_CREATE(itt_context_);
        }
        ~stack_context()
        {
            if (itt_context_)
                HPX_ITT_STACK_DESTROY(itt_context_);
        }

        stack_context(stack_context const& rhs) = delete;
        stack_context(stack_context&& rhs)
          : itt_context_(rhs.itt_context_)
        {
            rhs.itt_context_ = nullptr;
        }

        stack_context& operator=(stack_context const& rhs) = delete;
        stack_context& operator=(stack_context&& rhs)
        {
            if (this != &rhs)
            {
                itt_context_ = rhs.itt_context_;
                rhs.itt_context_ = nullptr;
            }
            return *this;
        }

        struct ___itt_caller* itt_context_;
    };

    struct caller_context
    {
        caller_context(stack_context& ctx)
          : ctx_(ctx)
        {
            HPX_ITT_STACK_CALLEE_ENTER(ctx_.itt_context_);
        }
        ~caller_context()
        {
            HPX_ITT_STACK_CALLEE_LEAVE(ctx_.itt_context_);
        }

        stack_context& ctx_;
    };

    //////////////////////////////////////////////////////////////////////////
    struct domain
    {
        HPX_NON_COPYABLE(domain);

        HPX_LOCAL_EXPORT domain(char const*) noexcept;
        HPX_LOCAL_EXPORT domain() noexcept;

        ___itt_domain* domain_;
    };

    struct thread_domain : domain
    {
        HPX_NON_COPYABLE(thread_domain);

        HPX_LOCAL_EXPORT thread_domain() noexcept;
    };

    struct id
    {
        id(domain const& domain, void* addr, unsigned long extra = 0) noexcept
        {
            id_ = HPX_ITT_MAKE_ID(addr, extra);
            HPX_ITT_ID_CREATE(domain.domain_, id_);
        }
        ~id()
        {
            HPX_ITT_ID_DESTROY(id_);
        }

        id(id const& rhs) = delete;
        id(id&& rhs) noexcept
          : id_(rhs.id_)
        {
            rhs.id_ = nullptr;
        }

        id& operator=(id const& rhs) = delete;
        id& operator=(id&& rhs) noexcept
        {
            if (this != &rhs)
            {
                id_ = rhs.id_;
                rhs.id_ = nullptr;
            }
            return *this;
        }

        ___itt_id* id_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct frame_context
    {
        frame_context(domain const& domain, id* ident = nullptr) noexcept
          : domain_(domain)
          , ident_(ident)
        {
            HPX_ITT_FRAME_BEGIN(
                domain_.domain_, ident_ ? ident_->id_ : nullptr);
        }
        ~frame_context()
        {
            HPX_ITT_FRAME_END(domain_.domain_, ident_ ? ident_->id_ : nullptr);
        }

        domain const& domain_;
        id* ident_;
    };

    struct undo_frame_context
    {
        undo_frame_context(frame_context& frame) noexcept
          : frame_(frame)
        {
            HPX_ITT_FRAME_END(frame_.domain_.domain_, nullptr);
        }
        ~undo_frame_context()
        {
            HPX_ITT_FRAME_BEGIN(frame_.domain_.domain_, nullptr);
        }

        frame_context& frame_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct mark_context
    {
        mark_context(char const* name) noexcept
          : itt_mark_(0)
          , name_(name)
        {
            HPX_ITT_MARK_CREATE(itt_mark_, name);
        }
        ~mark_context()
        {
            HPX_ITT_MARK_OFF(itt_mark_);
        }

        int itt_mark_;
        char const* name_;
    };

    struct undo_mark_context
    {
        undo_mark_context(mark_context& mark) noexcept
          : mark_(mark)
        {
            HPX_ITT_MARK_OFF(mark_.itt_mark_);
        }
        ~undo_mark_context()
        {
            HPX_ITT_MARK_CREATE(mark_.itt_mark_, mark_.name_);
        }

        mark_context& mark_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct string_handle
    {
        string_handle() noexcept
          : handle_(nullptr)
        {
        }
        string_handle(char const* s) noexcept
          : handle_(s == nullptr ? nullptr : HPX_ITT_STRING_HANDLE_CREATE(s))
        {
        }
        string_handle(___itt_string_handle* h) noexcept
          : handle_(h)
        {
        }

        string_handle& operator=(___itt_string_handle* h) noexcept
        {
            handle_ = h;
            return *this;
        }

        explicit operator bool() const noexcept
        {
            return handle_ != nullptr;
        }

        ___itt_string_handle* handle_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct task
    {
        HPX_LOCAL_EXPORT task(domain const&, string_handle const&,
            std::uint64_t metadata) noexcept;
        HPX_LOCAL_EXPORT task(domain const&, string_handle const&) noexcept;
        HPX_LOCAL_EXPORT ~task();

        void add_metadata(string_handle const& name, std::uint64_t val) noexcept
        {
            HPX_ITT_METADATA_ADD(domain_.domain_, id_, name.handle_, val);
        }
        void add_metadata(string_handle const& name, double val) noexcept
        {
            HPX_ITT_METADATA_ADD(domain_.domain_, id_, name.handle_, val);
        }
        void add_metadata(string_handle const& name, char const* val) noexcept
        {
            HPX_ITT_METADATA_ADD(domain_.domain_, id_, name.handle_, val);
        }
        template <typename T>
        void add_metadata(string_handle const& name, T const& val) noexcept
        {
            HPX_ITT_METADATA_ADD(domain_.domain_, id_, name.handle_,
                static_cast<void const*>(&val));
        }

        domain const& domain_;
        ___itt_id* id_;
        string_handle sh_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct heap_function
    {
        heap_function(char const* name, char const* domain) noexcept
          : heap_function_(HPX_ITT_HEAP_FUNCTION_CREATE(name, domain))
        {
        }

        __itt_heap_function heap_function_;
    };

    struct heap_internal_access
    {
        heap_internal_access() noexcept
        {
            HPX_ITT_HEAP_INTERNAL_ACCESS_BEGIN();
        }

        ~heap_internal_access()
        {
            HPX_ITT_HEAP_INTERNAL_ACCESS_END();
        }
    };

    struct heap_allocate
    {
        template <typename T>
        heap_allocate(heap_function& heap_function, T**& addr, std::size_t size,
            int init) noexcept
          : heap_function_(heap_function)
          , addr_(reinterpret_cast<void**&>(addr))
          , size_(size)
          , init_(init)
        {
            HPX_ITT_HEAP_ALLOCATE_BEGIN(
                heap_function_.heap_function_, size_, init_);
        }

        ~heap_allocate()
        {
            HPX_ITT_HEAP_ALLOCATE_END(
                heap_function_.heap_function_, addr_, size_, init_);
        }

    private:
        heap_function& heap_function_;
        void**& addr_;
        std::size_t size_;
        int init_;
    };

    struct heap_free
    {
        heap_free(heap_function& heap_function, void* addr) noexcept
          : heap_function_(heap_function)
          , addr_(addr)
        {
            HPX_ITT_HEAP_FREE_BEGIN(heap_function_.heap_function_, addr_);
        }

        ~heap_free()
        {
            HPX_ITT_HEAP_FREE_END(heap_function_.heap_function_, addr_);
        }

    private:
        heap_function& heap_function_;
        void* addr_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct counter
    {
        counter(char const* name, char const* domain) noexcept
          : id_(HPX_ITT_COUNTER_CREATE(name, domain))
        {
        }
        counter(char const* name, char const* domain, int type) noexcept
          : id_(HPX_ITT_COUNTER_CREATE_TYPED(name, domain, type))
        {
        }
        ~counter()
        {
            if (id_)
                HPX_ITT_COUNTER_DESTROY(id_);
        }

        template <typename T>
        void set_value(T const& value) noexcept
        {
            if (id_)
            {
                HPX_ITT_COUNTER_SET_VALUE(
                    id_, const_cast<void*>(static_cast<const void*>(&value)));
            }
        }

        counter(counter const& rhs) = delete;
        counter(counter&& rhs) noexcept
          : id_(rhs.id_)
        {
            rhs.id_ = nullptr;
        }

        counter& operator=(counter const& rhs) = delete;
        counter& operator=(counter&& rhs) noexcept
        {
            if (this != &rhs)
            {
                id_ = rhs.id_;
                rhs.id_ = nullptr;
            }
            return *this;
        }

    private:
        ___itt_counter* id_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct event
    {
        event(char const* name) noexcept
          : event_(itt_event_create(name, (int) strnlen(name, 256)))
        {
        }

        void start() const noexcept
        {
            itt_event_start(event_);
        }

        void end() const noexcept
        {
            itt_event_end(event_);
        }

    private:
        int event_;
    };

    struct mark_event
    {
        mark_event(event const& e) noexcept
          : e_(e)
        {
            e_.start();
        }
        ~mark_event()
        {
            e_.end();
        }

    private:
        event e_;
    };

    inline void event_tick(event const& e) noexcept
    {
        e.start();
    }
}}}    // namespace hpx::util::itt

#else

inline constexpr void itt_sync_create(void*, const char*, const char*) noexcept
{
}
inline constexpr void itt_sync_rename(void*, const char*) noexcept {}
inline constexpr void itt_sync_prepare(void*) noexcept {}
inline constexpr void itt_sync_acquired(void*) noexcept {}
inline constexpr void itt_sync_cancel(void*) noexcept {}
inline constexpr void itt_sync_releasing(void*) noexcept {}
inline constexpr void itt_sync_released(void*) noexcept {}
inline constexpr void itt_sync_destroy(void*) noexcept {}

inline constexpr ___itt_caller* itt_stack_create() noexcept
{
    return nullptr;
}
inline constexpr void itt_stack_enter(___itt_caller*) noexcept {}
inline constexpr void itt_stack_leave(___itt_caller*) noexcept {}
inline constexpr void itt_stack_destroy(___itt_caller*) noexcept {}

inline constexpr void itt_frame_begin(___itt_domain const*, ___itt_id*) noexcept
{
}
inline constexpr void itt_frame_end(___itt_domain const*, ___itt_id*) noexcept
{
}

inline constexpr int itt_mark_create(char const*) noexcept
{
    return 0;
}
inline constexpr void itt_mark_off(int) noexcept {}
inline constexpr void itt_mark(int, char const*) noexcept {}

inline constexpr void itt_thread_set_name(char const*) noexcept {}
inline constexpr void itt_thread_ignore() noexcept {}

inline constexpr void itt_task_begin(
    ___itt_domain const*, ___itt_string_handle*) noexcept
{
}
inline constexpr void itt_task_begin(
    ___itt_domain const*, ___itt_id*, ___itt_string_handle*) noexcept
{
}
inline constexpr void itt_task_end(___itt_domain const*) noexcept {}

inline constexpr ___itt_domain* itt_domain_create(char const*) noexcept
{
    return nullptr;
}
inline constexpr ___itt_string_handle* itt_string_handle_create(
    char const*) noexcept
{
    return nullptr;
}

inline constexpr ___itt_id* itt_make_id(void*, unsigned long)
{
    return nullptr;
}
inline constexpr void itt_id_create(___itt_domain const*, ___itt_id*) noexcept
{
}
inline constexpr void itt_id_destroy(___itt_id*) noexcept {}

inline constexpr __itt_heap_function itt_heap_function_create(
    const char*, const char*) noexcept
{
    return nullptr;
}
inline constexpr void itt_heap_allocate_begin(
    __itt_heap_function, std::size_t, int) noexcept
{
}
inline constexpr void itt_heap_allocate_end(
    __itt_heap_function, void**, std::size_t, int) noexcept
{
}
inline constexpr void itt_heap_free_begin(__itt_heap_function, void*) noexcept
{
}
inline constexpr void itt_heap_free_end(__itt_heap_function, void*) noexcept {}
inline constexpr void itt_heap_reallocate_begin(
    __itt_heap_function, void*, std::size_t, int) noexcept
{
}
inline constexpr void itt_heap_reallocate_end(
    __itt_heap_function, void*, void**, std::size_t, int) noexcept
{
}
inline constexpr void itt_heap_internal_access_begin() noexcept {}
inline constexpr void itt_heap_internal_access_end() noexcept {}

inline constexpr ___itt_counter* itt_counter_create(
    char const*, char const*) noexcept
{
    return nullptr;
}
inline constexpr ___itt_counter* itt_counter_create_typed(
    char const*, char const*, int) noexcept
{
    return nullptr;
}
inline constexpr void itt_counter_destroy(___itt_counter*) noexcept {}
inline constexpr void itt_counter_set_value(___itt_counter*, void*) noexcept {}

inline constexpr int itt_event_create(char const*, int) noexcept
{
    return 0;
}
inline constexpr int itt_event_start(int) noexcept
{
    return 0;
}
inline constexpr int itt_event_end(int) noexcept
{
    return 0;
}

inline constexpr void itt_metadata_add(___itt_domain*, ___itt_id*,
    ___itt_string_handle*, std::uint64_t const&) noexcept
{
}
inline constexpr void itt_metadata_add(
    ___itt_domain*, ___itt_id*, ___itt_string_handle*, double const&) noexcept
{
}
inline constexpr void itt_metadata_add(
    ___itt_domain*, ___itt_id*, ___itt_string_handle*, char const*) noexcept
{
}
inline constexpr void itt_metadata_add(
    ___itt_domain*, ___itt_id*, ___itt_string_handle*, void const*) noexcept
{
}

//////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util {

    struct thread_description;
}}    // namespace hpx::util

namespace hpx { namespace util { namespace itt {

    struct stack_context
    {
        stack_context() = default;
        ~stack_context() = default;
    };

    struct caller_context
    {
        constexpr caller_context(stack_context&) noexcept {}
        ~caller_context() = default;
    };

    //////////////////////////////////////////////////////////////////////////
    struct domain
    {
        HPX_NON_COPYABLE(domain);

        constexpr domain(char const*) noexcept {}
        domain() = default;
    };

    struct thread_domain : domain
    {
        HPX_NON_COPYABLE(thread_domain);

        thread_domain() = default;
    };

    struct id
    {
        constexpr id(domain const&, void*, unsigned long = 0) noexcept {}
        ~id() = default;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct frame_context
    {
        constexpr frame_context(domain const&, id* = nullptr) noexcept {}
        ~frame_context() = default;
    };

    struct undo_frame_context
    {
        constexpr undo_frame_context(frame_context const&) noexcept {}
        ~undo_frame_context() = default;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct mark_context
    {
        constexpr mark_context(char const*) noexcept {}
        ~mark_context() = default;
    };

    struct undo_mark_context
    {
        constexpr undo_mark_context(mark_context const&) noexcept {}
        ~undo_mark_context() = default;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct string_handle
    {
        constexpr string_handle(char const* = nullptr) noexcept {}
    };

    //////////////////////////////////////////////////////////////////////////
    struct task
    {
        constexpr task(
            domain const&, string_handle const&, std::uint64_t) noexcept
        {
        }
        constexpr task(domain const&, string_handle const&) noexcept {}

        ~task() = default;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct heap_function
    {
        constexpr heap_function(char const*, char const*) noexcept {}
        ~heap_function() = default;
    };

    struct heap_allocate
    {
        template <typename T>
        constexpr heap_allocate(
            heap_function& /*heap_function*/, T**, std::size_t, int) noexcept
        {
        }
        ~heap_allocate() = default;
    };

    struct heap_free
    {
        constexpr heap_free(heap_function& /*heap_function*/, void*) noexcept {}
        ~heap_free() = default;
    };

    struct heap_internal_access
    {
        heap_internal_access() = default;
        ~heap_internal_access() = default;
    };

    struct counter
    {
        constexpr counter(char const* /*name*/, char const* /*domain*/) noexcept
        {
        }
        ~counter() = default;
    };

    struct event
    {
        constexpr event(char const*) noexcept {}
    };

    struct mark_event
    {
        constexpr mark_event(event const&) noexcept {}
        ~mark_event() = default;
    };

    inline constexpr void event_tick(event const&) noexcept {}
}}}    // namespace hpx::util::itt

#endif    // HPX_HAVE_ITTNOTIFY
