//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

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
HPX_CORE_EXPORT extern bool use_ittnotify_api;

///////////////////////////////////////////////////////////////////////////////
HPX_CORE_EXPORT void itt_sync_create(
    void* addr, char const* objtype, char const* objname) noexcept;
HPX_CORE_EXPORT void itt_sync_rename(void* addr, char const* name) noexcept;
HPX_CORE_EXPORT void itt_sync_prepare(void* addr) noexcept;
HPX_CORE_EXPORT void itt_sync_acquired(void* addr) noexcept;
HPX_CORE_EXPORT void itt_sync_cancel(void* addr) noexcept;
HPX_CORE_EXPORT void itt_sync_releasing(void* addr) noexcept;
HPX_CORE_EXPORT void itt_sync_released(void* addr) noexcept;
HPX_CORE_EXPORT void itt_sync_destroy(void* addr) noexcept;

[[nodiscard]] HPX_CORE_EXPORT ___itt_caller* itt_stack_create() noexcept;
HPX_CORE_EXPORT void itt_stack_enter(___itt_caller* ctx) noexcept;
HPX_CORE_EXPORT void itt_stack_leave(___itt_caller* ctx) noexcept;
HPX_CORE_EXPORT void itt_stack_destroy(___itt_caller* ctx) noexcept;

HPX_CORE_EXPORT void itt_frame_begin(
    ___itt_domain const* frame, ___itt_id* id) noexcept;
HPX_CORE_EXPORT void itt_frame_end(
    ___itt_domain const* frame, ___itt_id* id) noexcept;

[[nodiscard]] HPX_CORE_EXPORT int itt_mark_create(char const*) noexcept;
HPX_CORE_EXPORT void itt_mark_off(int mark) noexcept;
HPX_CORE_EXPORT void itt_mark(int mark, char const*) noexcept;

HPX_CORE_EXPORT void itt_thread_set_name(char const*) noexcept;
HPX_CORE_EXPORT void itt_thread_ignore() noexcept;

HPX_CORE_EXPORT void itt_task_begin(
    ___itt_domain const*, ___itt_string_handle*) noexcept;
HPX_CORE_EXPORT void itt_task_begin(
    ___itt_domain const*, ___itt_id*, ___itt_string_handle*) noexcept;
HPX_CORE_EXPORT void itt_task_end(___itt_domain const*) noexcept;

HPX_CORE_EXPORT ___itt_domain* itt_domain_create(char const*) noexcept;
HPX_CORE_EXPORT ___itt_string_handle* itt_string_handle_create(
    char const*) noexcept;

[[nodiscard]] HPX_CORE_EXPORT ___itt_id* itt_make_id(void*, std::size_t);
HPX_CORE_EXPORT void itt_id_create(
    ___itt_domain const*, ___itt_id* id) noexcept;
HPX_CORE_EXPORT void itt_id_destroy(___itt_id* id) noexcept;

[[nodiscard]] HPX_CORE_EXPORT __itt_heap_function itt_heap_function_create(
    char const*, char const*) noexcept;
HPX_CORE_EXPORT void itt_heap_allocate_begin(
    __itt_heap_function, std::size_t, int) noexcept;
HPX_CORE_EXPORT void itt_heap_allocate_end(
    __itt_heap_function, void**, std::size_t, int) noexcept;
HPX_CORE_EXPORT void itt_heap_free_begin(__itt_heap_function, void*) noexcept;
HPX_CORE_EXPORT void itt_heap_free_end(__itt_heap_function, void*) noexcept;
HPX_CORE_EXPORT void itt_heap_reallocate_begin(
    __itt_heap_function, void*, std::size_t, int) noexcept;
HPX_CORE_EXPORT void itt_heap_reallocate_end(
    __itt_heap_function, void*, void**, std::size_t, int) noexcept;
HPX_CORE_EXPORT void itt_heap_internal_access_begin() noexcept;
HPX_CORE_EXPORT void itt_heap_internal_access_end() noexcept;

[[nodiscard]] HPX_CORE_EXPORT ___itt_counter* itt_counter_create(
    char const*, char const*) noexcept;
[[nodiscard]] HPX_CORE_EXPORT ___itt_counter* itt_counter_create_typed(
    char const*, char const*, int) noexcept;
HPX_CORE_EXPORT void itt_counter_destroy(___itt_counter*) noexcept;
HPX_CORE_EXPORT void itt_counter_set_value(___itt_counter*, void*) noexcept;

[[nodiscard]] HPX_CORE_EXPORT int itt_event_create(
    char const* name, int namelen) noexcept;
HPX_CORE_EXPORT int itt_event_start(int evnt) noexcept;
HPX_CORE_EXPORT int itt_event_end(int evnt) noexcept;

HPX_CORE_EXPORT void itt_metadata_add(___itt_domain* domain, ___itt_id* id,
    ___itt_string_handle* key, std::uint64_t const& data) noexcept;
HPX_CORE_EXPORT void itt_metadata_add(___itt_domain* domain, ___itt_id* id,
    ___itt_string_handle* key, double const& data) noexcept;
HPX_CORE_EXPORT void itt_metadata_add(___itt_domain* domain, ___itt_id* id,
    ___itt_string_handle* key, char const* data) noexcept;
HPX_CORE_EXPORT void itt_metadata_add(___itt_domain* domain, ___itt_id* id,
    ___itt_string_handle* key, void const* data) noexcept;

///////////////////////////////////////////////////////////////////////////////
namespace hpx::threads {

    struct thread_description;
}    // namespace hpx::threads

namespace hpx::util::itt {

    struct stack_context
    {
        HPX_CORE_EXPORT stack_context();
        HPX_CORE_EXPORT ~stack_context();

        stack_context(stack_context const& rhs) = delete;
        stack_context(stack_context&& rhs) noexcept
          : itt_context_(rhs.itt_context_)
        {
            rhs.itt_context_ = nullptr;
        }

        stack_context& operator=(stack_context const& rhs) = delete;
        stack_context& operator=(stack_context&& rhs) noexcept
        {
            if (this != &rhs)
            {
                itt_context_ = rhs.itt_context_;
                rhs.itt_context_ = nullptr;
            }
            return *this;
        }

        struct ___itt_caller* itt_context_ = nullptr;
    };

    struct caller_context
    {
        HPX_CORE_EXPORT explicit caller_context(stack_context& ctx);
        HPX_CORE_EXPORT ~caller_context();

        stack_context& ctx_;
    };

    //////////////////////////////////////////////////////////////////////////
    struct domain
    {
        HPX_NON_COPYABLE(domain);

        domain() = default;
        HPX_CORE_EXPORT explicit domain(char const*) noexcept;

        ___itt_domain* domain_ = nullptr;
    };

    struct thread_domain : domain
    {
        HPX_NON_COPYABLE(thread_domain);

        HPX_CORE_EXPORT thread_domain() noexcept;
    };

    struct id
    {
        HPX_CORE_EXPORT id(
            domain const& domain, void* addr, unsigned long extra = 0) noexcept;
        HPX_CORE_EXPORT ~id();

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

        ___itt_id* id_ = nullptr;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct frame_context
    {
        HPX_CORE_EXPORT explicit frame_context(
            domain const& domain, id* ident = nullptr) noexcept;
        HPX_CORE_EXPORT ~frame_context();

        domain const& domain_;
        id* ident_ = nullptr;
    };

    struct undo_frame_context
    {
        HPX_CORE_EXPORT explicit undo_frame_context(
            frame_context& frame) noexcept;
        HPX_CORE_EXPORT ~undo_frame_context();

        frame_context& frame_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct mark_context
    {
        HPX_CORE_EXPORT explicit mark_context(char const* name) noexcept;
        HPX_CORE_EXPORT ~mark_context();

        int itt_mark_;
        char const* name_ = nullptr;
    };

    struct undo_mark_context
    {
        HPX_CORE_EXPORT explicit undo_mark_context(mark_context& mark) noexcept;
        HPX_CORE_EXPORT ~undo_mark_context();

        mark_context& mark_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct string_handle
    {
        string_handle() noexcept = default;

        HPX_CORE_EXPORT explicit string_handle(char const* s) noexcept;

        explicit string_handle(___itt_string_handle* h) noexcept
          : handle_(h)
        {
        }

        string_handle(string_handle const&) = default;
        string_handle(string_handle&& rhs)
          : handle_(rhs.handle_)
        {
            rhs.handle_ = nullptr;
        }

        string_handle& operator=(string_handle const&) = default;
        string_handle& operator=(string_handle&& rhs) noexcept
        {
            if (this != &rhs)
            {
                handle_ = rhs.handle_;
                rhs.handle_ = nullptr;
            }
            return *this;
        }

        string_handle& operator=(___itt_string_handle* h) noexcept
        {
            handle_ = h;
            return *this;
        }

        explicit constexpr operator bool() const noexcept
        {
            return handle_ != nullptr;
        }

        ___itt_string_handle* handle_ = nullptr;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct task
    {
        HPX_CORE_EXPORT task(domain const&, string_handle const&,
            std::uint64_t metadata) noexcept;
        HPX_CORE_EXPORT task(domain const&, string_handle const&) noexcept;
        HPX_CORE_EXPORT ~task();

        HPX_CORE_EXPORT void add_metadata(
            string_handle const& name, std::uint64_t val) noexcept;
        HPX_CORE_EXPORT void add_metadata(
            string_handle const& name, double val) noexcept;
        HPX_CORE_EXPORT void add_metadata(
            string_handle const& name, char const* val) noexcept;
        HPX_CORE_EXPORT void add_metadata(
            string_handle const& name, void const* val) noexcept;

        template <typename T>
        void add_metadata(string_handle const& name, T const& val) noexcept
        {
            add_metadata(name, static_cast<void const*>(&val));
        }

        domain const& domain_;
        ___itt_id* id_ = nullptr;
        string_handle sh_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct heap_function
    {
        HPX_CORE_EXPORT heap_function(
            char const* name, char const* domain) noexcept;

        __itt_heap_function heap_function_ = nullptr;
    };

    struct heap_internal_access
    {
        HPX_CORE_EXPORT heap_internal_access() noexcept;
        HPX_CORE_EXPORT ~heap_internal_access();
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
            if (use_ittnotify_api)
            {
                HPX_ITT_HEAP_ALLOCATE_BEGIN(
                    heap_function_.heap_function_, size_, init_);
            }
        }

        ~heap_allocate()
        {
            if (use_ittnotify_api)
            {
                HPX_ITT_HEAP_ALLOCATE_END(
                    heap_function_.heap_function_, addr_, size_, init_);
            }
        }

    private:
        heap_function& heap_function_;
        void**& addr_;
        std::size_t size_;
        int init_;
    };

    struct heap_free
    {
        HPX_CORE_EXPORT heap_free(
            heap_function& heap_function, void* addr) noexcept;
        HPX_CORE_EXPORT ~heap_free();

    private:
        heap_function& heap_function_;
        void* addr_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct counter
    {
        HPX_CORE_EXPORT counter(char const* name, char const* domain) noexcept;
        HPX_CORE_EXPORT counter(
            char const* name, char const* domain, int type) noexcept;
        HPX_CORE_EXPORT ~counter();

        template <typename T>
        void set_value(T const& value) noexcept
        {
            if (use_ittnotify_api && id_)
            {
                HPX_ITT_COUNTER_SET_VALUE(
                    id_, const_cast<void*>(static_cast<void const*>(&value)));
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
        ___itt_counter* id_ = nullptr;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct event
    {
        explicit event(char const* name) noexcept
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
        int event_ = 0;
    };

    struct mark_event
    {
        explicit mark_event(event const& e) noexcept
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
}    // namespace hpx::util::itt

#else

constexpr void itt_sync_create(void*, char const*, char const*) noexcept {}
constexpr void itt_sync_rename(void*, char const*) noexcept {}
constexpr void itt_sync_prepare(void*) noexcept {}
constexpr void itt_sync_acquired(void*) noexcept {}
constexpr void itt_sync_cancel(void*) noexcept {}
constexpr void itt_sync_releasing(void*) noexcept {}
constexpr void itt_sync_released(void*) noexcept {}
constexpr void itt_sync_destroy(void*) noexcept {}

[[nodiscard]] constexpr ___itt_caller* itt_stack_create() noexcept
{
    return nullptr;
}
constexpr void itt_stack_enter(___itt_caller*) noexcept {}
constexpr void itt_stack_leave(___itt_caller*) noexcept {}
constexpr void itt_stack_destroy(___itt_caller*) noexcept {}

constexpr void itt_frame_begin(___itt_domain const*, ___itt_id*) noexcept {}
constexpr void itt_frame_end(___itt_domain const*, ___itt_id*) noexcept {}

[[nodiscard]] constexpr int itt_mark_create(char const*) noexcept
{
    return 0;
}
constexpr void itt_mark_off(int) noexcept {}
constexpr void itt_mark(int, char const*) noexcept {}

constexpr void itt_thread_set_name(char const*) noexcept {}
constexpr void itt_thread_ignore() noexcept {}

constexpr void itt_task_begin(
    ___itt_domain const*, ___itt_string_handle*) noexcept
{
}
constexpr void itt_task_begin(
    ___itt_domain const*, ___itt_id*, ___itt_string_handle*) noexcept
{
}
constexpr void itt_task_end(___itt_domain const*) noexcept {}

[[nodiscard]] constexpr ___itt_domain* itt_domain_create(char const*) noexcept
{
    return nullptr;
}
[[nodiscard]] constexpr ___itt_string_handle* itt_string_handle_create(
    char const*) noexcept
{
    return nullptr;
}

[[nodiscard]] constexpr ___itt_id* itt_make_id(void*, unsigned long)
{
    return nullptr;
}
constexpr void itt_id_create(___itt_domain const*, ___itt_id*) noexcept {}
constexpr void itt_id_destroy(___itt_id*) noexcept {}

[[nodiscard]] constexpr __itt_heap_function itt_heap_function_create(
    char const*, char const*) noexcept
{
    return nullptr;
}
constexpr void itt_heap_allocate_begin(
    __itt_heap_function, std::size_t, int) noexcept
{
}
constexpr void itt_heap_allocate_end(
    __itt_heap_function, void**, std::size_t, int) noexcept
{
}
constexpr void itt_heap_free_begin(__itt_heap_function, void*) noexcept {}
constexpr void itt_heap_free_end(__itt_heap_function, void*) noexcept {}
constexpr void itt_heap_reallocate_begin(
    __itt_heap_function, void*, std::size_t, int) noexcept
{
}
constexpr void itt_heap_reallocate_end(
    __itt_heap_function, void*, void**, std::size_t, int) noexcept
{
}
constexpr void itt_heap_internal_access_begin() noexcept {}
constexpr void itt_heap_internal_access_end() noexcept {}

[[nodiscard]] constexpr ___itt_counter* itt_counter_create(
    char const*, char const*) noexcept
{
    return nullptr;
}
[[nodiscard]] constexpr ___itt_counter* itt_counter_create_typed(
    char const*, char const*, int) noexcept
{
    return nullptr;
}
constexpr void itt_counter_destroy(___itt_counter*) noexcept {}
constexpr void itt_counter_set_value(___itt_counter*, void*) noexcept {}

[[nodiscard]] constexpr int itt_event_create(char const*, int) noexcept
{
    return 0;
}
constexpr int itt_event_start(int) noexcept
{
    return 0;
}
constexpr int itt_event_end(int) noexcept
{
    return 0;
}

constexpr void itt_metadata_add(
    ___itt_domain*, ___itt_id*, ___itt_string_handle*, std::uint64_t) noexcept
{
}
constexpr void itt_metadata_add(
    ___itt_domain*, ___itt_id*, ___itt_string_handle*, double) noexcept
{
}
constexpr void itt_metadata_add(
    ___itt_domain*, ___itt_id*, ___itt_string_handle*, char const*) noexcept
{
}
constexpr void itt_metadata_add(
    ___itt_domain*, ___itt_id*, ___itt_string_handle*, void const*) noexcept
{
}

//////////////////////////////////////////////////////////////////////////////
namespace hpx::threads {

    struct thread_description;
}    // namespace hpx::threads

namespace hpx::util::itt {

    struct stack_context
    {
        stack_context() = default;
        ~stack_context() = default;
    };

    struct caller_context
    {
        constexpr explicit caller_context(stack_context&) noexcept {}
        ~caller_context() = default;
    };

    //////////////////////////////////////////////////////////////////////////
    struct domain
    {
        HPX_NON_COPYABLE(domain);

        constexpr explicit domain(char const*) noexcept {}
        domain() = default;
        ~domain() = default;
    };

    struct thread_domain : domain
    {
        HPX_NON_COPYABLE(thread_domain);

        thread_domain() = default;
        ~thread_domain() = default;
    };

    struct id
    {
        constexpr id(domain const&, void*, unsigned long = 0) noexcept {}
        ~id() = default;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct frame_context
    {
        constexpr explicit frame_context(domain const&, id* = nullptr) noexcept
        {
        }
        ~frame_context() = default;
    };

    struct undo_frame_context
    {
        constexpr explicit undo_frame_context(frame_context const&) noexcept {}
        ~undo_frame_context() = default;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct mark_context
    {
        constexpr explicit mark_context(char const*) noexcept {}
        ~mark_context() = default;
    };

    struct undo_mark_context
    {
        constexpr explicit undo_mark_context(mark_context const&) noexcept {}
        ~undo_mark_context() = default;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct string_handle
    {
        constexpr explicit string_handle(char const* = nullptr) noexcept {}
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
        constexpr explicit event(char const*) noexcept {}
    };

    struct mark_event
    {
        constexpr explicit mark_event(event const&) noexcept {}
        ~mark_event() = default;
    };

    constexpr void event_tick(event const&) noexcept {}
}    // namespace hpx::util::itt

#endif    // HPX_HAVE_ITTNOTIFY
