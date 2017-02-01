//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ITT_NOTIFY_AUG_17_2010_1237PM)
#define HPX_UTIL_ITT_NOTIFY_AUG_17_2010_1237PM

#include <hpx/config.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>

struct ___itt_caller;
struct ___itt_string_handle;
struct ___itt_domain;
struct ___itt_id;
typedef void* __itt_heap_function;
struct ___itt_counter;

///////////////////////////////////////////////////////////////////////////////
#define HPX_ITT_SYNC_CREATE(obj, type, name)  itt_sync_create(obj, type, name)
#define HPX_ITT_SYNC_RENAME(obj, name)        itt_sync_rename(obj, name)
#define HPX_ITT_SYNC_PREPARE(obj)             itt_sync_prepare(obj)
#define HPX_ITT_SYNC_CANCEL(obj)              itt_sync_cancel(obj)
#define HPX_ITT_SYNC_ACQUIRED(obj)            itt_sync_acquired(obj)
#define HPX_ITT_SYNC_RELEASING(obj)           itt_sync_releasing(obj)
#define HPX_ITT_SYNC_RELEASED(obj)            itt_sync_released(obj)
#define HPX_ITT_SYNC_DESTROY(obj)             itt_sync_destroy(obj)

#define HPX_ITT_STACK_CREATE(ctx)             ctx = itt_stack_create()
#define HPX_ITT_STACK_CALLEE_ENTER(ctx)       itt_stack_enter(ctx)
#define HPX_ITT_STACK_CALLEE_LEAVE(ctx)       itt_stack_leave(ctx)
#define HPX_ITT_STACK_DESTROY(ctx)            itt_stack_destroy(ctx)

#define HPX_ITT_FRAME_BEGIN(frame, id)        itt_frame_begin(frame, id)
#define HPX_ITT_FRAME_END(frame, id)          itt_frame_end(frame, id)

#define HPX_ITT_MARK_CREATE(mark, name)       mark = itt_mark_create(name)
#define HPX_ITT_MARK_OFF(mark)                itt_mark_off(mark)
#define HPX_ITT_MARK(mark, parameter)         itt_mark(mark, parameter)

#define HPX_ITT_THREAD_SET_NAME(name)         itt_thread_set_name(name)
#define HPX_ITT_THREAD_IGNORE()               itt_thread_ignore()

#define HPX_ITT_TASK_BEGIN(domain, name)      itt_task_begin(domain, name)
#define HPX_ITT_TASK_BEGIN_ID(domain, id, name) itt_task_begin(domain, id, name)
#define HPX_ITT_TASK_END(domain)              itt_task_end(domain)

#define HPX_ITT_DOMAIN_CREATE(name)           itt_domain_create(name)
#define HPX_ITT_STRING_HANDLE_CREATE(name)    itt_string_handle_create(name)

#define HPX_ITT_MAKE_ID(addr, extra)          itt_make_id(addr, extra)
#define HPX_ITT_ID_CREATE(domain, id)         itt_id_create(domain, id)
#define HPX_ITT_ID_DESTROY(id)                itt_id_destroy(id)

#define HPX_ITT_HEAP_FUNCTION_CREATE(name, domain)                            \
    itt_heap_function_create(name, domain)                                    \
/**/
#define HPX_ITT_HEAP_ALLOCATE_BEGIN(f, size, initialized)                     \
    itt_heap_allocate_begin(f, size, initialized)                             \
/**/
#define HPX_ITT_HEAP_ALLOCATE_END(f, addr, size, initialized)                 \
    itt_heap_allocate_end(f, addr, size, initialized)                         \
/**/
#define HPX_ITT_HEAP_FREE_BEGIN(f, addr)      itt_heap_free_begin(f, addr)
#define HPX_ITT_HEAP_FREE_END(f, addr)        itt_heap_free_end(f, addr)
#define HPX_ITT_HEAP_REALLOCATE_BEGIN(f, addr, new_size, initialized)         \
    itt_heap_reallocate_begin(f, addr, new_size, initialized)                 \
/**/
#define HPX_ITT_HEAP_REALLOCATE_END(f, addr, new_addr, new_size, initialized) \
    itt_heap_reallocate_end(f, addr, new_addr, new_size, initialized)         \
/**/
#define HPX_ITT_HEAP_INTERNAL_ACCESS_BEGIN()  itt_heap_internal_access_begin()
#define HPX_ITT_HEAP_INTERNAL_ACCESS_END()    itt_heap_internal_access_end()

#define HPX_ITT_COUNTER_CREATE(name, domain)                                  \
    itt_counter_create(name, domain)                                          \
/**/
#define HPX_ITT_COUNTER_CREATE_TYPED(name, domain, type)                      \
    itt_counter_create_typed(name, domain, type)                              \
/**/
#define HPX_ITT_COUNTER_SET_VALUE(id, value_ptr)                              \
    itt_counter_set_value(id, value_ptr)                                      \
/**/
#define HPX_ITT_COUNTER_DESTROY(id)             itt_counter_destroy(id)

#define HPX_ITT_METADATA_ADD(domain, id, key, data)                           \
    itt_metadata_add(domain, id, key, data)                                   \
/**/

///////////////////////////////////////////////////////////////////////////////
// decide whether to use the ITT notify API if it's available

#if HPX_HAVE_ITTNOTIFY != 0
extern bool use_ittnotify_api;

///////////////////////////////////////////////////////////////////////////////
HPX_EXPORT void itt_sync_create(void* addr, const char* objtype, const char* objname);
HPX_EXPORT void itt_sync_rename(void* addr, const char* name);
HPX_EXPORT void itt_sync_prepare(void* addr);
HPX_EXPORT void itt_sync_acquired(void* addr);
HPX_EXPORT void itt_sync_cancel(void* addr);
HPX_EXPORT void itt_sync_releasing(void* addr);
HPX_EXPORT void itt_sync_released(void* addr);
HPX_EXPORT void itt_sync_destroy(void* addr);

HPX_EXPORT ___itt_caller* itt_stack_create();
HPX_EXPORT void itt_stack_enter(___itt_caller* ctx);
HPX_EXPORT void itt_stack_leave(___itt_caller* ctx);
HPX_EXPORT void itt_stack_destroy(___itt_caller* ctx);

HPX_EXPORT void itt_frame_begin(___itt_domain const* frame, ___itt_id* id);
HPX_EXPORT void itt_frame_end(___itt_domain const* frame, ___itt_id* id);

HPX_EXPORT int itt_mark_create(char const*);
HPX_EXPORT void itt_mark_off(int mark);
HPX_EXPORT void itt_mark(int mark, char const*);

HPX_EXPORT void itt_thread_set_name(char const*);
HPX_EXPORT void itt_thread_ignore();

HPX_EXPORT void itt_task_begin(___itt_domain const*, ___itt_string_handle*);
HPX_EXPORT void itt_task_begin(___itt_domain const*, ___itt_id*,
    ___itt_string_handle*);
HPX_EXPORT void itt_task_end(___itt_domain const*);

HPX_EXPORT ___itt_domain* itt_domain_create(char const*);
HPX_EXPORT ___itt_string_handle* itt_string_handle_create(char const*);

HPX_EXPORT ___itt_id* itt_make_id(void*, std::size_t);
HPX_EXPORT void itt_id_create(___itt_domain const*, ___itt_id* id);
HPX_EXPORT void itt_id_destroy(___itt_id* id);

HPX_EXPORT __itt_heap_function itt_heap_function_create(const char*, const char*);
HPX_EXPORT void itt_heap_allocate_begin(__itt_heap_function, std::size_t, int);
HPX_EXPORT void itt_heap_allocate_end(__itt_heap_function, void**, std::size_t, int);
HPX_EXPORT void itt_heap_free_begin(__itt_heap_function, void*);
HPX_EXPORT void itt_heap_free_end(__itt_heap_function, void*);
HPX_EXPORT void itt_heap_reallocate_begin(__itt_heap_function, void*, std::size_t, int);
HPX_EXPORT void itt_heap_reallocate_end(__itt_heap_function, void*, void**,
           std::size_t, int);
HPX_EXPORT void itt_heap_internal_access_begin();
HPX_EXPORT void itt_heap_internal_access_end();

HPX_EXPORT ___itt_counter* itt_counter_create(char const*, char const*);
HPX_EXPORT ___itt_counter* itt_counter_create_typed(char const*, char const*, int);
HPX_EXPORT void itt_counter_destroy(___itt_counter*);
HPX_EXPORT void itt_counter_set_value(___itt_counter*, void *);

HPX_EXPORT int itt_event_create(char const *name, int namelen);
HPX_EXPORT int itt_event_start(int evnt);
HPX_EXPORT int itt_event_end(int evnt);

HPX_EXPORT void itt_metadata_add(___itt_domain* domain, ___itt_id* id,
    ___itt_string_handle* key, std::uint64_t const& data);
HPX_EXPORT void itt_metadata_add(___itt_domain* domain, ___itt_id* id,
    ___itt_string_handle* key, double const& data);
HPX_EXPORT void itt_metadata_add(___itt_domain* domain, ___itt_id* id,
    ___itt_string_handle* key, char const* data);
HPX_EXPORT void itt_metadata_add(___itt_domain* domain, ___itt_id* id,
    ___itt_string_handle* key, void const* data);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    struct thread_description;
}}

namespace hpx { namespace util { namespace itt
{
    struct stack_context
    {
        stack_context()
          : itt_context_(0)
        {
            HPX_ITT_STACK_CREATE(itt_context_);
        }
        ~stack_context()
        {
            if (itt_context_) HPX_ITT_STACK_DESTROY(itt_context_);
        }

        stack_context(stack_context const& rhs) = delete;
        stack_context(stack_context && rhs)
          : itt_context_(rhs.itt_context_)
        {
            rhs.itt_context_ = nullptr;
        }

        stack_context& operator=(stack_context const& rhs) = delete;
        stack_context& operator=(stack_context && rhs)
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
        HPX_EXPORT domain(char const* name);

        ___itt_domain* domain_;
    };

    struct id
    {
        id (domain const& domain, void* addr, unsigned long extra = 0)
        {
            id_ = HPX_ITT_MAKE_ID(addr, extra);
            HPX_ITT_ID_CREATE(domain.domain_, id_);
        }
        ~id()
        {
            HPX_ITT_ID_DESTROY(id_);
        }

        id(id const& rhs) = delete;
        id(id && rhs)
          : id_(rhs.id_)
        {
            rhs.id_ = nullptr;
        }

        id& operator=(id const& rhs) = delete;
        id& operator=(id && rhs)
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
        frame_context(domain const& domain, id* ident = nullptr)
          : domain_(domain), ident_(ident)
        {
            HPX_ITT_FRAME_BEGIN(domain_.domain_, ident_? ident_->id_ : 0);
        }
        ~frame_context()
        {
            HPX_ITT_FRAME_END(domain_.domain_, ident_ ? ident_->id_ : 0);
        }

        domain const& domain_;
        id* ident_;
    };

    struct undo_frame_context
    {
        undo_frame_context(frame_context& frame)
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
        mark_context(char const* name)
          : itt_mark_(0), name_(name)
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
        undo_mark_context(mark_context& mark)
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
        string_handle() HPX_NOEXCEPT
          : handle_(0)
        {}
        string_handle(char const* s)
          : handle_(HPX_ITT_STRING_HANDLE_CREATE(s))
        {}
        string_handle(___itt_string_handle* h) HPX_NOEXCEPT
          : handle_(h)
        {}

        string_handle& operator=(___itt_string_handle* h) HPX_NOEXCEPT
        {
            handle_ = h;
            return *this;
        }

        explicit operator bool() const HPX_NOEXCEPT
        {
            return handle_ != 0;
        }

        ___itt_string_handle* handle_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct task
    {
        HPX_EXPORT task(domain const&, util::thread_description const&);
        ~task()
        {
            HPX_ITT_TASK_END(domain_.domain_);
        }

        void add_metadata(string_handle const& name, std::uint64_t val)
        {
            HPX_ITT_METADATA_ADD(domain_.domain_, id_, name.handle_, val);
        }
        void add_metadata(string_handle const& name, double val)
        {
            HPX_ITT_METADATA_ADD(domain_.domain_, id_, name.handle_, val);
        }
        void add_metadata(string_handle const& name, char const* val)
        {
            HPX_ITT_METADATA_ADD(domain_.domain_, id_, name.handle_, val);
        }
        template <typename T>
        void add_metadata(string_handle const& name, T const& val)
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
        heap_function(char const* name, char const* domain)
          : heap_function_(HPX_ITT_HEAP_FUNCTION_CREATE(name, domain))
        {}

        __itt_heap_function heap_function_;
    };

    struct heap_internal_access
    {
        heap_internal_access()
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
                      int init)
          : heap_function_(heap_function),
            addr_(reinterpret_cast<void**&>(addr)), size_(size), init_(init)
        {
            HPX_ITT_HEAP_ALLOCATE_BEGIN(heap_function_.heap_function_, size_, init_);
        }

        ~heap_allocate()
        {
            HPX_ITT_HEAP_ALLOCATE_END(heap_function_.heap_function_, addr_,
                size_, init_);
        }

    private:
        heap_function& heap_function_;
        void**& addr_;
        std::size_t size_;
        int init_;
    };

    struct heap_free
    {
        heap_free(heap_function& heap_function, void* addr)
          : heap_function_(heap_function), addr_(addr)
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
        counter(char const* name, char const* domain)
          : id_(HPX_ITT_COUNTER_CREATE(name, domain))
        {
        }
        counter(char const* name, char const* domain, int type)
          : id_(HPX_ITT_COUNTER_CREATE_TYPED(name, domain, type))
        {
        }
        ~counter()
        {
            if (id_) HPX_ITT_COUNTER_DESTROY(id_);
        }

        template <typename T>
        void set_value(T const& value)
        {
            if (id_) HPX_ITT_COUNTER_SET_VALUE(id_, (void*)&value);
        }

        counter(counter const& rhs) = delete;
        counter(counter && rhs)
          : id_(rhs.id_)
        {
            rhs.id_ = nullptr;
        }

        counter& operator=(counter const& rhs) = delete;
        counter& operator=(counter && rhs)
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
        event(char const* name)
          : event_(itt_event_create(name, (int)std::strlen(name)))
        {}

        void start() const
        {
            itt_event_start(event_);
        }

        void end() const
        {
            itt_event_end(event_);
        }

    private:
        int event_;
    };

    struct mark_event
    {
        mark_event(event const& e)
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

    inline void event_tick(event const& e)
    {
        e.start();
    }
}}}

#else

inline void itt_sync_create(void*, const char*, const char*) {}
inline void itt_sync_rename(void*, const char*) {}
inline void itt_sync_prepare(void*) {}
inline void itt_sync_acquired(void*) {}
inline void itt_sync_cancel(void*) {}
inline void itt_sync_releasing(void*) {}
inline void itt_sync_released(void*) {}
inline void itt_sync_destroy(void*) {}

inline ___itt_caller* itt_stack_create() { return nullptr; }
inline void itt_stack_enter(___itt_caller*) {}
inline void itt_stack_leave(___itt_caller*) {}
inline void itt_stack_destroy(___itt_caller*) {}

inline void itt_frame_begin(___itt_domain const*, ___itt_id*) {}
inline void itt_frame_end(___itt_domain const*, ___itt_id*) {}

inline int itt_mark_create(char const*) { return 0; }
inline void itt_mark_off(int) {}
inline void itt_mark(int, char const*) {}

inline void itt_thread_set_name(char const*) {}
inline void itt_thread_ignore() {}

inline void itt_task_begin(___itt_domain const*, ___itt_string_handle*) {}
inline void itt_task_begin(___itt_domain const*, ___itt_id*,
    ___itt_string_handle*) {}
inline void itt_task_end(___itt_domain const*) {}

inline ___itt_domain* itt_domain_create(char const*) { return nullptr; }
inline ___itt_string_handle* itt_string_handle_create(char const*) { return nullptr; }

inline ___itt_id* itt_make_id(void*, unsigned long) { return nullptr; }
inline void itt_id_create(___itt_domain const*, ___itt_id*) {}
inline void itt_id_destroy(___itt_id*) {}

inline __itt_heap_function itt_heap_function_create(const char*,
            const char*) { return nullptr; }
inline void itt_heap_allocate_begin(__itt_heap_function, std::size_t, int) {}
inline void itt_heap_allocate_end(__itt_heap_function, void**, std::size_t, int) {}
inline void itt_heap_free_begin(__itt_heap_function, void*) {}
inline void itt_heap_free_end(__itt_heap_function, void*) {}
inline void itt_heap_reallocate_begin(__itt_heap_function, void*, std::size_t, int) {}
inline void itt_heap_reallocate_end(__itt_heap_function, void*, void**,
            std::size_t, int) {}
inline void itt_heap_internal_access_begin() {}
inline void itt_heap_internal_access_end() {}

inline ___itt_counter* itt_counter_create(char const*, char const*) { return nullptr; }
inline ___itt_counter* itt_counter_create_typed(char const*, char const*, int)
    { return nullptr; }
inline void itt_counter_destroy(___itt_counter*) {}
inline void itt_counter_set_value(___itt_counter*, void *) {}

inline int itt_event_create(char const *name, int namelen)  { return 0; }
inline int itt_event_start(int evnt) { return 0; }
inline int itt_event_end(int evnt) { return 0; }

inline void itt_metadata_add(___itt_domain* domain, ___itt_id* id,
    ___itt_string_handle* key, std::uint64_t const& data) {}
inline void itt_metadata_add(___itt_domain* domain, ___itt_id* id,
    ___itt_string_handle* key, double const& data) {}
inline void itt_metadata_add(___itt_domain* domain, ___itt_id* id,
    ___itt_string_handle* key, char const* data) {}
inline void itt_metadata_add(___itt_domain* domain, ___itt_id* id,
    ___itt_string_handle* key, void const* data) {}

//////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    struct thread_description;
}}

namespace hpx { namespace util { namespace itt
{
    struct stack_context
    {
        stack_context() {}
        ~stack_context() {}
    };

    struct caller_context
    {
        caller_context(stack_context&) {}
        ~caller_context() {}
    };

    //////////////////////////////////////////////////////////////////////////
    struct domain
    {
        domain(char const*) {}
        ~domain() {}
    };

    struct id
    {
        id (domain const& /*domain*/, void* /*addr*/, unsigned long /*extra*/ = 0) {}
        ~id() {}
    };

    ///////////////////////////////////////////////////////////////////////////
    struct frame_context
    {
        frame_context(domain const&, id* = nullptr) {}
        ~frame_context() {}
    };

    struct undo_frame_context
    {
        undo_frame_context(frame_context const&) {}
        ~undo_frame_context() {}
    };

    ///////////////////////////////////////////////////////////////////////////
    struct mark_context
    {
        mark_context(char const*) {}
        ~mark_context() {}
    };

    struct undo_mark_context
    {
        undo_mark_context(mark_context const&) {}
        ~undo_mark_context() {}
    };

    ///////////////////////////////////////////////////////////////////////////
    struct string_handle
    {
        string_handle(char const* s) {}
    };

    //////////////////////////////////////////////////////////////////////////
    struct task
    {
        task(domain const&, util::thread_description const&) {}
        ~task() {}
    };

    ///////////////////////////////////////////////////////////////////////////
    struct heap_function
    {
        heap_function(char const*, char const*) {}
        ~heap_function() {}
    };

    struct heap_allocate
    {
        template <typename T>
        heap_allocate(heap_function& /*heap_function*/, T**, std::size_t, int) {}
        ~heap_allocate() {}
    };

    struct heap_free
    {
        heap_free(heap_function& /*heap_function*/, void*) {}
        ~heap_free() {}
    };

    struct heap_internal_access
    {
        heap_internal_access() {}
        ~heap_internal_access() {}
    };

    struct counter
    {
        counter(char const* /*name*/, char const* /*domain*/) {}
        ~counter() {}
    };

    struct event
    {
        event(char const*) {}
    };

    struct mark_event
    {
        mark_event(event const&) {}
        ~mark_event() {}
    };

    inline void event_tick(event const&) {}
}}}

#endif // HPX_HAVE_ITTNOTIFY

#endif
