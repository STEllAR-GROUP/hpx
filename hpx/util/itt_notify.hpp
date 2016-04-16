//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ITT_NOTIFY_AUG_17_2010_1237PM)
#define HPX_UTIL_ITT_NOTIFY_AUG_17_2010_1237PM

#include <hpx/config.hpp>
#include <hpx/util/thread_description.hpp>

#include <cstddef>

struct ___itt_caller;
struct ___itt_string_handle;
struct ___itt_domain;
struct ___itt_id;
typedef void* __itt_heap_function;

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
#define HPX_ITT_TASK_END(domain)              itt_task_end(domain)

#define HPX_ITT_DOMAIN_CREATE(name)           itt_domain_create(name)
#define HPX_ITT_TASK_HANDLE_CREATE(name)      itt_task_handle_create(name)

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
HPX_EXPORT void itt_task_end(___itt_domain const*);

HPX_EXPORT ___itt_domain* itt_domain_create(char const*);
HPX_EXPORT ___itt_string_handle* itt_task_handle_create(char const*);

HPX_EXPORT ___itt_id* itt_make_id(void*, unsigned long);
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

///////////////////////////////////////////////////////////////////////////////
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
            HPX_ITT_STACK_DESTROY(itt_context_);
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
        domain(char const* name)
          : domain_(HPX_ITT_DOMAIN_CREATE(name))
        {}

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

        ___itt_id* id_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct frame_context
    {
        frame_context(domain const& domain, id* ident = 0)
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
            HPX_ITT_FRAME_END(frame_.domain_.domain_, NULL);
        }
        ~undo_frame_context()
        {
            HPX_ITT_FRAME_BEGIN(frame_.domain_.domain_, NULL);
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

    //////////////////////////////////////////////////////////////////////////
    struct task
    {
        task(domain const& domain, util::thread_description name)
          : domain_(domain)
        {
            if (name.kind() == util::thread_description::data_type_description)
            {
                HPX_ITT_TASK_BEGIN(domain_.domain_,
                    HPX_ITT_TASK_HANDLE_CREATE(name.get_description()));
            }
            else
            {
                HPX_ITT_TASK_BEGIN(domain_.domain_,
                    HPX_ITT_TASK_HANDLE_CREATE("address"));
            }
        }
        ~task()
        {
            HPX_ITT_TASK_END(domain_.domain_);
        }

        domain const& domain_;
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

inline ___itt_caller* itt_stack_create() { return 0; }
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
inline void itt_task_end(___itt_domain const*) {}

inline ___itt_domain* itt_domain_create(char const*) { return 0; }
inline ___itt_string_handle* itt_task_handle_create(char const*) { return 0; }

inline ___itt_id* itt_make_id(void*, unsigned long) { return 0; }
inline void itt_id_create(___itt_domain const*, ___itt_id*) {}
inline void itt_id_destroy(___itt_id*) {}

inline __itt_heap_function itt_heap_function_create(const char*,
            const char*) { return 0; }
inline void itt_heap_allocate_begin(__itt_heap_function, std::size_t, int) {}
inline void itt_heap_allocate_end(__itt_heap_function, void**, std::size_t, int) {}
inline void itt_heap_free_begin(__itt_heap_function, void*) {}
inline void itt_heap_free_end(__itt_heap_function, void*) {}
inline void itt_heap_reallocate_begin(__itt_heap_function, void*, std::size_t, int) {}
inline void itt_heap_reallocate_end(__itt_heap_function, void*, void**,
            std::size_t, int) {}
inline void itt_heap_internal_access_begin() {}
inline void itt_heap_internal_access_end() {}

//////////////////////////////////////////////////////////////////////////////
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
        frame_context(domain const&, id* = 0) {}
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
}}}

#endif // HPX_HAVE_ITTNOTIFY

#endif
