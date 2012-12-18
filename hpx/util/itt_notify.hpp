//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ITT_NOTIFY_AUG_17_2010_1237PM)
#define HPX_UTIL_ITT_NOTIFY_AUG_17_2010_1237PM

#include <hpx/config.hpp>

struct ___itt_caller;
struct __itt_frame_t;

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

#define HPX_ITT_FRAME_CREATE(frame, name)     frame = itt_frame_create(name)
#define HPX_ITT_FRAME_BEGIN(frame)            itt_frame_begin(frame)
#define HPX_ITT_FRAME_END(frame)              itt_frame_end(frame)
#define HPX_ITT_FRAME_DESTROY(frame)          itt_frame_destroy(frame)

#define HPX_ITT_MARK_CREATE(mark, name)       mark = itt_mark_create(name)
#define HPX_ITT_MARK_OFF(mark)                itt_mark_off(mark)
#define HPX_ITT_MARK(mark, parameter)         itt_mark(mark, parameter)

#define HPX_ITT_THREAD_SET_NAME(name)         itt_thread_set_name(name)
#define HPX_ITT_THREAD_IGNORE()               itt_thread_ignore()

///////////////////////////////////////////////////////////////////////////////
// decide whether to use the ITT notify API if it's available

#if HPX_USE_ITTNOTIFY != 0
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

HPX_EXPORT __itt_frame_t* itt_frame_create(char const*);
HPX_EXPORT void itt_frame_begin(__itt_frame_t* frame);
HPX_EXPORT void itt_frame_end(__itt_frame_t* frame);
HPX_EXPORT void itt_frame_destroy(__itt_frame_t* frame);

HPX_EXPORT int itt_mark_create(char const*);
HPX_EXPORT void itt_mark_off(int mark);
HPX_EXPORT void itt_mark(int mark, char const*);

HPX_EXPORT void itt_thread_set_name(char const*);
HPX_EXPORT void itt_thread_ignore();

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

    ///////////////////////////////////////////////////////////////////////////
    struct frame_context
    {
        frame_context(char const* name)
          : itt_frame_(0)
        {
            HPX_ITT_FRAME_CREATE(itt_frame_, name);
            HPX_ITT_FRAME_BEGIN(itt_frame_);
        }
        ~frame_context()
        {
            HPX_ITT_FRAME_END(itt_frame_);
            HPX_ITT_FRAME_DESTROY(itt_frame_);
        }

        struct __itt_frame_t* itt_frame_;
    };

    struct undo_frame_context
    {
        undo_frame_context(frame_context& frame)
          : frame_(frame)
        {
            HPX_ITT_FRAME_END(frame_.itt_frame_);
        }
        ~undo_frame_context()
        {
            HPX_ITT_FRAME_BEGIN(frame_.itt_frame_);
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

inline __itt_frame_t* itt_frame_create(char const*) { return 0; }
inline void itt_frame_begin(__itt_frame_t*) {}
inline void itt_frame_end(__itt_frame_t*) {}
inline void itt_frame_destroy(__itt_frame_t*) {}

inline int itt_mark_create(char const*) { return 0; }
inline void itt_mark_off(int) {}
inline void itt_mark(int, char const*) {}

inline void itt_thread_set_name(char const*) {}
inline void itt_thread_ignore() {}

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

    ///////////////////////////////////////////////////////////////////////////
    struct frame_context
    {
        frame_context(char const*) {}
        ~frame_context() {}
    };

    struct undo_frame_context
    {
        undo_frame_context(frame_context&) {}
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
        undo_mark_context(mark_context&) {}
        ~undo_mark_context() {}
    };
}}}

#endif // HPX_USE_ITTNOTIFY

#endif
