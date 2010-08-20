//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ITT_NOTIFY_AUG_17_2010_1237PM)
#define HPX_UTIL_ITT_NOTIFY_AUG_17_2010_1237PM

#include <hpx/config.hpp>

struct ___itt_caller;

#if defined(HPX_USE_ITT)

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

#else

inline void itt_sync_create(void* addr, const char* objtype, const char* objname) {}
inline void itt_sync_rename(void* addr, const char* name) {}
inline void itt_sync_prepare(void* addr) {}
inline void itt_sync_acquired(void* addr) {}
inline void itt_sync_cancel(void* addr) {}
inline void itt_sync_releasing(void* addr) {}
inline void itt_sync_released(void* addr) {}
inline void itt_sync_destroy(void* addr) {}

inline ___itt_caller* itt_stack_create() {}
inline void itt_stack_enter(___itt_caller* ctx) {}
inline void itt_stack_leave(___itt_caller* ctx) {}
inline void itt_stack_destroy(___itt_caller* ctx) {}

#endif // HPX_USE_ITT

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

#endif
