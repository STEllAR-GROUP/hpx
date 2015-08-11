//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_REGISTER_LOCKS_GLOBALLY_JAN_27_2014_0607PM)
#define HPX_UTIL_REGISTER_LOCKS_GLOBALLY_JAN_27_2014_0607PM

#include <hpx/config.hpp>

namespace hpx { namespace util
{
    // Always provide function exports, which guarantees ABI compatibility of
    // Debug and Release builds.

#if defined(HPX_HAVE_VERIFY_LOCKS_GLOBALLY) || defined(HPX_EXPORTS)
    HPX_API_EXPORT bool register_lock_globally(void const* lock);
    HPX_API_EXPORT bool unregister_lock_globally(void const* lock);
    HPX_API_EXPORT void enable_global_lock_detection();
#else
    inline bool register_lock_globally(void const*)
    {
        return true;
    }
    inline bool unregister_lock_globally(void const*)
    {
        return true;
    }
    inline void enable_global_lock_detection()
    {
    }
#endif

}}

#endif

