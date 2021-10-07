//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>

namespace hpx { namespace util {

    // Always provide function exports, which guarantees ABI compatibility of
    // Debug and Release builds.

#if defined(HPX_HAVE_VERIFY_LOCKS_GLOBALLY) || defined(HPX_LOCAL_EXPORTS)
    HPX_LOCAL_EXPORT bool register_lock_globally(void const* lock);
    HPX_LOCAL_EXPORT bool unregister_lock_globally(void const* lock);
    HPX_LOCAL_EXPORT void enable_global_lock_detection();
    HPX_LOCAL_EXPORT void disable_global_lock_detection();
#else
    inline bool register_lock_globally(void const*)
    {
        return true;
    }
    inline bool unregister_lock_globally(void const*)
    {
        return true;
    }
    inline void enable_global_lock_detection() {}
    inline void disable_global_lock_detection() {}
#endif

}}    // namespace hpx::util
