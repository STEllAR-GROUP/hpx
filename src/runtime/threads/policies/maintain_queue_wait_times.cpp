//  Copyright (c) 2005-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/threads/policies/maintain_queue_wait_times.hpp>

namespace hpx { namespace threads { namespace policies {
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
    ///////////////////////////////////////////////////////////////////////////
    // We globally control whether to collect wait times using this global bool.
    // It will be set by any of the related performance counters. Once set it
    // stays set, thus no race conditions will occur.
    HPX_EXPORT bool maintain_queue_wait_times = false;
#endif
}}}    // namespace hpx::threads::policies
