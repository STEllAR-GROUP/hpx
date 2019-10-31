//  Copyright (c) 2005-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_POLICIES_MAINTAIN_QUEUE_WAIT_TIMES_HPP
#define HPX_RUNTIME_THREADS_POLICIES_MAINTAIN_QUEUE_WAIT_TIMES_HPP

#include <hpx/config.hpp>

namespace hpx { namespace threads { namespace policies {
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
    extern bool maintain_queue_wait_times;
#endif
}}}    // namespace hpx::threads::policies

#endif
