//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

// From file hpx/schedulers/queue_numa_holder.hpp
#if !defined(QUEUE_HOLDER_NUMA_DEBUG)
#define QUEUE_HOLDER_NUMA_DEBUG false
#endif

// From file hpx/schedulers/queue_holder_thread.hpp
#if !defined(QUEUE_HOLDER_THREAD_DEBUG)
#define QUEUE_HOLDER_THREAD_DEBUG false
#endif

// From file hpx/schedulers/shared_priority_scheduler.hpp
#if !defined(SHARED_PRIORITY_SCHEDULER_DEBUG)
#define SHARED_PRIORITY_SCHEDULER_DEBUG false
#endif

#if defined(__linux) || defined(linux) || defined(__linux__)
#define SHARED_PRIORITY_SCHEDULER_LINUX
#endif

// From file hpx/schedulers/thread_queue_mc.hpp
#if !defined(THREAD_QUEUE_MC_DEBUG)
#define THREAD_QUEUE_MC_DEBUG false
#endif
