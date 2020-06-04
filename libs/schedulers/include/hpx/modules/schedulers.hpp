//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_LOCAL_SCHEDULER)
#include <hpx/schedulers/local_queue_scheduler.hpp>
#endif
#if defined(HPX_HAVE_STATIC_SCHEDULER)
#include <hpx/schedulers/static_queue_scheduler.hpp>
#endif
#include <hpx/schedulers/local_priority_queue_scheduler.hpp>
#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
#include <hpx/schedulers/static_priority_queue_scheduler.hpp>
#endif
#if defined(HPX_HAVE_SHARED_PRIORITY_SCHEDULER)
#include <hpx/schedulers/shared_priority_queue_scheduler.hpp>
#endif
