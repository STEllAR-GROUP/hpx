//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_SCHEDULERS_DEC_22_2011_0946PM)
#define HPX_THREADMANAGER_SCHEDULERS_DEC_22_2011_0946PM

#include <hpx/config.hpp>

#if defined(HPX_GLOBAL_SCHEDULER)
#include <hpx/runtime/threads/policies/global_queue_scheduler.hpp>
#endif
#if defined(HPX_LOCAL_SCHEDULER)
#include <hpx/runtime/threads/policies/local_queue_scheduler.hpp>
#endif
#if defined(HPX_ABP_SCHEDULER)
#include <hpx/runtime/threads/policies/abp_queue_scheduler.hpp>
#endif
#if defined(HPX_ABP_PRIORITY_SCHEDULER)
#include <hpx/runtime/threads/policies/abp_priority_queue_scheduler.hpp>
#endif
#include <hpx/runtime/threads/policies/local_priority_queue_scheduler.hpp>
#if defined(HPX_HIERARCHY_SCHEDULER)
#include <hpx/runtime/threads/policies/hierarchy_scheduler.hpp>
#endif
#if defined(HPX_PERIODIC_PRIORITY_SCHEDULER)
#include <hpx/runtime/threads/policies/periodic_priority_scheduler.hpp>
#endif

#endif
