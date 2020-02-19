//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_THREAD_INIT_DATA_HPP
#define HPX_RUNTIME_THREADS_THREAD_INIT_DATA_HPP

#include <hpx/config.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/threading_base/thread_description.hpp>
#include <hpx/threading_base/threading_base_fwd.hpp>
#if defined(HPX_HAVE_APEX)
#include <hpx/threading_base/external_timer.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace hpx { namespace threads {
    HPX_API_EXPORT std::ptrdiff_t get_default_stack_size();
    HPX_API_EXPORT std::ptrdiff_t get_stack_size(thread_stacksize);

    ///////////////////////////////////////////////////////////////////////////
    class thread_init_data
    {
    public:
        thread_init_data()
          : func()
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
          , description()
#endif
#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
          , parent_locality_id(0)
          , parent_id(nullptr)
          , parent_phase(0)
#endif
#ifdef HPX_HAVE_APEX
          , timer_data(nullptr)
#endif
          , priority(thread_priority_normal)
          , schedulehint()
          , stacksize(HPX_SMALL_STACK_SIZE)
          , scheduler_base(nullptr)
        {
        }

        thread_init_data& operator=(thread_init_data&& rhs)
        {
            func = std::move(rhs.func);
            priority = rhs.priority;
            schedulehint = rhs.schedulehint;
            stacksize = rhs.stacksize;
            scheduler_base = rhs.scheduler_base;
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            description = rhs.description;
#endif
#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
            parent_locality_id = rhs.parent_locality_id;
            parent_id = rhs.parent_id;
            parent_phase = rhs.parent_phase;
#endif
#ifdef HPX_HAVE_APEX
            // HPX_HAVE_APEX forces the HPX_HAVE_THREAD_DESCRIPTION
            // and HPX_HAVE_THREAD_PARENT_REFERENCE settings to be on
            timer_data = rhs.timer_data;
#endif
            return *this;
        }

        thread_init_data(thread_init_data&& rhs)
          : func(std::move(rhs.func))
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
          , description(rhs.description)
#endif
#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
          , parent_locality_id(rhs.parent_locality_id)
          , parent_id(rhs.parent_id)
          , parent_phase(rhs.parent_phase)
#endif
#ifdef HPX_HAVE_APEX
          // HPX_HAVE_APEX forces the HPX_HAVE_THREAD_DESCRIPTION and
          // HPX_HAVE_THREAD_PARENT_REFERENCE settings to be on
          , timer_data(util::external_timer::new_task(
                description, parent_locality_id, parent_id))
#endif
          , priority(rhs.priority)
          , schedulehint(rhs.schedulehint)
          , stacksize(rhs.stacksize)
          , scheduler_base(rhs.scheduler_base)
        {
            if (stacksize == 0)
                stacksize = HPX_SMALL_STACK_SIZE;
        }

        template <typename F>
        thread_init_data(F&& f, util::thread_description const& desc,
            thread_priority priority_ = thread_priority_normal,
            thread_schedule_hint os_thread = thread_schedule_hint(),
            std::ptrdiff_t stacksize_ = HPX_SMALL_STACK_SIZE,
            policies::scheduler_base* scheduler_base_ = nullptr)
          : func(std::forward<F>(f))
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
          , description(desc)
#endif
#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
          , parent_locality_id(0)
          , parent_id(nullptr)
          , parent_phase(0)
#endif
#ifdef HPX_HAVE_APEX
          // HPX_HAVE_APEX forces the HPX_HAVE_THREAD_DESCRIPTION and
          // HPX_HAVE_THREAD_PARENT_REFERENCE settings to be on
          , timer_data(util::external_timer::new_task(
                description, parent_locality_id, parent_id))
#endif
          , priority(priority_)
          , schedulehint(os_thread)
          , stacksize(stacksize_)
          , scheduler_base(scheduler_base_)
        {
            if (stacksize == 0)
                stacksize = HPX_SMALL_STACK_SIZE;
        }

        threads::thread_function_type func;

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
        util::thread_description description;
#endif
#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
        std::uint32_t parent_locality_id;
        threads::thread_id_type parent_id;
        std::size_t parent_phase;
#endif
#ifdef HPX_HAVE_APEX
        // HPX_HAVE_APEX forces the HPX_HAVE_THREAD_DESCRIPTION and
        // HPX_HAVE_THREAD_PARENT_REFERENCE settings to be on
        std::shared_ptr<util::external_timer::task_wrapper> timer_data;
#endif

        thread_priority priority;
        thread_schedule_hint schedulehint;
        std::ptrdiff_t stacksize;

        policies::scheduler_base* scheduler_base;
    };
}}    // namespace hpx::threads

#endif /*HPX_RUNTIME_THREADS_THREAD_INIT_DATA_HPP*/
