//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_THREAD_INIT_DATA_HPP
#define HPX_RUNTIME_THREADS_THREAD_INIT_DATA_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/runtime/threads_fwd.hpp>
#include <hpx/util/thread_description.hpp>
#if defined(HPX_HAVE_APEX)
#include <hpx/util/apex.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <utility>

namespace hpx { namespace threads
{
    HPX_API_EXPORT std::ptrdiff_t get_default_stack_size();
    HPX_API_EXPORT std::ptrdiff_t get_stack_size(thread_stacksize);

    ///////////////////////////////////////////////////////////////////////////
    class thread_init_data
    {
    public:
        thread_init_data()
          : func(),
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            description(),
#endif
#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
            parent_locality_id(0), parent_id(nullptr), parent_phase(0),
#endif
#ifdef HPX_HAVE_APEX
            apex_data(nullptr),
#endif
            priority(thread_priority_normal),
            schedulehint(),
            stacksize(HPX_SMALL_STACK_SIZE),
            scheduler_base(nullptr)
        {}

        thread_init_data& operator=(thread_init_data&& rhs) {
            func            = std::move(rhs.func);
            priority        = rhs.priority;
            schedulehint    = rhs.schedulehint;
            stacksize       = rhs.stacksize;
            scheduler_base  = rhs.scheduler_base;
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
            if (description.kind() == description.data_type_description) {
                apex_data = util::apex_new_task(description.get_description(),
                        parent_id);
            } else {
                apex_data = util::apex_new_task(description.get_address(),
                        parent_id);
            }
#endif
            return *this;
        }

        thread_init_data(thread_init_data&& rhs)
          : func(std::move(rhs.func)),
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            description(rhs.description),
#endif
#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
            parent_locality_id(rhs.parent_locality_id), parent_id(rhs.parent_id),
            parent_phase(rhs.parent_phase),
#endif
            priority(rhs.priority),
            schedulehint(rhs.schedulehint),
            stacksize(rhs.stacksize),
            scheduler_base(rhs.scheduler_base)
        {
#ifdef HPX_HAVE_APEX
        /* HPX_HAVE_APEX forces the HPX_HAVE_THREAD_DESCRIPTION
         * and HPX_HAVE_THREAD_PARENT_REFERENCE settings to be on */
            if (description.kind() == description.data_type_description) {
                apex_data = util::apex_new_task(description.get_description(),
                        parent_id);
            } else {
                apex_data = util::apex_new_task(description.get_address(),
                        parent_id);
            }
#endif
            if (stacksize == 0)
                stacksize = HPX_SMALL_STACK_SIZE;
        }

        template <typename F>
        thread_init_data(F && f, util::thread_description const& desc,
                thread_priority priority_ = thread_priority_normal,
                thread_schedule_hint os_thread = thread_schedule_hint(),
                std::ptrdiff_t stacksize_ = HPX_SMALL_STACK_SIZE,
                policies::scheduler_base* scheduler_base_ = nullptr)
          : func(std::forward<F>(f)),
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            description(desc),
#endif
#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
            parent_locality_id(0), parent_id(nullptr), parent_phase(0),
#endif
            priority(priority_), schedulehint(os_thread),
            stacksize(stacksize_),
            scheduler_base(scheduler_base_)
        {
#ifdef HPX_HAVE_APEX
          /* HPX_HAVE_APEX forces the HPX_HAVE_THREAD_DESCRIPTION
           * and HPX_HAVE_THREAD_PARENT_REFERENCE settings to be on */
            if (description.kind() == description.data_type_description) {
                apex_data = util::apex_new_task(description.get_description(),
                            parent_id);
            } else {
                apex_data = util::apex_new_task(description.get_address(),
                            parent_id);
            }
#endif
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
        /* HPX_HAVE_APEX forces the HPX_HAVE_THREAD_DESCRIPTION
         * and HPX_HAVE_THREAD_PARENT_REFERENCE settings to be on */
        apex_task_wrapper apex_data;
#endif

        thread_priority priority;
        thread_schedule_hint schedulehint;
        std::ptrdiff_t stacksize;

        policies::scheduler_base* scheduler_base;
    };
}}

#endif /*HPX_RUNTIME_THREADS_THREAD_INIT_DATA_HPP*/
