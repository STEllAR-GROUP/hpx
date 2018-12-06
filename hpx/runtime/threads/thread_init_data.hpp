//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_THREAD_INIT_DATA_HPP
#define HPX_RUNTIME_THREADS_THREAD_INIT_DATA_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
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
#if defined(HPX_HAVE_THREAD_TARGET_ADDRESS)
            lva(0),
#endif
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
            stacksize(get_default_stack_size()),
            scheduler_base(nullptr)
        {}

        thread_init_data(thread_init_data&& rhs)
          : func(std::move(rhs.func)),
#if defined(HPX_HAVE_THREAD_TARGET_ADDRESS)
            lva(rhs.lva),
#endif
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            description(rhs.description),
#endif
#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
            parent_locality_id(rhs.parent_locality_id), parent_id(rhs.parent_id),
            parent_phase(rhs.parent_phase),
#endif
#ifdef HPX_HAVE_APEX
        /* HPX_HAVE_APEX forces the HPX_HAVE_THREAD_DESCRIPTION
         * and HPX_HAVE_THREAD_PARENT_REFERENCE settings to be on */
            apex_data(apex_new_task(description, parent_locality_id, parent_id )),
#endif
            priority(rhs.priority),
            schedulehint(rhs.schedulehint),
            stacksize(rhs.stacksize),
            scheduler_base(rhs.scheduler_base)
        {
            if (stacksize == 0)
                stacksize = get_default_stack_size();
        }

        template <typename F>
        thread_init_data(F && f, util::thread_description const& desc,
                naming::address_type lva_ = 0,
                thread_priority priority_ = thread_priority_normal,
                thread_schedule_hint os_thread = thread_schedule_hint(),
                std::ptrdiff_t stacksize_ = std::ptrdiff_t(-1),
                policies::scheduler_base* scheduler_base_ = nullptr)
          : func(std::forward<F>(f)),
#if defined(HPX_HAVE_THREAD_TARGET_ADDRESS)
            lva(lva_),
#endif
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            description(desc),
#endif
#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
            parent_locality_id(0), parent_id(nullptr), parent_phase(0),
#endif
#ifdef HPX_HAVE_APEX
        /* HPX_HAVE_APEX forces the HPX_HAVE_THREAD_DESCRIPTION
         * and HPX_HAVE_THREAD_PARENT_REFERENCE settings to be on */
            apex_data(apex_new_task(description,parent_locality_id,parent_id)),
#endif
            priority(priority_), schedulehint(os_thread),
            stacksize(stacksize_ == std::ptrdiff_t(-1) ?
                get_default_stack_size() : stacksize_),
            scheduler_base(scheduler_base_)
        {
            if (stacksize == 0)
                stacksize = get_default_stack_size();
        }

        threads::thread_function_type func;

#if defined(HPX_HAVE_THREAD_TARGET_ADDRESS)
        naming::address_type lva;
#endif
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
