//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once // prevent multiple inclusions of this header file.

#include <hpx/config.hpp>
#include <hpx/runtime/get_locality_id.hpp>
#include <hpx/util/thread_description.hpp>
#include <hpx/runtime/get_num_localities.hpp>
#include <hpx/runtime/startup_function.hpp>

#ifdef HPX_HAVE_APEX
#include "apex_api.hpp"
#endif

namespace hpx { namespace util
{
#ifdef HPX_HAVE_APEX
    static void hpx_util_apex_init_startup(void)
    {
        apex::init(nullptr, hpx::get_locality_id(),
            hpx::get_initial_num_localities());
    }

    inline void apex_init()
    {
        hpx_util_apex_init_startup();
        //hpx::register_pre_startup_function(&hpx_util_apex_init_startup);
    }

    inline void apex_finalize()
    {
        apex::finalize();
    }

    inline void * apex_new_task(
                thread_description const& description,
                void * parent_task)
    {
        if (description.kind() ==
                thread_description::data_type_description) {
            return (void*)apex::new_task(description.get_description(),
                UINTMAX_MAX, (apex::task_wrapper*)parent_task);
        } else {
            return (void*)apex::new_task(description.get_address(),
                UINTMAX_MAX, (apex::task_wrapper*)parent_task);
        }
    }

    inline void * apex_update_task(void * wrapper,
                thread_description const& description)
    {
        if (description.kind() == thread_description::data_type_description) {
            return (void*)apex::update_task((apex::task_wrapper*)wrapper,
                description.get_description());
        } else {
            return (void*)apex::update_task((apex::task_wrapper*)wrapper,
                description.get_address());
        }
    }

    inline void * apex_update_task(void * wrapper, char const* name)
    {
        return (void*)apex::update_task((apex::task_wrapper*)wrapper, name);
    }

    /* This is a scoped object around task scheduling to measure the time
     * spent executing hpx threads */
    struct apex_wrapper
    {
        apex_wrapper(void* const data_ptr) : stopped(false), data_(nullptr)
        {
            /* APEX internal actions are not timed.  Otherwise, we would
             * end up with recursive timers. So it's possible to have
             * a null task wrapper pointer here. */
            if (data_ptr != nullptr) {
                data_ = (apex::task_wrapper*)data_ptr;
                apex::start(data_);
            }
        }
        ~apex_wrapper()
        {
            stop();
        }

        void stop() {
            if(!stopped) {
                stopped = true;
            /* APEX internal actions are not timed.  Otherwise, we would
             * end up with recursive timers. So it's possible to have
             * a null task wrapper pointer here. */
                if (data_ != nullptr) {
                    apex::stop(data_);
                }
            }
        }

        void yield() {
            if(!stopped) {
                stopped = true;
            /* APEX internal actions are not timed.  Otherwise, we would
             * end up with recursive timers. So it's possible to have
             * a null task wrapper pointer here. */
                if (data_ != nullptr) {
                    apex::yield(data_);
                }
            }
        }

        bool stopped;
        apex::task_wrapper * data_;
    };

    struct apex_wrapper_init
    {
        apex_wrapper_init(int argc, char **argv)
        {
            //apex::init(nullptr, hpx::get_locality_id(),
            //    hpx::get_initial_num_localities());
            hpx::register_pre_startup_function(&hpx_util_apex_init_startup);
        }
        ~apex_wrapper_init()
        {
            apex::finalize();
        }
    };
#else
    inline void apex_init() {}
    inline void apex_finalize() {}

    struct apex_wrapper
    {
        apex_wrapper(thread_description const& name) {}
        ~apex_wrapper() {}
    };

    struct apex_wrapper_init
    {
        apex_wrapper_init(int argc, char **argv) {}
        ~apex_wrapper_init() {}
    };
#endif
}}

