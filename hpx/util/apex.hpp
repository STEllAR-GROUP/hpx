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
#include <memory>
#include <cstdint>
#include <string>
typedef std::shared_ptr<apex::task_wrapper> apex_task_wrapper;
#else
typedef void* apex_task_wrapper;
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

    HPX_EXPORT apex_task_wrapper apex_new_task(
                thread_description const& description,
                std::uint32_t parent_task_locality,
                threads::thread_id_type const& parent_task);

    inline apex_task_wrapper apex_update_task(apex_task_wrapper wrapper,
                thread_description const& description)
    {
        if (wrapper == nullptr) {
            threads::thread_id_type parent_task(nullptr);
            // doesn't matter which locality we use, the parent is null
            return apex_new_task(description, 0, parent_task);
        } else if (description.kind() == thread_description::data_type_description) {
            return apex::update_task(wrapper,
                description.get_description());
        } else {
            return apex::update_task(wrapper,
                description.get_address());
        }
    }

    inline apex_task_wrapper apex_update_task(apex_task_wrapper wrapper, char const* name)
    {
        if (wrapper == nullptr) {
            apex_task_wrapper parent_task(nullptr);
            return apex::new_task(std::string(name), UINTMAX_MAX, parent_task);
        }
        return apex::update_task(wrapper, name);
    }

    /* This is a scoped object around task scheduling to measure the time
     * spent executing hpx threads */
    struct apex_wrapper
    {
        apex_wrapper(apex_task_wrapper data_ptr) : stopped(false), data_(nullptr)
        {
            /* APEX internal actions are not timed.  Otherwise, we would
             * end up with recursive timers. So it's possible to have
             * a null task wrapper pointer here. */
            if (data_ptr != nullptr) {
                data_ = data_ptr;
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
        apex_task_wrapper data_;
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

    inline apex_task_wrapper apex_new_task(
                thread_description const& description,
                std::uint32_t parent_task_locality,
                threads::thread_id_type const& parent_task) {return nullptr;}

    inline apex_task_wrapper apex_update_task(apex_task_wrapper wrapper,
                thread_description const& description) {return nullptr;}

    inline apex_task_wrapper apex_update_task(apex_task_wrapper wrapper,
                char const* name) {return nullptr;}

    struct apex_wrapper
    {
        apex_wrapper(apex_task_wrapper data_ptr) {}
        ~apex_wrapper() {}
        void stop(void) {}
        void yield(void) {}
    };

    struct apex_wrapper_init
    {
        apex_wrapper_init(int argc, char **argv) {}
        ~apex_wrapper_init() {}
    };
#endif
}}

