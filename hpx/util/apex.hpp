//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once // prevent multiple inclusions of this header file.

#include <hpx/config.hpp>
#include <hpx/coroutines/thread_id_type.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>

#ifdef HPX_HAVE_APEX
#include "apex_api.hpp"
#include <memory>
#include <cstddef>
#include <cstdint>
#include <string>
typedef std::shared_ptr<apex::task_wrapper> apex_task_wrapper;
#else
typedef void* apex_task_wrapper;
#endif

namespace hpx { namespace util {

#ifdef HPX_HAVE_APEX

    using enable_parent_task_handler_type = std::function<bool()>;

    HPX_EXPORT void set_enable_parent_task_handler(enable_parent_task_handler_type f);

    HPX_EXPORT apex_task_wrapper apex_new_task(
                std::size_t address,
                threads::thread_id_type const& parent_task);

    HPX_EXPORT apex_task_wrapper apex_new_task(
                char const *description,
                threads::thread_id_type const& parent_task);

    inline apex_task_wrapper apex_update_task(apex_task_wrapper wrapper,
                std::size_t address)
    {
        if (wrapper == nullptr) {
            threads::thread_id_type parent_task(nullptr);
            // doesn't matter which locality we use, the parent is null
            return apex_new_task(address, parent_task);
        }
        return apex::update_task(wrapper, address);
    }

    inline apex_task_wrapper apex_update_task(apex_task_wrapper wrapper,
                char const *description)
    {
        if (wrapper == nullptr) {
            threads::thread_id_type parent_task(nullptr);
            // doesn't matter which locality we use, the parent is null
            return apex_new_task(description, parent_task);
        }
        return apex::update_task(wrapper, description);
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

#else
    inline apex_task_wrapper apex_new_task(
                std::size_t address,
                threads::thread_id_type const& parent_task) {return nullptr;}

    inline apex_task_wrapper apex_new_task(
                char const *description,
                threads::thread_id_type const& parent_task) {return nullptr;}

    inline apex_task_wrapper apex_update_task(apex_task_wrapper wrapper,
                std::size_t address) {return nullptr;}

    inline apex_task_wrapper apex_update_task(apex_task_wrapper wrapper,
                char const* description) {return nullptr;}

    struct apex_wrapper
    {
        apex_wrapper(apex_task_wrapper data_ptr) {}
        ~apex_wrapper() {}
        void stop(void) {}
        void yield(void) {}
    };

#endif
}}

