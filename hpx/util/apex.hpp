//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once    // prevent multiple inclusions of this header file.

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/coroutines/thread_id_type.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/util/thread_description.hpp>

#ifdef HPX_HAVE_APEX
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include "apex_api.hpp"
typedef std::shared_ptr<apex::task_wrapper> apex_task_wrapper;
#else
typedef void* apex_task_wrapper;
#endif

namespace hpx { namespace util {

#ifdef HPX_HAVE_APEX

    using enable_parent_task_handler_type = std::function<bool()>;

    HPX_EXPORT void set_enable_parent_task_handler(
        enable_parent_task_handler_type f);

    HPX_EXPORT apex_task_wrapper apex_new_task(
        thread_description const& description,
        std::uint32_t parent_locality_id,
        threads::thread_id_type const& parent_task);

    inline apex_task_wrapper apex_update_task(
        apex_task_wrapper wrapper, thread_description const& description)
    {
        if (wrapper == nullptr)
        {
            threads::thread_id_type parent_task(nullptr);
            // doesn't matter which locality we use, the parent is null
            return apex_new_task(description, 0, parent_task);
        }
        else if (description.kind() ==
            thread_description::data_type_description)
        {
            return apex::update_task(wrapper, description.get_description());
        }
        else
        {
            HPX_ASSERT(
                description.kind() == thread_description::data_type_address);
            return apex::update_task(wrapper, description.get_address());
        }
    }

    /* This is a scoped object around task scheduling to measure the time
     * spent executing hpx threads */
    struct apex_wrapper
    {
        explicit apex_wrapper(apex_task_wrapper data_ptr)
          : stopped(false)
          , data_(nullptr)
        {
            /* APEX internal actions are not timed.  Otherwise, we would
             * end up with recursive timers. So it's possible to have
             * a null task wrapper pointer here. */
            if (data_ptr != nullptr)
            {
                data_ = data_ptr;
                apex::start(data_);
            }
        }
        ~apex_wrapper()
        {
            stop();
        }

        void stop()
        {
            if (!stopped)
            {
                stopped = true;
                /* APEX internal actions are not timed.  Otherwise, we would
             * end up with recursive timers. So it's possible to have
             * a null task wrapper pointer here. */
                if (data_ != nullptr)
                {
                    apex::stop(data_);
                }
            }
        }

        void yield()
        {
            if (!stopped)
            {
                stopped = true;
                /* APEX internal actions are not timed.  Otherwise, we would
             * end up with recursive timers. So it's possible to have
             * a null task wrapper pointer here. */
                if (data_ != nullptr)
                {
                    apex::yield(data_);
                }
            }
        }

        bool stopped;
        apex_task_wrapper data_;
    };

#else
    inline apex_task_wrapper apex_new_task(
        thread_description const& description,
        std::uint32_t parent_locality_id,
        threads::thread_id_type const& parent_task)
    {
        return nullptr;
    }

    inline apex_task_wrapper apex_update_task(
        apex_task_wrapper wrapper, thread_description const& description)
    {
        return nullptr;
    }

    struct apex_wrapper
    {
        apex_wrapper(apex_task_wrapper data_ptr) {}
        ~apex_wrapper() {}
        void stop(void) {}
        void yield(void) {}
    };

#endif
}}    // namespace hpx::util
