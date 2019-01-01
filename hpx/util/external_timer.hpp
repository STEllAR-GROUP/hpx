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

namespace hpx { namespace util { 

#ifdef HPX_HAVE_APEX

    typedef uint64_t init_t(const char * thread_name, const uint64_t comm_rank, const uint64_t comm_size);

    using enable_parent_task_handler_type = std::function<bool()>;

    HPX_EXPORT void set_enable_parent_task_handler(
        enable_parent_task_handler_type f);

    /* HPX provides a smart pointer to a data object that maintains
     * information about an hpx_thread.  Any library (i.e. APEX)
     * that wants to use this callback API needs to extend this class.
     */

    namespace external_timer {

    struct task_wrapper {
        bool dummy;
    };
    class profiler { };

    class timer_interface {
        public:
            timer_interface(void) {};
            ~timer_interface(void) {};
            static uint64_t init(const char * thread_name,
                const uint64_t comm_rank, const uint64_t comm_size) { return 0ULL; };
            static void finalize(void) {};
            static void register_thread(const std::string &name) {};
            static std::shared_ptr<task_wrapper> new_task(
                const std::string &name, const uint64_t task_id,
                const std::shared_ptr<task_wrapper> &parent_task) {return nullptr; };
            static std::shared_ptr<task_wrapper> new_task(
                uintptr_t address, const uint64_t task_id,
                const std::shared_ptr<task_wrapper> &parent_task) {return nullptr; };
            static void sample_value(const std::string &name, double value) {};
            static void send (uint64_t tag, uint64_t size, uint64_t target) {};
            static void recv (uint64_t tag, uint64_t size,
                uint64_t source_rank, uint64_t source_thread) {};
            static std::shared_ptr<task_wrapper> update_task(
                std::shared_ptr<task_wrapper> &wrapper, const std::string &name) {return nullptr; };
            static std::shared_ptr<task_wrapper> update_task(
                std::shared_ptr<task_wrapper> &wrapper, uintptr_t address) {return nullptr; };
            static profiler * start(std::shared_ptr<task_wrapper> &task_wrapper_ptr) {return nullptr; };
            static void stop(std::shared_ptr<task_wrapper> &task_wrapper_ptr) {};
            static void yield(std::shared_ptr<task_wrapper> &task_wrapper_ptr) {};
    };

    static timer_interface timer;

    void hpx_register_external_timer(timer_interface& new_timer) {
        timer = std::move(new_timer);
    }

    HPX_EXPORT std::shared_ptr<task_wrapper> new_task(
        thread_description const& description,
        std::uint32_t parent_locality_id,
        threads::thread_id_type const& parent_task);

    inline std::shared_ptr<task_wrapper> update_task(
        std::shared_ptr<task_wrapper> wrapper, thread_description const& description)
    {
        if (wrapper == nullptr)
        {
            threads::thread_id_type parent_task(nullptr);
            // doesn't matter which locality we use, the parent is null
            return new_task(description, 0, parent_task);
        }
        else if (description.kind() ==
            thread_description::data_type_description)
        {
            return timer.update_task(wrapper, description.get_description());
        }
        else
        {
            HPX_ASSERT(
                description.kind() == thread_description::data_type_address);
            return timer.update_task(wrapper, description.get_address());
        }
    }

    /* This is a scoped object around task scheduling to measure the time
     * spent executing hpx threads */
    struct scoped_timer
    {
        explicit scoped_timer(std::shared_ptr<task_wrapper> data_ptr)
          : stopped(false)
          , data_(nullptr)
        {
            /* APEX internal actions are not timed.  Otherwise, we would
             * end up with recursive timers. So it's possible to have
             * a null task wrapper pointer here. */
            if (data_ptr != nullptr)
            {
                data_ = data_ptr;
                timer.start(data_);
            }
        }
        ~scoped_timer()
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
                    timer.stop(data_);
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
                    timer.yield(data_);
                }
            }
        }

        bool stopped;
        std::shared_ptr<task_wrapper> data_;
    };

#else
    inline std::shared_ptr<task_wrapper> new_task(
        thread_description const& description,
        std::uint32_t parent_locality_id,
        threads::thread_id_type const& parent_task)
    {
        return nullptr;
    }

    inline std::shared_ptr<task_wrapper> update_task(
        std::shared_ptr<task_wrapper> wrapper, thread_description const& description)
    {
        return nullptr;
    }

    struct scoped_timer
    {
        scoped_timer(std::shared_ptr<task_wrapper> data_ptr) {}
        ~scoped_timer() {}
        void stop(void) {}
        void yield(void) {}
    };
#endif
    }    // namespace hpx::util::external_timer
}}    // namespace hpx::util
