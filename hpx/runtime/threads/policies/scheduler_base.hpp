//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_SCHEDULING_SCHEDULER_BASE_JUL_14_2013_1132AM)
#define HPX_THREADMANAGER_SCHEDULING_SCHEDULER_BASE_JUL_14_2013_1132AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/runtime/threads/topology.hpp>

#include <boost/noncopyable.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
    ///////////////////////////////////////////////////////////////////////////
    /// The scheduler_base defines the interface to be implemented by all
    /// scheduler policies
    struct scheduler_base : boost::noncopyable
    {
        virtual ~scheduler_base() {}

        virtual bool numa_sensitive() const { return false; }
        virtual threads::mask_cref_type get_pu_mask(topology const& topology,
            std::size_t num_thread) const = 0;
        virtual std::size_t get_pu_num(std::size_t num_thread) const = 0;
        virtual std::size_t get_num_stolen_threads(std::size_t num_thread,
            bool reset) = 0;
        virtual boost::int64_t get_queue_length(
            std::size_t num_thread = std::size_t(-1)) const = 0;
        virtual boost::int64_t get_thread_count(thread_state_enum state = unknown,
            thread_priority priority = thread_priority_default,
            std::size_t num_thread = std::size_t(-1), bool reset = false) const = 0;
        virtual void abort_all_suspended_threads() = 0;

        virtual bool cleanup_terminated(bool delete_all = false) = 0;

        virtual thread_id_type create_thread(thread_init_data& data,
            thread_state_enum initial_state, bool run_now, error_code& ec,
            std::size_t num_thread) = 0;

        virtual bool get_next_thread(std::size_t num_thread, bool running,
            boost::int64_t& idle_loop_count, threads::thread_data*& thrd) = 0;

        virtual void schedule_thread(threads::thread_data* thrd,
            std::size_t num_thread,
            thread_priority priority = thread_priority_normal) = 0;
        virtual void schedule_thread_last(threads::thread_data* thrd,
            std::size_t num_thread,
            thread_priority priority = thread_priority_normal) = 0;

        virtual bool destroy_thread(threads::thread_data* thrd,
            boost::int64_t& busy_count) = 0;

        virtual bool wait_or_add_new(std::size_t num_thread, bool running,
            boost::int64_t& idle_loop_count) = 0;

        virtual void do_some_work(std::size_t num_thread = std::size_t(-1)) = 0;

        virtual void add_punit(std::size_t virt_core, std::size_t thread_num) {}

        virtual void on_start_thread(std::size_t num_thread) = 0;
        virtual void on_stop_thread(std::size_t num_thread) = 0;
        virtual void on_error(std::size_t num_thread, boost::exception_ptr const& e) = 0;

#if HPX_THREAD_MAINTAIN_QUEUE_WAITTIME
        virtual boost::int64_t get_average_thread_wait_time(
            std::size_t num_thread = std::size_t(-1)) const = 0;
        virtual boost::int64_t get_average_task_wait_time(
            std::size_t num_thread = std::size_t(-1)) const = 0;
#endif
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
