//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/resource_manager.hpp>
#include <hpx/runtime/threads/detail/scheduling_loop.hpp>
#include <hpx/runtime/threads/detail/create_thread.hpp>
#include <hpx/runtime/threads/detail/set_thread_state.hpp>
#include <hpx/runtime/threads/executors/thread_pool_executor.hpp>
#include <hpx/util/register_locks.hpp>

#include <boost/scope_exit.hpp>

namespace hpx { namespace threads { namespace executors { namespace detail
{
    void get_processing_units(std::size_t num_threads,
        std::vector<std::size_t>& punits)
    {
        for (std::size_t i = 0; i < num_threads; ++i)
            punits.push_back(i);
    }

    ///////////////////////////////////////////////////////////////////////////
    class manage_thread_pool_executor
      : public threads::detail::manage_executor
    {
    public:
        manage_thread_pool_executor(thread_pool_executor& sched)
          : sched_(sched)
        {
        }

    protected:
        // Return the requested policy element.
        std::size_t get_policy_element(threads::detail::executor_parameter p,
            error_code& ec) const
        {
            return sched_.get_policy_element(p, ec);
        }

        // Return statistics collected by this scheduler
        void get_statistics(executor_statistics& stats, error_code& ec) const
        {
            sched_.get_statistics(stats, ec);
        }

        // Provide the given processing unit to the scheduler.
        void add_processing_unit(std::size_t virt_core, std::size_t thread_num, 
            error_code& ec)
        {
            sched_.add_processing_unit(virt_core, thread_num, ec);
        }

        // Remove the given processing unit from the scheduler.
        void remove_processing_unit(std::size_t thread_num, error_code& ec)
        {
            sched_.remove_processing_unit(thread_num, ec);
        }

    private:
        thread_pool_executor& sched_;
    };

    ///////////////////////////////////////////////////////////////////////////
    thread_pool_executor::thread_pool_executor(std::size_t max_punits,
            std::size_t min_punits)
      : scheduler_(max_punits), shutdown_sem_(0),
        states_(max_punits), puinits_(max_punits),
        current_concurrency_(0), tasks_scheduled_(0), tasks_completed_(0),
        min_punits_(min_punits), max_punits_(max_punits), cookie_(0)
    {
        // Inform the resource manager about this new executor. This causes the
        // resource manager to interact with this executor using the
        // manage_executor interface.
        resource_manager& rm = resource_manager::get();
        cookie_ = rm.initial_allocation(new manage_thread_pool_executor(*this));
    }

    thread_pool_executor::~thread_pool_executor()
    {
        // Inform the resource manager that this executor is about to be 
        // destroyed. This will cause it to invoke remove_processing_unit below
        // for each of the currently allocated virtual cores.
        resource_manager& rm = resource_manager::get();
        rm.detach(cookie_);

        shutdown_sem_.wait();
    }

    threads::thread_state_enum thread_pool_executor::thread_function_nullary(
        HPX_STD_FUNCTION<void()> const& func)
    {
        // execute the actual thread function
        func();

        // update statistics
        ++tasks_completed_;

        // Verify that there are no more registered locks for this
        // OS-thread. This will throw if there are still any locks
        // held.
        util::force_error_on_lock();

        return threads::terminated;
    }

    // Schedule the specified function for execution in this executor.
    // Depending on the subclass implementation, this may block in some
    // situations.
    void thread_pool_executor::add(HPX_STD_FUNCTION<void()> f,
        char const* desc, threads::thread_state_enum initial_state,
        bool run_now, error_code& ec)
    {
        // create a new thread
        boost::intrusive_ptr<thread_pool_executor> this_(this);
        thread_init_data data(util::bind(
            &thread_pool_executor::thread_function_nullary, this_,
            boost::move(f)), desc);

        threads::detail::create_thread(scheduler_, data, initial_state, run_now, ec);
        if (ec) return;

        // update statistics
        ++tasks_scheduled_;

        if (&ec != &throws)
            ec = make_success_code();
    }

    // Like add(), except that if the attempt to add the function would
    // cause the caller to block in add, try_add would instead do
    // nothing and return false.
    bool thread_pool_executor::try_add(HPX_STD_FUNCTION<void()> f,
        char const* desc, threads::thread_state_enum initial_state,
        bool run_now, error_code& ec)
    {
        // create a new thread
        boost::intrusive_ptr<thread_pool_executor> this_(this);
        thread_init_data data(util::bind(
            &thread_pool_executor::thread_function_nullary, this_,
            boost::move(f)), desc);

        threads::detail::create_thread(scheduler_, data, initial_state, run_now, ec);
        if (ec) return false;

        // update statistics
        ++tasks_scheduled_;

        if (&ec != &throws)
            ec = make_success_code();

        return true;      // this function will never block
    }

    // Schedule given function for execution in this executor no sooner
    // than time abs_time. This call never blocks, and may violate
    // bounds on the executor's queue size.
    void thread_pool_executor::add_at(
        boost::posix_time::ptime const& abs_time,
        HPX_STD_FUNCTION<void()> f, char const* desc, error_code& ec)
    {
        // create a new suspended thread
        boost::intrusive_ptr<thread_pool_executor> this_(this);
        thread_init_data data(util::bind(
            &thread_pool_executor::thread_function_nullary, this_,
            boost::move(f)), desc);

        thread_id_type id = threads::detail::create_thread(
            scheduler_, data, suspended, true, ec);
        if (ec) return;
        BOOST_ASSERT(invalid_thread_id != id);    // would throw otherwise

        // now schedule new thread for execution
        threads::detail::set_thread_state_timed(scheduler_, abs_time, id, ec);
        if (ec) return;

        // update statistics
        ++tasks_scheduled_;

        if (&ec != &throws)
            ec = make_success_code();
    }

    // Schedule given function for execution in this executor no sooner
    // than time rel_time from now. This call never blocks, and may
    // violate bounds on the executor's queue size.
    void thread_pool_executor::add_after(
        boost::posix_time::time_duration const& rel_time,
        HPX_STD_FUNCTION<void()> f, char const* desc, error_code& ec)
    {
        // create a new suspended thread
        boost::intrusive_ptr<thread_pool_executor> this_(this);
        thread_init_data data(util::bind(
            &thread_pool_executor::thread_function_nullary, this_,
            boost::move(f)), desc);

        thread_id_type id = threads::detail::create_thread(
            scheduler_, data, suspended, true, ec);
        if (ec) return;
        BOOST_ASSERT(invalid_thread_id != id);    // would throw otherwise

        // now schedule new thread for execution
        threads::detail::set_thread_state_timed(scheduler_, rel_time, id, ec);
        if (ec) return;

        // update statistics
        ++tasks_scheduled_;

        if (&ec != &throws)
            ec = make_success_code();
    }

    // Return an estimate of the number of waiting tasks.
    std::size_t thread_pool_executor::num_pending_closures(error_code& ec) const
    {
        if (&ec != &throws)
            ec = make_success_code();
        return scheduler_.get_queue_length();
    }

    // execute all work
    void thread_pool_executor::run(std::size_t virt_core)
    {
        states_[virt_core] = running;
        ++current_concurrency_;

        BOOST_SCOPE_EXIT(&current_concurrency_) {
            --current_concurrency_;
        } BOOST_SCOPE_EXIT_END

        boost::int64_t executed_threads = 0;
        boost::uint64_t overall_times = 0, thread_times = 0;
        threads::detail::scheduling_loop(virt_core, scheduler_, 
            states_[virt_core], executed_threads, overall_times, thread_times);

        shutdown_sem_.signal();
    }

    // Return statistics collected by this scheduler
    void thread_pool_executor::get_statistics(executor_statistics& stats,
        error_code& ec) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        stats.queue_length_ = scheduler_.get_queue_length();
        stats.tasks_scheduled_ = tasks_scheduled_;
        stats.tasks_completed_ = tasks_completed_;
    }

    // Return the requested policy element
    std::size_t thread_pool_executor::get_policy_element(
        threads::detail::executor_parameter p, error_code& ec) const
    {
        switch(p) {
        case threads::detail::min_concurrency:
            return min_punits_;

        case threads::detail::max_concurrency:
            return max_punits_;

        case threads::detail::current_concurrency:
            return current_concurrency_;

        default:
            break;
        }

        HPX_THROWS_IF(ec, bad_parameter,
            "thread_pool_executor::get_policy_element",
            "requested value of invalid policy element");
        return std::size_t(-1);
    }

    // Provide the given processing unit to the scheduler.
    void thread_pool_executor::add_processing_unit(std::size_t virt_core,
        std::size_t thread_num, error_code& ec)
    {
        register_thread_nullary(
            util::bind(&thread_pool_executor::run, this, virt_core),
            "thread_pool_executor thread", threads::pending, true,
            threads::thread_priority_normal, thread_num,
            threads::thread_stacksize_default, ec);
    }

    // Remove the given processing unit from the scheduler.
    void thread_pool_executor::remove_processing_unit(std::size_t virt_core,
        error_code& ec)
    {
        // inform the scheduler to stop the virtual core
        states_[virt_core] = stopping;
    }
}}}}
