//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_EXECUTORS_MANAGE_THREAD_EXECUTOR_JUL_16_2015_0745PM)
#define HPX_RUNTIME_THREADS_EXECUTORS_MANAGE_THREAD_EXECUTOR_JUL_16_2015_0745PM

#include <hpx/config.hpp>
#include <hpx/error.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>

#include <cstddef>

namespace hpx { namespace threads { namespace executors { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ExecutorImpl>
    class manage_thread_executor
      : public threads::detail::manage_executor
    {
    public:
        manage_thread_executor(ExecutorImpl& sched)
          : sched_(sched)
        {}

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

        // return the description string of the underlying scheduler
        char const* get_description() const
        {
            return sched_.get_description();
        }

    private:
        ExecutorImpl& sched_;
    };
}}}}

#endif
