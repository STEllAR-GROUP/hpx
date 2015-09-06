//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_THREAD_EXECUTOR_JAN_11_2013_0700PM)
#define HPX_RUNTIME_THREADS_THREAD_EXECUTOR_JAN_11_2013_0700PM

#include <hpx/config.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/util/date_time_chrono.hpp>
#include <hpx/util/unique_function.hpp>
#include <hpx/util/safe_bool.hpp>

#include <boost/intrusive_ptr.hpp>
#include <boost/detail/atomic_count.hpp>
#include <boost/cstdint.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    namespace threads
    {
        class HPX_EXPORT executor;
    }

    HPX_EXPORT std::size_t get_os_thread_count(threads::executor const&);
}

namespace hpx { namespace threads
{
    /// \brief Data structure which stores statistics collected by an
    ///        executor instance.
    struct executor_statistics
    {
        executor_statistics()
          : tasks_scheduled_(0), tasks_completed_(0), queue_length_(0)
        {}

        boost::uint64_t tasks_scheduled_;
        boost::uint64_t tasks_completed_;
        boost::uint64_t queue_length_;
    };

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        enum executor_parameter
        {
            min_concurrency = 1,
            max_concurrency = 2,
            current_concurrency = 3
        };

        ///////////////////////////////////////////////////////////////////////
        // The interface below is used by the resource manager to
        // interact with the executor.
        struct manage_executor
        {
            virtual ~manage_executor() {}

            // Return the requested policy element
            virtual std::size_t get_policy_element(executor_parameter p,
                error_code& ec) const = 0;

            // Return statistics collected by this scheduler
            virtual void get_statistics(executor_statistics& stats,
                error_code& ec) const = 0;

            // Provide the given processing unit to the scheduler.
            virtual void add_processing_unit(std::size_t virt_core,
                std::size_t thread_num, error_code& ec) = 0;

            // Remove the given processing unit from the scheduler.
            virtual void remove_processing_unit(std::size_t thread_num,
                error_code& ec) = 0;
        };

        ///////////////////////////////////////////////////////////////////////
        // Main executor interface
        class executor_base;

        void intrusive_ptr_add_ref(executor_base* p);
        void intrusive_ptr_release(executor_base* p);

        class HPX_EXPORT executor_base
        {
        public:
            typedef util::unique_function_nonser<void()> closure_type;

            executor_base() : count_(0) {}
            virtual ~executor_base() {}

            // Scheduling methods.

            // Schedule the specified function for execution in this executor.
            // Depending on the subclass implementation, this may block in some
            // situations.
            virtual void add(closure_type && f, char const* desc,
                threads::thread_state_enum initial_state, bool run_now,
                threads::thread_stacksize stacksize, error_code& ec) = 0;

            // Return an estimate of the number of waiting closures.
            virtual boost::uint64_t num_pending_closures(error_code& ec) const = 0;

            // Reset internal (round robin) thread distribution scheme
            virtual void reset_thread_distribution() {}

            // Return the requested policy element
            virtual std::size_t get_policy_element(
                threads::detail::executor_parameter p, error_code& ec) const = 0;

            /// Return the mask for processing units the given thread is allowed
            /// to run on.
            virtual mask_cref_type get_pu_mask(topology const& topology,
                std::size_t num_thread) const;

            /// Set the new scheduler mode
            virtual void set_scheduler_mode(
                threads::policies::scheduler_mode mode) {}

        private:
            // reference counting
            friend void intrusive_ptr_add_ref(executor_base* p);
            friend void intrusive_ptr_release(executor_base* p);

            boost::detail::atomic_count count_;
        };

        /// support functions for boost::intrusive_ptr
        inline void intrusive_ptr_add_ref(executor_base* p)
        {
            ++p->count_;
        }
        inline void intrusive_ptr_release(executor_base* p)
        {
            if (0 == --p->count_)
                delete p;
        }

        ///////////////////////////////////////////////////////////////////////
        class scheduled_executor_base : public executor_base
        {
        public:
            // Schedule given function for execution in this executor no sooner
            // than time abs_time. This call never blocks, and may violate
            // bounds on the executor's queue size.
            virtual void add_at(
                boost::chrono::steady_clock::time_point const& abs_time,
                closure_type && f, char const* desc,
                threads::thread_stacksize stacksize, error_code& ec) = 0;

            void add_at(util::steady_time_point const& abs_time,
                closure_type && f, char const* desc,
                threads::thread_stacksize stacksize, error_code& ec)
            {
                return add_at(abs_time.value(), std::move(f), desc,
                    stacksize, ec);
            }

            // Schedule given function for execution in this executor no sooner
            // than time rel_time from now. This call never blocks, and may
            // violate bounds on the executor's queue size.
            virtual void add_after(
                boost::chrono::steady_clock::duration const& rel_time,
                closure_type && f, char const* desc,
                threads::thread_stacksize stacksize, error_code& ec) = 0;

            void add_after(util::steady_duration const& rel_time,
                closure_type && f, char const* desc,
                threads::thread_stacksize stacksize, error_code& ec)
            {
                return add_after(rel_time.value(), std::move(f), desc,
                    stacksize, ec);
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // This is equivalent to the proposed executor interface (see N3562)
    //
    //    class executor
    //    {
    //    public:
    //        virtual ~executor_base() {}
    //        virtual void add(function<void()> closure) = 0;
    //        virtual size_t num_pending_closures() const = 0;
    //    };
    //
    class executor
    {
        friend std::size_t hpx::get_os_thread_count(threads::executor const&);

    protected:
        // generic executors can't be created directly
        executor(detail::executor_base* data)
          : executor_data_(data)
        {}

    public:
        typedef detail::executor_base::closure_type closure_type;

        // default constructor creates invalid (non-usable) executor
        executor() {}

        /// Schedule the specified function for execution in this executor.
        /// Depending on the subclass implementation, this may block in some
        /// situations.
        void add(closure_type f, char const* desc = "",
            threads::thread_state_enum initial_state = threads::pending,
            bool run_now = true,
            threads::thread_stacksize stacksize = threads::thread_stacksize_default,
            error_code& ec = throws)
        {
            executor_data_->add(std::move(f), desc, initial_state, run_now,
                stacksize, ec);
        }

        /// Return an estimate of the number of waiting closures.
        boost::uint64_t num_pending_closures(error_code& ec = throws) const
        {
            return executor_data_->num_pending_closures(ec);
        }

        /// Return an estimate of the number of waiting closures.
        void reset_thread_distribution() const
        {
            executor_data_->reset_thread_distribution();
        }

        /// Return the mask for processing units the given thread is allowed
        /// to run on.
        mask_cref_type get_pu_mask(topology const& topology,
                std::size_t num_thread) const
        {
            return executor_data_->get_pu_mask(topology, num_thread);
        }

        /// Set the new scheduler mode
        void set_scheduler_mode(threads::policies::scheduler_mode mode)
        {
            return executor_data_->set_scheduler_mode(mode);
        }

        operator util::safe_bool<executor>::result_type() const
        {
            // avoid compiler warning about conversion to bool
            return util::safe_bool<executor>()(executor_data_.get() ? true : false);
        }

    protected:
        boost::intrusive_ptr<detail::executor_base> executor_data_;
    };

    ///////////////////////////////////////////////////////////////////////////
    // This is equivalent to the proposed scheduled_executor interface (see
    // N3562)
    //
    //      class scheduled_executor : public executor
    //      {
    //      public:
    //          virtual void add_at(chrono::steady_clock::time_point abs_time,
    //              function<void()> closure) = 0;
    //          virtual void add_after(chrono::steady_clock::duration rel_time,
    //              function<void()> closure) = 0;
    //      };
    //
    class HPX_EXPORT scheduled_executor : public executor
    {
    private:
        struct tag {};

    protected:
        // generic executors can't be created directly
        scheduled_executor(detail::scheduled_executor_base* data)
          : executor(data)
        {}

    public:
        // default constructor creates invalid (non-usable) executor
        scheduled_executor() {}

        /// Effects: The specified function object shall be scheduled for
        /// execution by the executor at some point in the future no sooner
        /// than the time represented by abs_time.
        /// Synchronization: completion of closure on a particular thread
        /// happens before destruction of that thread's thread-duration
        /// variables.
        /// Error conditions: If invoking closure throws an exception, the
        /// executor shall call terminate.
        void add_at(
            boost::chrono::steady_clock::time_point const& abs_time,
            closure_type f, char const* desc = "",
            threads::thread_stacksize stacksize = threads::thread_stacksize_default,
            error_code& ec = throws)
        {
            boost::static_pointer_cast<detail::scheduled_executor_base>(
                executor_data_)->add_at(abs_time, std::move(f), desc, stacksize, ec);
        }

        void add_at(util::steady_time_point const& abs_time,
            closure_type f, char const* desc = "",
            threads::thread_stacksize stacksize = threads::thread_stacksize_default,
            error_code& ec = throws)
        {
            return add_at(abs_time.value(), std::move(f), desc,
                stacksize, ec);
        }

        /// Effects: The specified function object shall be scheduled for
        /// execution by the executor at some point in the future no sooner
        /// than time rel_time from now.
        /// Synchronization: completion of closure on a particular thread
        /// happens before destruction of that thread's thread-duration
        /// variables.
        /// Error conditions: If invoking closure throws an exception, the
        /// executor shall call terminate.
        void add_after(
            boost::chrono::steady_clock::duration const& rel_time,
            closure_type f, char const* desc = "",
            threads::thread_stacksize stacksize = threads::thread_stacksize_default,
            error_code& ec = throws)
        {
            boost::static_pointer_cast<detail::scheduled_executor_base>(
                executor_data_)->add_after(rel_time, std::move(f), desc, stacksize, ec);
        }

        void add_after(util::steady_duration const& rel_time,
            closure_type f, char const* desc = "",
            threads::thread_stacksize stacksize = threads::thread_stacksize_default,
            error_code& ec = throws)
        {
            return add_after(rel_time.value(), std::move(f), desc,
                stacksize, ec);
        }

        /// Return a reference to the default executor for this process.
        static scheduled_executor& default_executor();
    };

    ///////////////////////////////////////////////////////////////////////////
    /// Returns: the default executor defined by the
    /// active process. If set_default_executor hasn't been called then the
    /// return value is a pointer to an executor of unspecified type.
    HPX_EXPORT scheduled_executor default_executor();

    /// Effect: the default executor of the active process is set to the given
    /// executor instance.
    /// Requires: executor shall not be null.
    /// Synchronization: Changing and using the default executor is sequentially
    /// consistent.
    HPX_EXPORT void set_default_executor(scheduled_executor executor);
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
