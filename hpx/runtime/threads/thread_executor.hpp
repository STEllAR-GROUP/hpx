//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_THREAD_EXECUTOR_JAN_11_2013_0700PM)
#define HPX_RUNTIME_THREADS_THREAD_EXECUTOR_JAN_11_2013_0700PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/date_time_chrono.hpp>
#include <hpx/util/function.hpp>

#include <boost/intrusive_ptr.hpp>
#include <boost/detail/atomic_count.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads
{
    namespace detail
    {
        class executor_base;

        void intrusive_ptr_add_ref(executor_base* p);
        void intrusive_ptr_release(executor_base* p);

        class executor_base
        {
        public:
            executor_base() : count_(0) {}
            virtual ~executor_base() {}

            /// Scheduling methods.
            /// Schedule the specified function for execution in this executor.
            /// Depending on the subclass implementation, this may block in some
            /// situations.
            virtual void add(HPX_STD_FUNCTION<void()> f, char const* desc) = 0;

            /// Like add(), except that if the attempt to add the function would
            /// cause the caller to block in add, try_add would instead do
            /// nothing and return false.
            virtual bool try_add(HPX_STD_FUNCTION<void()> f, char const* desc) = 0;

            /// Schedule given function for execution in this executor no sooner
            /// than time abs_time. This call never blocks, and may violate
            /// bounds on the executor's queue size.
            virtual void add_at(boost::posix_time::ptime const& abs_time,
                HPX_STD_FUNCTION<void()> f, char const* desc) = 0;

            /// Schedule given function for execution in this executor no sooner
            /// than time rel_time from now. This call never blocks, and may
            /// violate bounds on the executor's queue size.
            virtual void add_after(
                boost::posix_time::time_duration const& rel_time,
                HPX_STD_FUNCTION<void()> f, char const* desc) = 0;

            /// Return an estimate of the number of waiting closures.
            virtual std::size_t num_pending_tasks() const = 0;

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
    }

    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT executor
    {
        struct tag {};

    protected:
        // generic executors can't be created directly
        executor(detail::executor_base* data)
          : executor_data_(data)
        {
        }

    public:
        ~executor()
        {
        }

        /// Schedule the specified function for execution in this executor.
        /// Depending on the subclass implementation, this may block in some
        /// situations.
        void add(HPX_STD_FUNCTION<void()> f, char const* desc = 0)
        {
            executor_data_->add(f, desc);
        }

        /// Like add(), except that if the attempt to add the function would
        /// cause the caller to block in add, try_add would instead do
        /// nothing and return false.
        bool try_add(HPX_STD_FUNCTION<void()> f, char const* desc = 0)
        {
            return executor_data_->try_add(f, desc);
        }

        /// Schedule given function for execution in this executor no sooner
        /// than time abs_time. This call never blocks, and may violate
        /// bounds on the executor's queue size.
        void add_at(boost::posix_time::ptime const& abs_time,
            HPX_STD_FUNCTION<void()> f, char const* desc = 0)
        {
            executor_data_->add_at(abs_time, f, desc);
        }

        template <typename Clock, typename Duration>
        void add_at(boost::chrono::time_point<Clock, Duration> const& abs_time,
            HPX_STD_FUNCTION<void()> f, char const* desc = 0)
        {
            add_at(util::to_ptime(abs_time), f, desc);
        }

        /// Schedule given function for execution in this executor no sooner
        /// than time rel_time from now. This call never blocks, and may
        /// violate bounds on the executor's queue size.
        void add_after(
            boost::posix_time::time_duration const& rel_time,
            HPX_STD_FUNCTION<void()> f, char const* desc = 0)
        {
            executor_data_->add_after(rel_time, f, desc);
        }

        template <typename Rep, typename Period>
        void add_after(boost::chrono::duration<Rep, Period> const& rel_time,
            HPX_STD_FUNCTION<void()> f, char const* desc = 0)
        {
            add_at(util::to_time_duration(rel_time), f, desc);
        }

        /// Return an estimate of the number of waiting closures.
        std::size_t num_pending_tasks() const
        {
            return executor_data_->num_pending_tasks();
        }

        /// Return a reference to the default executor for this process.
        static executor& default_executor();

    private:
        boost::intrusive_ptr<detail::executor_base> executor_data_;
    };
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
