//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/thread_support/atomic_count.hpp>
#include <hpx/threading_base/scheduler_mode.hpp>
#include <hpx/threading_base/thread_description.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>
#include <hpx/topology/cpu_mask.hpp>
#include <hpx/topology/topology.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

#include <iosfwd>

// NOTE: Thread executors are deprecated and will be removed. Until then this
// forward declaration serves to make sure the thread_executors module does not
// depend on the runtime_local module (although it really does).
namespace hpx {
    namespace threads {
        class executor;
    }

    HPX_EXPORT std::size_t get_os_thread_count(threads::executor const& exec);
}    // namespace hpx

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        class HPX_EXPORT executor_base;
    }

    class executor_id
    {
    private:
        std::size_t id_;

        friend bool operator==(
            executor_id const& x, executor_id const& y) noexcept;
        friend bool operator!=(
            executor_id const& x, executor_id const& y) noexcept;
        friend bool operator<(
            executor_id const& x, executor_id const& y) noexcept;
        friend bool operator>(
            executor_id const& x, executor_id const& y) noexcept;
        friend bool operator<=(
            executor_id const& x, executor_id const& y) noexcept;
        friend bool operator>=(
            executor_id const& x, executor_id const& y) noexcept;

        template <typename Char, typename Traits>
        friend std::basic_ostream<Char, Traits>& operator<<(
            std::basic_ostream<Char, Traits>&, executor_id const&);

        friend class detail::executor_base;

    public:
        executor_id() noexcept
          : id_(0)
        {
        }
        explicit executor_id(std::size_t i) noexcept
          : id_(i)
        {
        }
    };

    inline bool operator==(executor_id const& x, executor_id const& y) noexcept
    {
        return x.id_ == y.id_;
    }

    inline bool operator!=(executor_id const& x, executor_id const& y) noexcept
    {
        return !(x == y);
    }

    inline bool operator<(executor_id const& x, executor_id const& y) noexcept
    {
        return x.id_ < y.id_;
    }

    inline bool operator>(executor_id const& x, executor_id const& y) noexcept
    {
        return x.id_ > y.id_;
    }

    inline bool operator<=(executor_id const& x, executor_id const& y) noexcept
    {
        return !(x.id_ > y.id_);
    }

    inline bool operator>=(executor_id const& x, executor_id const& y) noexcept
    {
        return !(x.id_ < y.id_);
    }

    template <typename Char, typename Traits>
    std::basic_ostream<Char, Traits>& operator<<(
        std::basic_ostream<Char, Traits>& out, executor_id const& id)
    {
        out << id.id_;    //-V128
        return out;
    }

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        // Main executor interface
        void intrusive_ptr_add_ref(executor_base* p);
        void intrusive_ptr_release(executor_base* p);

        class executor_base
        {
        public:
            typedef util::unique_function_nonser<void()> closure_type;

            executor_base()
              : count_(0)
            {
            }
            virtual ~executor_base() {}

            // Scheduling methods.

            // Schedule the specified function for execution in this executor.
            // Depending on the subclass implementation, this may block in some
            // situations.
            virtual void add(closure_type&& f,
                util::thread_description const& desc,
                threads::thread_schedule_state initial_state, bool run_now,
                threads::thread_stacksize stacksize,
                threads::thread_schedule_hint schedulehint, error_code& ec) = 0;

            // Return an estimate of the number of waiting closures.
            virtual std::uint64_t num_pending_closures(
                error_code& ec) const = 0;

            // Reset internal (round robin) thread distribution scheme
            virtual void reset_thread_distribution() {}

            // Return the requested policy element
            virtual std::size_t get_policy_element(
                threads::detail::executor_parameter p,
                error_code& ec) const = 0;

            // Return the mask for processing units the given thread is allowed
            // to run on.
            virtual mask_cref_type get_pu_mask(
                topology const& topology, std::size_t num_thread) const;

            // Set the new scheduler mode
            virtual void set_scheduler_mode(
                threads::policies::scheduler_mode /*mode*/)
            {
            }

            // retrieve executor id
            virtual executor_id get_id() const
            {
                return create_id(reinterpret_cast<std::size_t>(this));
            }

            virtual void detach()
            {
                // by default, do nothing
            }

        protected:
            static executor_id create_id(std::size_t id)
            {
                return executor_id(id);
            }

        private:
            // reference counting
            friend void intrusive_ptr_add_ref(executor_base* p);
            friend void intrusive_ptr_release(executor_base* p);

            util::atomic_count count_;
        };

        /// support functions for hpx::intrusive_ptr
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
            scheduled_executor_base()
              : stacksize_(thread_stacksize::default_)
              , priority_(thread_priority::default_)
              , schedulehint_(thread_schedule_hint())
            {
            }

            scheduled_executor_base(thread_priority priority,
                thread_stacksize stacksize, thread_schedule_hint schedulehint)
              : stacksize_(stacksize)
              , priority_(priority)
              , schedulehint_(schedulehint)
            {
            }

            // Schedule given function for execution in this executor no sooner
            // than time abs_time. This call never blocks, and may violate
            // bounds on the executor's queue size.
            virtual void add_at(
                std::chrono::steady_clock::time_point const& abs_time,
                closure_type&& f, util::thread_description const& desc,
                threads::thread_stacksize stacksize, error_code& ec) = 0;

            void add_at(hpx::chrono::steady_time_point const& abs_time,
                closure_type&& f, util::thread_description const& desc,
                threads::thread_stacksize stacksize, error_code& ec)
            {
                return add_at(
                    abs_time.value(), std::move(f), desc, stacksize, ec);
            }

            // Schedule given function for execution in this executor no sooner
            // than time rel_time from now. This call never blocks, and may
            // violate bounds on the executor's queue size.
            virtual void add_after(
                std::chrono::steady_clock::duration const& rel_time,
                closure_type&& f, util::thread_description const& desc,
                threads::thread_stacksize stacksize, error_code& ec) = 0;

            void add_after(hpx::chrono::steady_duration const& rel_time,
                closure_type&& f, util::thread_description const& desc,
                threads::thread_stacksize stacksize, error_code& ec)
            {
                return add_after(
                    rel_time.value(), std::move(f), desc, stacksize, ec);
            }

            thread_priority get_priority() const
            {
                return priority_;
            }
            thread_stacksize get_stacksize() const
            {
                return stacksize_;
            }
            thread_schedule_hint get_schedulehint() const
            {
                return schedulehint_;
            }

        protected:
            thread_stacksize stacksize_;
            thread_priority priority_;
            thread_schedule_hint schedulehint_;
        };
    }    // namespace detail

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
        {
        }

    public:
        typedef detail::executor_base::closure_type closure_type;
        typedef executor_id id;

        // default constructor creates invalid (non-usable) executor
        executor() {}

        /// Schedule the specified function for execution in this executor.
        /// Depending on the subclass implementation, this may block in some
        /// situations.
        void add(closure_type f,
            util::thread_description const& desc = util::thread_description(),
            threads::thread_schedule_state initial_state =
                threads::thread_schedule_state::pending,
            bool run_now = true,
            threads::thread_stacksize stacksize =
                threads::thread_stacksize::default_,
            threads::thread_schedule_hint schedulehint =
                threads::thread_schedule_hint(),
            error_code& ec = throws)
        {
            executor_data_->add(std::move(f), desc, initial_state, run_now,
                stacksize, schedulehint, ec);
        }

        /// Return an estimate of the number of waiting closures.
        std::uint64_t num_pending_closures(error_code& ec = throws) const
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
        mask_cref_type get_pu_mask(
            topology const& topology, std::size_t num_thread) const
        {
            return executor_data_->get_pu_mask(topology, num_thread);
        }

        /// Set the new scheduler mode
        void set_scheduler_mode(threads::policies::scheduler_mode mode)
        {
            return executor_data_->set_scheduler_mode(mode);
        }

        explicit operator bool() const noexcept
        {
            // avoid compiler warning about conversion to bool
            return executor_data_.get() ? true : false;
        }

        bool operator==(executor const& rhs) const
        {
            return get_id() == rhs.get_id();
        }

        id get_id() const
        {
            return executor_data_->get_id();
        }

    protected:
        hpx::intrusive_ptr<detail::executor_base> executor_data_;
    };

    ///////////////////////////////////////////////////////////////////////////
    // This is equivalent to the proposed scheduled_executor interface (see
    // N3562)
    //
    //      class scheduled_executor : public executor
    //      {
    //      public:
    //          virtual void add_at(chstd::chrono::steady_clock::time_point abs_time,
    //              function<void()> closure) = 0;
    //          virtual void add_after(chstd::chrono::steady_clock::duration rel_time,
    //              function<void()> closure) = 0;
    //      };
    //
    class HPX_EXPORT scheduled_executor : public executor
    {
    private:
        struct tag
        {
        };

    protected:
        // generic executors can't be created directly
        scheduled_executor(detail::scheduled_executor_base* data)
          : executor(data)
        {
        }

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
        void add_at(std::chrono::steady_clock::time_point const& abs_time,
            closure_type f, char const* desc = "",
            threads::thread_stacksize stacksize =
                threads::thread_stacksize::default_,
            error_code& ec = throws)
        {
            hpx::static_pointer_cast<detail::scheduled_executor_base>(
                executor_data_)
                ->add_at(abs_time, std::move(f), desc, stacksize, ec);
        }

        void add_at(hpx::chrono::steady_time_point const& abs_time,
            closure_type f, char const* desc = "",
            threads::thread_stacksize stacksize =
                threads::thread_stacksize::default_,
            error_code& ec = throws)
        {
            return add_at(abs_time.value(), std::move(f), desc, stacksize, ec);
        }

        /// Effects: The specified function object shall be scheduled for
        /// execution by the executor at some point in the future no sooner
        /// than time rel_time from now.
        /// Synchronization: completion of closure on a particular thread
        /// happens before destruction of that thread's thread-duration
        /// variables.
        /// Error conditions: If invoking closure throws an exception, the
        /// executor shall call terminate.
        void add_after(std::chrono::steady_clock::duration const& rel_time,
            closure_type f, char const* desc = "",
            threads::thread_stacksize stacksize =
                threads::thread_stacksize::default_,
            error_code& ec = throws)
        {
            hpx::static_pointer_cast<detail::scheduled_executor_base>(
                executor_data_)
                ->add_after(rel_time, std::move(f), desc, stacksize, ec);
        }

        void add_after(hpx::chrono::steady_duration const& rel_time,
            closure_type f, char const* desc = "",
            threads::thread_stacksize stacksize =
                threads::thread_stacksize::default_,
            error_code& ec = throws)
        {
            return add_after(
                rel_time.value(), std::move(f), desc, stacksize, ec);
        }

        void detach()
        {
            executor_data_->detach();
        }

        thread_priority get_priority() const
        {
            return static_cast<detail::scheduled_executor_base*>(
                executor_data_.get())
                ->get_priority();
        }

        thread_stacksize get_stacksize() const
        {
            return static_cast<detail::scheduled_executor_base*>(
                executor_data_.get())
                ->get_stacksize();
        }

        thread_schedule_hint get_schedulehint() const
        {
            return static_cast<detail::scheduled_executor_base*>(
                executor_data_.get())
                ->get_schedulehint();
        }
    };
}}    // namespace hpx::threads

#include <hpx/config/warnings_suffix.hpp>

#endif
