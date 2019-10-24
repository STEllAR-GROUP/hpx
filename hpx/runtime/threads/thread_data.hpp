//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_THREAD_DATA_HPP
#define HPX_RUNTIME_THREADS_THREAD_DATA_HPP

#include <hpx/config.hpp>
#include <hpx/coroutines/coroutine.hpp>
#include <hpx/coroutines/detail/combined_tagged_state.hpp>
#include <hpx/errors.hpp>
#include <hpx/runtime/get_locality_id.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>

#include <hpx/assertion.hpp>
#include <hpx/concurrency/spinlock_pool.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/thread_support/atomic_count.hpp>
#include <hpx/util/backtrace.hpp>
#include <hpx/util/thread_description.hpp>
#if defined(HPX_HAVE_APEX)
#include <hpx/util/apex.hpp>
#endif

#include <boost/intrusive_ptr.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <forward_list>
#include <stack>
#include <string>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads {

    ////////////////////////////////////////////////////////////////////////////
    /// A \a thread is the representation of a ParalleX thread. It's a first
    /// class object in ParalleX. In our implementation this is a user level
    /// thread running on top of one of the OS threads spawned by the \a
    /// thread-manager.
    ///
    /// A \a thread encapsulates:
    ///  - A thread status word (see the functions \a thread#get_state and
    ///    \a thread#set_state)
    ///  - A function to execute (the thread function)
    ///  - A frame (in this implementation this is a block of memory used as
    ///    the threads stack)
    ///  - A block of registers (not implemented yet)
    ///
    /// Generally, \a threads are not created or executed directly. All
    /// functionality related to the management of \a threads is
    /// implemented by the thread-manager.
    class HPX_EXPORT thread_data
    {
    public:
        thread_data(thread_data const&) = delete;
        thread_data(thread_data&&) = delete;
        thread_data& operator=(thread_data const&) = delete;
        thread_data& operator=(thread_data&&) = delete;

    public:
        struct tag
        {
        };
        using mutex_type = util::spinlock_pool<tag>;

        /// The get_state function queries the state of this thread instance.
        ///
        /// \returns        This function returns the current state of this
        ///                 thread. It will return one of the values as defined
        ///                 by the \a thread_state enumeration.
        ///
        /// \note           This function will be seldom used directly. Most of
        ///                 the time the state of a thread will be retrieved
        ///                 by using the function \a threadmanager#get_state.
        thread_state get_state(
            std::memory_order order = std::memory_order_acquire) const
        {
            return current_state_.load(order);
        }

        /// The set_state function changes the state of this thread instance.
        ///
        /// \param newstate [in] The new state to be set for the thread.
        ///
        /// \note           This function will be seldomly used directly. Most of
        ///                 the time the state of a thread will have to be
        ///                 changed using the threadmanager. Moreover,
        ///                 changing the thread state using this function does
        ///                 not change its scheduling status. It only sets the
        ///                 thread's status word. To change the thread's
        ///                 scheduling status \a threadmanager#set_state should
        ///                 be used.
        thread_state set_state(thread_state_enum state,
            thread_state_ex_enum state_ex = wait_unknown,
            std::memory_order load_order = std::memory_order_acquire,
            std::memory_order exchange_order = std::memory_order_seq_cst)
        {
            thread_state prev_state = current_state_.load(load_order);

            for (;;)
            {
                thread_state tmp = prev_state;

                // ABA prevention for state only (not for state_ex)
                std::int64_t tag = tmp.tag();
                if (state != tmp.state())
                    ++tag;

                if (state_ex == wait_unknown)
                    state_ex = tmp.state_ex();

                if (HPX_LIKELY(current_state_.compare_exchange_strong(tmp,
                        thread_state(state, state_ex, tag), exchange_order)))
                {
                    return prev_state;
                }

                prev_state = tmp;
            }
        }

        bool set_state_tagged(thread_state_enum newstate,
            thread_state& prev_state, thread_state& new_tagged_state,
            std::memory_order exchange_order = std::memory_order_seq_cst)
        {
            thread_state tmp = prev_state;
            thread_state_ex_enum state_ex = tmp.state_ex();

            new_tagged_state =
                thread_state(newstate, state_ex, prev_state.tag() + 1);

            if (!current_state_.compare_exchange_strong(
                    tmp, new_tagged_state, exchange_order))
            {
                return false;
            }

            prev_state = tmp;
            return true;
        }

        /// The restore_state function changes the state of this thread
        /// instance depending on its current state. It will change the state
        /// atomically only if the current state is still the same as passed
        /// as the second parameter. Otherwise it won't touch the thread state
        /// of this instance.
        ///
        /// \param newstate [in] The new state to be set for the thread.
        /// \param oldstate [in] The old state of the thread which still has to
        ///                 be the current state.
        ///
        /// \note           This function will be seldomly used directly. Most of
        ///                 the time the state of a thread will have to be
        ///                 changed using the threadmanager. Moreover,
        ///                 changing the thread state using this function does
        ///                 not change its scheduling status. It only sets the
        ///                 thread's status word. To change the thread's
        ///                 scheduling status \a threadmanager#set_state should
        ///                 be used.
        ///
        /// \returns This function returns \a true if the state has been
        ///          changed successfully
        bool restore_state(thread_state new_state, thread_state old_state,
            std::memory_order load_order = std::memory_order_relaxed,
            std::memory_order load_exchange = std::memory_order_seq_cst)
        {
            // ABA prevention for state only (not for state_ex)
            std::int64_t tag = old_state.tag();
            if (new_state.state() != old_state.state())
                ++tag;

            // ignore the state_ex while compare-exchanging
            thread_state_ex_enum state_ex =
                current_state_.load(load_order).state_ex();

            thread_state old_tmp(old_state.state(), state_ex, old_state.tag());
            thread_state new_tmp(new_state.state(), state_ex, tag);

            return current_state_.compare_exchange_strong(
                old_tmp, new_tmp, load_exchange);
        }

        bool restore_state(thread_state_enum new_state,
            thread_state_ex_enum state_ex, thread_state old_state,
            std::memory_order load_exchange = std::memory_order_seq_cst)
        {
            // ABA prevention for state only (not for state_ex)
            std::int64_t tag = old_state.tag();
            if (new_state != old_state.state())
                ++tag;

            return current_state_.compare_exchange_strong(old_state,
                thread_state(new_state, state_ex, tag), load_exchange);
        }

    protected:
        /// The set_state function changes the extended state of this
        /// thread instance.
        ///
        /// \param newstate [in] The new extended state to be set for the
        ///                 thread.
        ///
        /// \note           This function will be seldom used directly. Most of
        ///                 the time the state of a thread will have to be
        ///                 changed using the threadmanager.
        thread_state_ex_enum set_state_ex(thread_state_ex_enum new_state)
        {
            thread_state prev_state =
                current_state_.load(std::memory_order_acquire);

            for (;;)
            {
                thread_state tmp = prev_state;

                if (HPX_LIKELY(current_state_.compare_exchange_strong(
                        tmp, thread_state(tmp.state(), new_state, tmp.tag()))))
                {
                    return prev_state.state_ex();
                }

                prev_state = tmp;
            }
        }

    public:
        /// Return the id of the component this thread is running in
        naming::address_type get_component_id() const
        {
            return 0;
        }

#ifndef HPX_HAVE_THREAD_DESCRIPTION
        util::thread_description get_description() const
        {
            return util::thread_description("<unknown>");
        }
        util::thread_description set_description(
            util::thread_description /*value*/)
        {
            return util::thread_description("<unknown>");
        }

        util::thread_description get_lco_description() const
        {
            return util::thread_description("<unknown>");
        }
        util::thread_description set_lco_description(
            util::thread_description /*value*/)
        {
            return util::thread_description("<unknown>");
        }
#else
        util::thread_description get_description() const
        {
            mutex_type::scoped_lock l(this);
            return description_;
        }
        util::thread_description set_description(util::thread_description value)
        {
            mutex_type::scoped_lock l(this);
            std::swap(description_, value);
            return value;
        }

        util::thread_description get_lco_description() const
        {
            mutex_type::scoped_lock l(this);
            return lco_description_;
        }
        util::thread_description set_lco_description(
            util::thread_description value)
        {
            mutex_type::scoped_lock l(this);
            std::swap(lco_description_, value);
            return value;
        }
#endif

#ifndef HPX_HAVE_THREAD_PARENT_REFERENCE
        /// Return the locality of the parent thread
        std::uint32_t get_parent_locality_id() const
        {
            return naming::invalid_locality_id;
        }

        /// Return the thread id of the parent thread
        thread_id_type get_parent_thread_id() const
        {
            return threads::invalid_thread_id;
        }

        /// Return the phase of the parent thread
        std::size_t get_parent_thread_phase() const
        {
            return 0;
        }
#else
        /// Return the locality of the parent thread
        std::uint32_t get_parent_locality_id() const
        {
            return parent_locality_id_;
        }

        /// Return the thread id of the parent thread
        thread_id_type get_parent_thread_id() const
        {
            return parent_thread_id_;
        }

        /// Return the phase of the parent thread
        std::size_t get_parent_thread_phase() const
        {
            return parent_thread_phase_;
        }
#endif

#ifdef HPX_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
        void set_marked_state(thread_state_enum mark) const
        {
            marked_state_ = mark;
        }
        thread_state_enum get_marked_state() const
        {
            return marked_state_;
        }
#endif

#ifndef HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION

#ifdef HPX_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
        char const* get_backtrace() const
        {
            return nullptr;
        }
        char const* set_backtrace(char const*)
        {
            return nullptr;
        }
#else
        util::backtrace const* get_backtrace() const
        {
            return nullptr;
        }
        util::backtrace const* set_backtrace(util::backtrace const*)
        {
            return nullptr;
        }
#endif

#else    // defined(HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION

#ifdef HPX_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
        char const* get_backtrace() const
        {
            mutex_type::scoped_lock l(this);
            return backtrace_;
        }
        char const* set_backtrace(char const* value)
        {
            mutex_type::scoped_lock l(this);

            char const* bt = backtrace_;
            backtrace_ = value;
            return bt;
        }
#else
        util::backtrace const* get_backtrace() const
        {
            mutex_type::scoped_lock l(this);
            return backtrace_;
        }
        util::backtrace const* set_backtrace(util::backtrace const* value)
        {
            mutex_type::scoped_lock l(this);

            util::backtrace const* bt = backtrace_;
            backtrace_ = value;
            return bt;
        }
#endif

        // Generate full backtrace for captured stack
        std::string backtrace()
        {
            mutex_type::scoped_lock l(this);
            std::string bt;
            if (0 != backtrace_)
            {
#ifdef HPX_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
                bt = *backtrace_;
#else
                bt = backtrace_->trace();
#endif
            }
            return bt;
        }
#endif

        thread_priority get_priority() const
        {
            return priority_;
        }
        void set_priority(thread_priority priority)
        {
            priority_ = priority;
        }

        // handle thread interruption
        bool interruption_requested() const
        {
            mutex_type::scoped_lock l(this);
            return requested_interrupt_;
        }

        bool interruption_enabled() const
        {
            mutex_type::scoped_lock l(this);
            return enabled_interrupt_;
        }

        bool set_interruption_enabled(bool enable)
        {
            mutex_type::scoped_lock l(this);
            std::swap(enabled_interrupt_, enable);
            return enable;
        }

        void interrupt(bool flag = true)
        {
            mutex_type::scoped_lock l(this);
            if (flag && !enabled_interrupt_)
            {
                l.unlock();
                HPX_THROW_EXCEPTION(thread_not_interruptable,
                    "thread_data::interrupt",
                    "interrupts are disabled for this thread");
                return;
            }
            requested_interrupt_ = flag;
        }

        bool interruption_point(bool throw_on_interrupt = true);

        bool add_thread_exit_callback(util::function_nonser<void()> const& f);
        void run_thread_exit_callbacks();
        void free_thread_exit_callbacks();

        policies::scheduler_base* get_scheduler_base() const
        {
            return scheduler_base_;
        }

        std::ptrdiff_t get_stack_size() const
        {
            return stacksize_;
        }

        template <typename ThreadQueue>
        ThreadQueue& get_queue()
        {
            return *static_cast<ThreadQueue*>(queue_);
        }

        /// \brief Execute the thread function
        ///
        /// \returns        This function returns the thread state the thread
        ///                 should be scheduled from this point on. The thread
        ///                 manager will use the returned value to set the
        ///                 thread's scheduling status.
        inline coroutine_type::result_type operator()();

        virtual thread_id_type get_thread_id() const
        {
            return thread_id_type{const_cast<thread_data*>(this)};
        }

#if !defined(HPX_HAVE_THREAD_PHASE_INFORMATION)
        virtual std::size_t get_thread_phase() const
        {
            return 0;
        }
#else
        virtual std::size_t get_thread_phase() const = 0;
#endif
        virtual std::size_t get_thread_data() const = 0;
        virtual std::size_t set_thread_data(std::size_t data) = 0;

        virtual void rebind(
            thread_init_data& init_data, thread_state_enum newstate) = 0;

#if defined(HPX_HAVE_APEX)
        apex_task_wrapper get_apex_data() const
        {
            return apex_data_;
        }
        void set_apex_data(apex_task_wrapper data)
        {
            apex_data_ = data;
        }
#endif

        // Construct a new \a thread
        thread_data(thread_init_data& init_data, void* queue,
            thread_state_enum newstate, bool  is_stackless = false);

        virtual ~thread_data();
        virtual void destroy() = 0;

    protected:
        void rebind_base(
            thread_init_data& init_data, thread_state_enum newstate);

    private:
        mutable std::atomic<thread_state> current_state_;

        ///////////////////////////////////////////////////////////////////////
        // Debugging/logging information
#ifdef HPX_HAVE_THREAD_DESCRIPTION
        util::thread_description description_;
        util::thread_description lco_description_;
#endif

#ifdef HPX_HAVE_THREAD_PARENT_REFERENCE
        std::uint32_t parent_locality_id_;
        thread_id_type parent_thread_id_;
        std::size_t parent_thread_phase_;
#endif

#ifdef HPX_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
        mutable thread_state_enum marked_state_;
#endif

#ifdef HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
#ifdef HPX_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
        char const* backtrace_;
#else
        util::backtrace const* backtrace_;
#endif
#endif
        ///////////////////////////////////////////////////////////////////////
        thread_priority priority_;

        bool requested_interrupt_;
        bool enabled_interrupt_;
        bool ran_exit_funcs_;

        // Singly linked list (heap-allocated)
        std::forward_list<util::function_nonser<void()>> exit_funcs_;

        // reference to scheduler which created/manages this thread
        policies::scheduler_base* scheduler_base_;

        std::ptrdiff_t stacksize_;

        void* queue_;
#if defined(HPX_HAVE_APEX)
        apex_task_wrapper apex_data_;
#endif
        bool is_stackless_;
    };

    HPX_CONSTEXPR thread_data* get_thread_id_data(thread_id_type const& tid)
    {
        return static_cast<thread_data*>(tid.get());
    }
}}    // namespace hpx::threads

#include <hpx/config/warnings_suffix.hpp>

#include <hpx/runtime/threads/thread_data_stackful.hpp>
#include <hpx/runtime/threads/thread_data_stackless.hpp>

namespace hpx { namespace threads {

    inline coroutine_type::result_type thread_data::operator()()
    {
        if (is_stackless_)
        {
            return static_cast<thread_data_stackless*>(this)->call();
        }
        return static_cast<thread_data_stackful*>(this)->call();
    }
}}
#endif /*HPX_RUNTIME_THREADS_THREAD_DATA_HPP*/
