//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/coroutines/detail/coroutine_accessor.hpp>
#include <hpx/execution_base/register_locks.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/thread_support/unlock_guard.hpp>
#include <hpx/threading_base/thread_data.hpp>
#if defined(HPX_HAVE_APEX)
#include <hpx/threading_base/external_timer.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <memory>

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads {
    namespace detail {
        static get_locality_id_type* get_locality_id_f;

        void set_get_locality_id(get_locality_id_type* f)
        {
            get_locality_id_f = f;
        }

        std::uint32_t get_locality_id(hpx::error_code& ec)
        {
            if (get_locality_id_f)
            {
                return get_locality_id_f(ec);
            }

            // same as naming::invalid_locality_id
            return ~static_cast<std::uint32_t>(0);
        }
    }    // namespace detail

    thread_data::thread_data(thread_init_data& init_data, void* queue,
        std::ptrdiff_t stacksize, bool is_stackless)
      : current_state_(thread_state(
            init_data.initial_state, thread_restart_state::signaled))
#ifdef HPX_HAVE_THREAD_DESCRIPTION
      , description_(init_data.description)
      , lco_description_()
#endif
#ifdef HPX_HAVE_THREAD_PARENT_REFERENCE
      , parent_locality_id_(init_data.parent_locality_id)
      , parent_thread_id_(init_data.parent_id)
      , parent_thread_phase_(init_data.parent_phase)
#endif
#ifdef HPX_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
      , marked_state_(thread_schedule_state::unknown)
#endif
#ifdef HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
      , backtrace_(nullptr)
#endif
      , priority_(init_data.priority)
      , requested_interrupt_(false)
      , enabled_interrupt_(true)
      , ran_exit_funcs_(false)
      , scheduler_base_(init_data.scheduler_base)
      , last_worker_thread_num_(std::size_t(-1))
      , stacksize_(stacksize)
      , stacksize_enum_(init_data.stacksize)
      , queue_(queue)
      , is_stackless_(is_stackless)
    {
        LTM_(debug) << "thread::thread(" << this << "), description("
                    << get_description() << ")";

        HPX_ASSERT(stacksize_enum_ != threads::thread_stacksize::current);

#ifdef HPX_HAVE_THREAD_PARENT_REFERENCE
        // store the thread id of the parent thread, mainly for debugging
        // purposes
        if (parent_thread_id_)
        {
            thread_self* self = get_self_ptr();
            if (self)
            {
                parent_thread_id_ = threads::get_self_id();
                parent_thread_phase_ = self->get_thread_phase();
            }
        }
        if (0 == parent_locality_id_)
            parent_locality_id_ = detail::get_locality_id(hpx::throws);
#endif
#if defined(HPX_HAVE_APEX)
        set_timer_data(init_data.timer_data);
#endif
    }

    thread_data::~thread_data()
    {
        free_thread_exit_callbacks();
    }

    void thread_data::run_thread_exit_callbacks()
    {
        std::unique_lock<hpx::util::detail::spinlock> l(
            spinlock_pool::spinlock_for(this));

        while (!exit_funcs_.empty())
        {
            {
                hpx::util::unlock_guard<
                    std::unique_lock<hpx::util::detail::spinlock>>
                    ul(l);
                if (!exit_funcs_.front().empty())
                    exit_funcs_.front()();
            }
            exit_funcs_.pop_front();
        }
        ran_exit_funcs_ = true;
    }

    bool thread_data::add_thread_exit_callback(
        util::function_nonser<void()> const& f)
    {
        std::lock_guard<hpx::util::detail::spinlock> l(
            spinlock_pool::spinlock_for(this));

        if (ran_exit_funcs_ ||
            get_state().state() == thread_schedule_state::terminated)
        {
            return false;
        }

        exit_funcs_.push_front(f);

        return true;
    }

    void thread_data::free_thread_exit_callbacks()
    {
        std::lock_guard<hpx::util::detail::spinlock> l(
            spinlock_pool::spinlock_for(this));

        // Exit functions should have been executed.
        HPX_ASSERT(exit_funcs_.empty() || ran_exit_funcs_);

        exit_funcs_.clear();
    }

    bool thread_data::interruption_point(bool throw_on_interrupt)
    {
        // We do not protect enabled_interrupt_ and requested_interrupt_
        // from concurrent access here (which creates a benign data race) in
        // order to avoid infinite recursion. This function is called by
        // this_thread::suspend which causes problems if the lock would call
        // suspend itself.
        if (enabled_interrupt_ && requested_interrupt_)
        {
            // Verify that there are no more registered locks for this
            // OS-thread. This will throw if there are still any locks
            // held.
            util::force_error_on_lock();

            // now interrupt this thread
            if (throw_on_interrupt)
                throw hpx::thread_interrupted();

            return true;
        }
        return false;
    }

    void thread_data::rebind_base(thread_init_data& init_data)
    {
        LTM_(debug) << "~thread(" << this << "), description("    //-V128
                    << get_description() << "), phase(" << get_thread_phase()
                    << "), rebind";

        free_thread_exit_callbacks();

        current_state_.store(thread_state(
            init_data.initial_state, thread_restart_state::signaled));

#ifdef HPX_HAVE_THREAD_DESCRIPTION
        description_ = (init_data.description);
        lco_description_ = util::thread_description();
#endif
#ifdef HPX_HAVE_THREAD_PARENT_REFERENCE
        parent_locality_id_ = init_data.parent_locality_id;
        parent_thread_id_ = init_data.parent_id;
        parent_thread_phase_ = init_data.parent_phase;
#endif
#ifdef HPX_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
        set_marked_state(thread_schedule_state::unknown);
#endif
#ifdef HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
        backtrace_ = nullptr;
#endif
        priority_ = init_data.priority;
        requested_interrupt_ = false;
        enabled_interrupt_ = true;
        ran_exit_funcs_ = false;
        exit_funcs_.clear();
        scheduler_base_ = init_data.scheduler_base;
        last_worker_thread_num_ = std::size_t(-1);

        HPX_ASSERT(stacksize_ == get_stack_size());
        HPX_ASSERT(stacksize_ != 0);

        LTM_(debug) << "thread::thread(" << this << "), description("
                    << get_description() << "), rebind";

#ifdef HPX_HAVE_THREAD_PARENT_REFERENCE
        // store the thread id of the parent thread, mainly for debugging
        // purposes
        if (nullptr == parent_thread_id_)
        {
            thread_self* self = get_self_ptr();
            if (self)
            {
                parent_thread_id_ = threads::get_self_id();
                parent_thread_phase_ = self->get_thread_phase();
            }
        }
        if (0 == parent_locality_id_)
        {
            parent_locality_id_ = detail::get_locality_id(hpx::throws);
        }
#endif
#if defined(HPX_HAVE_APEX)
        set_timer_data(init_data.timer_data);
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_self& get_self()
    {
        thread_self* p = get_self_ptr();
        if (HPX_UNLIKELY(p == nullptr))
        {
            HPX_THROW_EXCEPTION(null_thread_id, "threads::get_self",
                "null thread id encountered (is this executed on a "
                "HPX-thread?)");
        }
        return *p;
    }

    thread_self* get_self_ptr()
    {
        return thread_self::get_self();
    }

    namespace detail {
        void set_self_ptr(thread_self* self)
        {
            thread_self::set_self(self);
        }
    }    // namespace detail

    thread_self::impl_type* get_ctx_ptr()
    {
        using hpx::threads::coroutines::detail::coroutine_accessor;
        return coroutine_accessor::get_impl(get_self());
    }

    thread_self* get_self_ptr_checked(error_code& ec)
    {
        thread_self* p = thread_self::get_self();

        if (HPX_UNLIKELY(p == nullptr))
        {
            HPX_THROWS_IF(ec, null_thread_id, "threads::get_self_ptr_checked",
                "null thread id encountered (is this executed on a "
                "HPX-thread?)");
            return nullptr;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return p;
    }

    thread_id_type get_self_id()
    {
        thread_self* self = get_self_ptr();
        if (HPX_LIKELY(nullptr != self))
            return self->get_thread_id();

        return threads::invalid_thread_id;
    }

    thread_data* get_self_id_data()
    {
        thread_self* self = get_self_ptr();
        if (HPX_LIKELY(nullptr != self))
            return get_thread_id_data(self->get_thread_id());

        return nullptr;
    }

    std::ptrdiff_t get_self_stacksize()
    {
        thread_data* thrd_data = get_self_id_data();
        return thrd_data ? thrd_data->get_stack_size() : 0;
    }

    thread_stacksize get_self_stacksize_enum()
    {
        thread_data* thrd_data = get_self_id_data();
        thread_stacksize stacksize = thrd_data ?
            thrd_data->get_stack_size_enum() :
            thread_stacksize::default_;
        HPX_ASSERT(stacksize != thread_stacksize::current);
        return stacksize;
    }

#ifndef HPX_HAVE_THREAD_PARENT_REFERENCE
    thread_id_type get_parent_id()
    {
        return threads::invalid_thread_id;
    }

    std::size_t get_parent_phase()
    {
        return 0;
    }

    std::uint32_t get_parent_locality_id()
    {
        // same as naming::invalid_locality_id
        return ~static_cast<std::uint32_t>(0);
    }
#else
    thread_id_type get_parent_id()
    {
        thread_data* thrd_data = get_self_id_data();
        if (HPX_LIKELY(nullptr != thrd_data))
        {
            return thrd_data->get_parent_thread_id();
        }
        return threads::invalid_thread_id;
    }

    std::size_t get_parent_phase()
    {
        thread_data* thrd_data = get_self_id_data();
        if (HPX_LIKELY(nullptr != thrd_data))
        {
            return thrd_data->get_parent_thread_phase();
        }
        return 0;
    }

    std::uint32_t get_parent_locality_id()
    {
        thread_data* thrd_data = get_self_id_data();
        if (HPX_LIKELY(nullptr != thrd_data))
        {
            return thrd_data->get_parent_locality_id();
        }

        // same as naming::invalid_locality_id
        return ~static_cast<std::uint32_t>(0);
    }
#endif

    std::uint64_t get_self_component_id()
    {
#ifndef HPX_HAVE_THREAD_TARGET_ADDRESS
        return 0;
#else
        thread_data* thrd_data = get_self_id_data();
        if (HPX_LIKELY(nullptr != thrd_data))
        {
            return thrd_data->get_component_id();
        }
        return 0;
#endif
    }

#if defined(HPX_HAVE_APEX)
    std::shared_ptr<hpx::util::external_timer::task_wrapper>
    get_self_timer_data()
    {
        thread_data* thrd_data = get_self_id_data();
        if (HPX_LIKELY(nullptr != thrd_data))
        {
            return thrd_data->get_timer_data();
        }
        return nullptr;
    }
    void set_self_timer_data(
        std::shared_ptr<hpx::util::external_timer::task_wrapper> data)
    {
        thread_data* thrd_data = get_self_id_data();
        if (HPX_LIKELY(nullptr != thrd_data))
        {
            thrd_data->set_timer_data(data);
        }
        return;
    }
#endif
}}    // namespace hpx::threads
