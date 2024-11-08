//  Copyright (c) 2007-2024 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/coroutines/detail/coroutine_accessor.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/lock_registration/detail/register_locks.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/thread_support/unlock_guard.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/thread_data.hpp>
#if defined(HPX_HAVE_APEX)
#include <hpx/threading_base/external_timer.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <memory>

////////////////////////////////////////////////////////////////////////////////
namespace hpx::threads {

    namespace detail {

        namespace {

            get_locality_id_type* get_locality_id_f = nullptr;
        }

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
        std::ptrdiff_t stacksize, bool is_stackless, thread_id_addref addref)
      : detail::thread_data_reference_counting(addref)
      , current_state_(thread_state(
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
      , is_stackless_(is_stackless)
      , runs_as_child_(init_data.schedulehint.runs_as_child_mode() ==
            hpx::threads::thread_execution_hint::run_as_child)
      , scheduler_base_(init_data.scheduler_base)
      , last_worker_thread_num_(static_cast<std::size_t>(-1))
      , stacksize_(stacksize)
      , stacksize_enum_(init_data.stacksize)
      , queue_(queue)
    {
        LTM_(debug).format(
            "thread::thread({}), description({})", this, get_description());

        HPX_ASSERT(stacksize_enum_ != threads::thread_stacksize::current);

#ifdef HPX_HAVE_THREAD_PARENT_REFERENCE
        // store the thread id of the parent thread, mainly for debugging
        // purposes
        if (parent_thread_id_ == nullptr)
        {
            if (thread_self const* self = get_self_ptr())
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
        LTM_(debug).format("thread_data::~thread_data({})", this);
        free_thread_exit_callbacks();
    }

    void thread_data::destroy_thread()
    {
        LTM_(debug).format(
            "thread_data::destroy_thread({}), description({}), phase({})", this,
            this->get_description(), this->get_thread_phase());

        get_scheduler_base()->destroy_thread(this);
    }

    void thread_data::run_thread_exit_callbacks()
    {
        std::unique_lock<hpx::util::detail::spinlock> l(
            spinlock_pool::spinlock_for(this));

        while (!exit_funcs_.empty())
        {
            {
                hpx::unlock_guard<std::unique_lock<hpx::util::detail::spinlock>>
                    ul(l);
                if (!exit_funcs_.front().empty())
                    exit_funcs_.front()();
            }
            exit_funcs_.pop_front();
        }
        ran_exit_funcs_ = true;
    }

    bool thread_data::add_thread_exit_callback(hpx::function<void()> const& f)
    {
        std::lock_guard<hpx::util::detail::spinlock> l(
            spinlock_pool::spinlock_for(this));

        if (ran_exit_funcs_ ||
            get_state().state() == thread_schedule_state::terminated ||
            get_state().state() == thread_schedule_state::deleted)
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
        // We do not protect enabled_interrupt_ and requested_interrupt_ from
        // concurrent access here (which creates a benign data race) in order to
        // avoid infinite recursion. This function is called by
        // this_thread::suspend which causes problems if the lock would call
        // suspend itself.
        if (enabled_interrupt_ && requested_interrupt_)
        {
            // Verify that there are no more registered locks for this
            // OS-thread. This will throw if there are still any locks held.
            util::force_error_on_lock();

            // now interrupt this thread
            if (throw_on_interrupt)
            {
                requested_interrupt_ = false;    // avoid recursive exceptions
                throw hpx::thread_interrupted();
            }

            return true;
        }
        return false;
    }

    void thread_data::rebind_base(thread_init_data& init_data)
    {
        LTM_(debug).format(
            "thread_data::rebind_base({}), description({}), phase({}), rebind",
            this, get_description(), get_thread_phase());

        free_thread_exit_callbacks();

        current_state_.store(thread_state(
            init_data.initial_state, thread_restart_state::signaled));

#ifdef HPX_HAVE_THREAD_DESCRIPTION
        description_ = init_data.description;
        lco_description_ = threads::thread_description();
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

        runs_as_child_.store(init_data.schedulehint.runs_as_child_mode() ==
                hpx::threads::thread_execution_hint::run_as_child,
            std::memory_order_relaxed);

        exit_funcs_.clear();
        scheduler_base_ = init_data.scheduler_base;
        last_worker_thread_num_ = static_cast<std::size_t>(-1);

        // We explicitly set the logical stack size again as it can be different
        // from what the previous use required. However, the physical stack size
        // must be the same as before.
        stacksize_enum_ = init_data.stacksize;
        HPX_ASSERT(stacksize_ == get_stack_size());
        HPX_ASSERT(stacksize_ != 0);

        LTM_(debug).format("thread::thread({}), description({}), rebind", this,
            get_description());

#ifdef HPX_HAVE_THREAD_PARENT_REFERENCE
        // store the thread id of the parent thread, mainly for debugging
        // purposes
        if (parent_thread_id_ == nullptr)
        {
            if (thread_self const* self = get_self_ptr())
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
            HPX_THROW_EXCEPTION(hpx::error::null_thread_id, "threads::get_self",
                "null thread id encountered (is this executed on a "
                "HPX-thread?)");
        }
        return *p;
    }

    thread_self* get_self_ptr() noexcept
    {
        return thread_self::get_self();
    }

    namespace detail {

        void set_self_ptr(thread_self* self) noexcept
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
            HPX_THROWS_IF(ec, hpx::error::null_thread_id,
                "threads::get_self_ptr_checked",
                "null thread id encountered (is this executed on a "
                "HPX-thread?)");
            return nullptr;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return p;
    }

    thread_id_type get_self_id() noexcept
    {
        if (thread_self const* self = get_self_ptr();
            HPX_LIKELY(nullptr != self))
        {
            return self->get_outer_thread_id();
        }
        return threads::invalid_thread_id;
    }

    thread_data* get_self_id_data() noexcept
    {
        if (thread_self const* self = get_self_ptr();
            HPX_LIKELY(nullptr != self))
        {
            return get_thread_id_data(self->get_outer_thread_id());
        }
        return nullptr;
    }

    std::ptrdiff_t get_self_stacksize() noexcept
    {
        thread_data const* thrd_data = get_self_id_data();
        return thrd_data ? thrd_data->get_stack_size() : 0;
    }

    thread_stacksize get_self_stacksize_enum() noexcept
    {
        thread_data const* thrd_data = get_self_id_data();
        thread_stacksize const stacksize = thrd_data ?
            thrd_data->get_stack_size_enum() :
            thread_stacksize::default_;
        HPX_ASSERT(stacksize != thread_stacksize::current);
        return stacksize;
    }

#ifndef HPX_HAVE_THREAD_PARENT_REFERENCE
    thread_id_type get_parent_id() noexcept
    {
        return threads::invalid_thread_id;
    }

    std::size_t get_parent_phase() noexcept
    {
        return 0;
    }

    std::uint32_t get_parent_locality_id() noexcept
    {
        // same as naming::invalid_locality_id
        return ~static_cast<std::uint32_t>(0);
    }
#else
    thread_id_type get_parent_id() noexcept
    {
        if (thread_data const* thrd_data = get_self_id_data();
            HPX_LIKELY(nullptr != thrd_data))
        {
            return thrd_data->get_parent_thread_id();
        }
        return threads::invalid_thread_id;
    }

    std::size_t get_parent_phase() noexcept
    {
        if (thread_data const* thrd_data = get_self_id_data();
            HPX_LIKELY(nullptr != thrd_data))
        {
            return thrd_data->get_parent_thread_phase();
        }
        return 0;
    }

    std::uint32_t get_parent_locality_id() noexcept
    {
        if (thread_data const* thrd_data = get_self_id_data();
            HPX_LIKELY(nullptr != thrd_data))
        {
            return thrd_data->get_parent_locality_id();
        }

        // same as naming::invalid_locality_id
        return ~static_cast<std::uint32_t>(0);
    }
#endif

    std::uint64_t get_self_component_id() noexcept
    {
#ifndef HPX_HAVE_THREAD_TARGET_ADDRESS
        return 0;
#else
        if (thread_data const* thrd_data = get_self_id_data();
            HPX_LIKELY(nullptr != thrd_data))
        {
            return hpx::threads::thread_data::get_component_id();
        }
        return 0;
#endif
    }

#if defined(HPX_HAVE_APEX)
    std::shared_ptr<hpx::util::external_timer::task_wrapper>
    get_self_timer_data()
    {
        if (thread_data* thrd_data = get_self_id_data();
            HPX_LIKELY(nullptr != thrd_data))
        {
            return thrd_data->get_timer_data();
        }
        return nullptr;
    }

    void set_self_timer_data(
        std::shared_ptr<hpx::util::external_timer::task_wrapper> data)
    {
        if (thread_data* thrd_data = get_self_id_data();
            HPX_LIKELY(nullptr != thrd_data))
        {
            thrd_data->set_timer_data(data);
        }
    }
#endif
}    // namespace hpx::threads
