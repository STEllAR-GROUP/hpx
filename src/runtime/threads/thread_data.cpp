//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assertion.hpp>
#include <hpx/basic_execution.hpp>
#include <hpx/coroutines/detail/coroutine_accessor.hpp>
#include <hpx/errors.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/logging.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/policies/scheduler_base.hpp>
#include <hpx/thread_support/unlock_guard.hpp>
#if defined(HPX_HAVE_APEX)
#include <hpx/util/external_timer.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <memory>

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads {

    thread_data::thread_data(thread_init_data& init_data, void* queue,
        thread_state_enum newstate)
      : current_state_(thread_state(newstate, wait_signaled))
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
      , marked_state_(unknown)
#endif
#ifdef HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
      , backtrace_(nullptr
#endif
      , priority_(init_data.priority)
      , requested_interrupt_(false)
      , enabled_interrupt_(true)
      , ran_exit_funcs_(false)
      , runs_as_child_(init_data.schedulehint.runs_as_child)
      , scheduler_base_(init_data.scheduler_base)
      , stacksize_(init_data.stacksize)
      , queue_(queue)
      , coroutine_(std::move(init_data.func), thread_id_type(this_()),
          init_data.stacksize)
      , agent_(coroutine_.impl())
    {
        LTM_(debug) << "thread::thread(" << this << "), description("
                    << get_description() << ")";

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
            parent_locality_id_ = get_locality_id();
#endif
#if defined(HPX_HAVE_APEX)
        set_timer_data(init_data.timer_data);
#endif
    }

    util::internal_allocator<thread_data> thread_data::thread_alloc_;

    thread_data::~thread_data()
    {
        LTM_(debug) << "~thread_data(" << this
                    << "), description("    //-V128
                    << this->get_description() << "), phase("
                    << this->get_thread_phase() << ")";

        free_thread_exit_callbacks();
    }

    void thread_data::destroy_thread()
    {
        this->get_scheduler_base()->destroy_thread(this);
    }

    void thread_data::run_thread_exit_callbacks()
    {
        mutex_type::scoped_lock l(this);

        while (!exit_funcs_.empty())
        {
            {
                hpx::util::unlock_guard<mutex_type::scoped_lock> ul(l);
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
        mutex_type::scoped_lock l(this);

        if (ran_exit_funcs_ || get_state().state() == terminated)
        {
            return false;
        }

        exit_funcs_.push_front(f);

        return true;
    }

    void thread_data::free_thread_exit_callbacks()
    {
        mutex_type::scoped_lock l(this);

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

    void thread_data::rebind_base(
        thread_init_data& init_data, thread_state_enum newstate)
    {
        LTM_(debug) << "thread_data::rebind_base(" << this
                    << "), description("    //-V128
                    << get_description() << "), phase("
                    << get_thread_phase() << "), rebind";

        free_thread_exit_callbacks();

        current_state_.store(thread_state(newstate, wait_signaled));

#ifdef HPX_HAVE_THREAD_DESCRIPTION
        description_ = init_data.description;
        lco_description_ = util::thread_description();
#endif
#ifdef HPX_HAVE_THREAD_PARENT_REFERENCE
        parent_locality_id_ = init_data.parent_locality_id;
        parent_thread_id_ = init_data.parent_id;
        parent_thread_phase_ = init_data.parent_phase;
#endif
#ifdef HPX_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
        set_marked_state(unknown);
#endif
#ifdef HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
        backtrace_ = nullptr;
#endif
        priority_ = init_data.priority;
        requested_interrupt_ = false;
        enabled_interrupt_ = true;
        ran_exit_funcs_ = false;

        *const_cast<bool*>(&runs_as_child_) =
            init_data.schedulehint.runs_as_child;

        exit_funcs_.clear();
        scheduler_base_ = init_data.scheduler_base;

        HPX_ASSERT(init_data.stacksize == get_stack_size());

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
            parent_locality_id_ = get_locality_id();
#endif
#if defined(HPX_HAVE_APEX)
        set_timer_data(init_data.timer_data);
#endif
        HPX_ASSERT(init_data.stacksize != 0);
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
    }     // namespace detail

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
            HPX_THROWS_IF(ec, null_thread_id,
                "threads::get_self_ptr_checked",
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

    std::size_t get_self_stacksize()
    {
        thread_data* thrd_data = get_self_id_data();
        return thrd_data ? thrd_data->get_stack_size() : 0;
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
        return naming::invalid_locality_id;
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
        return naming::invalid_locality_id;
    }
#endif

    naming::address::address_type get_self_component_id()
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
