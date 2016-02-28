//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/runtime/coroutine/detail/coroutine_impl_impl.hpp>

// #if HPX_DEBUG
// #  define HPX_DEBUG_THREAD_POOL 1
// #endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
#ifdef HPX_DEBUG_THREAD_POOL
    enum guard_value
    {
        initial_value = 0xcc,           // memory has been initialized
        freed_value = 0xdd              // memory has been freed
    };
#endif

    void intrusive_ptr_add_ref(thread_data* p)
    {
        ++p->count_;
    }
    void intrusive_ptr_release(thread_data* p)
    {
        if (0 == --p->count_)
        {
            thread_data::pool_type* pool = p->get_pool();
            p->~thread_data();
            pool->deallocate(p);
        }
    }

    void thread_data::run_thread_exit_callbacks()
    {
        mutex_type::scoped_lock l(this);

        while(!exit_funcs_.empty())
        {
            {
                hpx::util::unlock_guard<mutex_type::scoped_lock> ul(l);
                if(!exit_funcs_.back().empty())
                    exit_funcs_.back()();
            }
            exit_funcs_.pop_back();
        }
        ran_exit_funcs_ = true;
    }

    bool thread_data::add_thread_exit_callback(util
        ::function_nonser<void()> const& f)
    {
        mutex_type::scoped_lock l(this);
        if (ran_exit_funcs_ || get_state() == terminated)
        {
            return false;
        }

        exit_funcs_.push_back(f);

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
                boost::throw_exception(hpx::thread_interrupted());

            return true;
        }
        return false;
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_self& get_self()
    {
        thread_self* p = get_self_ptr();
        if (HPX_UNLIKELY(!p)) {
            HPX_THROW_EXCEPTION(null_thread_id, "threads::get_self",
                "NULL thread id encountered (is this executed on a HPX-thread?)");
        }
        return *p;
    }

    thread_self* get_self_ptr()
    {
        return thread_self::impl_type::get_self();
    }

    namespace detail
    {
        void set_self_ptr(thread_self* self)
        {
            thread_self::impl_type::set_self(self);
        }
    }

    thread_self::impl_type* get_ctx_ptr()
    {
        return hpx::coroutines::detail::coroutine_accessor::get_impl(get_self());
    }

    thread_self* get_self_ptr_checked(error_code& ec)
    {
        thread_self* p = thread_self::impl_type::get_self();

        if (HPX_UNLIKELY(!p))
        {
            HPX_THROWS_IF(ec, null_thread_id, "threads::get_self_ptr_checked",
                "NULL thread id encountered (is this executed on a HPX-thread?)");
            return 0;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return p;
    }

    thread_id_type get_self_id()
    {
        thread_self* self = get_self_ptr();
        if (0 == self)
            return threads::invalid_thread_id;

        return thread_id_type(
                reinterpret_cast<thread_data*>(self->get_thread_id())
            );
    }

#ifndef HPX_HAVE_THREAD_PARENT_REFERENCE
    thread_id_repr_type get_parent_id()
    {
        return threads::invalid_thread_id_repr;
    }

    std::size_t get_parent_phase()
    {
        return 0;
    }

    boost::uint32_t get_parent_locality_id()
    {
        return naming::invalid_locality_id;
    }
#else
    thread_id_repr_type get_parent_id()
    {
        thread_self* self = get_self_ptr();
        if (0 == self)
            return threads::invalid_thread_id_repr;
        return get_self_id()->get_parent_thread_id();
    }

    std::size_t get_parent_phase()
    {
        thread_self* self = get_self_ptr();
        if (0 == self)
            return 0;
        return get_self_id()->get_parent_thread_phase();
    }

    boost::uint32_t get_parent_locality_id()
    {
        thread_self* self = get_self_ptr();
        if (0 == self)
            return naming::invalid_locality_id;
        return get_self_id()->get_parent_locality_id();
    }
#endif

    naming::address::address_type get_self_component_id()
    {
#ifndef HPX_HAVE_THREAD_TARGET_ADDRESS
        return 0;
#else
        thread_self* self = get_self_ptr();
        if (0 == self)
            return 0;
        return get_self_id()->get_component_id();
#endif
    }
}}

///////////////////////////////////////////////////////////////////////////////
// explicit instantiation of the thread_self functions
template HPX_EXPORT void
hpx::threads::thread_self::impl_type::set_self(hpx::threads::thread_self*);

template HPX_EXPORT hpx::threads::thread_self*
hpx::threads::thread_self::impl_type::get_self();

template HPX_EXPORT void
hpx::threads::thread_self::impl_type::init_self();

template HPX_EXPORT void
hpx::threads::thread_self::impl_type::reset_self();
