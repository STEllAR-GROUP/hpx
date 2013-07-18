//  Copyright (c) 2007-2012 Hartmut Kaiser
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
#include <hpx/util/coroutine/detail/coroutine_impl_impl.hpp>

#include <boost/assert.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    void* thread_data::operator new(std::size_t size, thread_pool& pool)
    {
        BOOST_ASSERT(sizeof(thread_data) == size);

        void *ret = reinterpret_cast<void*>(pool.allocate());
        if (0 == ret)
            HPX_THROW_EXCEPTION(out_of_memory,
                "thread_data::operator new",
                "could not allocate memory for thread_data");
        return ret;
    }

    void thread_data::operator delete(void *p, std::size_t size)
    {
        BOOST_ASSERT(sizeof(thread_data) == size);

        if (0 != p)
        {
            thread_data* pt = reinterpret_cast<thread_data*>(p);
            BOOST_ASSERT(pt->pool_);
            pt->pool_->deallocate(pt);
        }
    }

    void thread_data::operator delete(void *p, thread_pool& pool)
    {
        if (0 != p)
            pool.deallocate(reinterpret_cast<thread_data*>(p));
    }

    void thread_data::run_thread_exit_callbacks()
    {
        mutex_type::scoped_lock l(this);
        while (exit_funcs_)
        {
            detail::thread_exit_callback_node* const current_node = exit_funcs_;
            exit_funcs_ = current_node->next_;
            if (!current_node->f_.empty())
            {
                (current_node->f_)();
            }
            delete current_node;
        }
        ran_exit_funcs_ = true;
    }

    bool thread_data::add_thread_exit_callback(HPX_STD_FUNCTION<void()> const& f)
    {
        mutex_type::scoped_lock l(this);
        if (ran_exit_funcs_ || get_state() == terminated)
            return false;

        detail::thread_exit_callback_node* new_node =
            new detail::thread_exit_callback_node(f, exit_funcs_);
        exit_funcs_ = new_node;
        return true;
    }

    void thread_data::free_thread_exit_callbacks()
    {
        mutex_type::scoped_lock l(this);

        // Exit functions should have been executed.
        BOOST_ASSERT(!exit_funcs_ || ran_exit_funcs_);

        while (exit_funcs_)
        {
            detail::thread_exit_callback_node* const current_node = exit_funcs_;
            exit_funcs_ = current_node->next_;
            delete current_node;
        }
    }

    bool thread_data::interruption_point(bool throw_on_interrupt)
    {
        mutex_type::scoped_lock l(this);
        if (enabled_interrupt_ && requested_interrupt_)
        {
            l.unlock();

            // Verify that there are no more registered locks for this
            // OS-thread. This will throw if there are still any locks
            // held.
            util::force_error_on_lock();

            // now interrupt this thread
            if (throw_on_interrupt)
            {
                HPX_THROW_EXCEPTION(thread_interrupted,
                    "hpx::threads::thread_data::interruption_point",
                    "thread aborts itself due to requested thread interruption");
            }

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

    thread_self::impl_type* get_ctx_ptr()
    {
        return hpx::util::coroutines::detail::coroutine_accessor::get_impl(get_self());
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
        return (0 != self) ? self->get_thread_id() : threads::invalid_thread_id;
    }

#if !HPX_THREAD_MAINTAIN_PARENT_REFERENCE
    thread_id_type get_parent_id()
    {
        return threads::invalid_thread_id;
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
    thread_id_type get_parent_id()
    {
        thread_self* self = get_self_ptr();
        return (0 != self) ?
            reinterpret_cast<thread_data*>(self->get_thread_id())->get_parent_thread_id() :
            threads::invalid_thread_id;
    }

    std::size_t get_parent_phase()
    {
        thread_self* self = get_self_ptr();
        return (0 != self) ?
            reinterpret_cast<thread_data*>(self->get_thread_id())->get_parent_thread_phase() :
            0;
    }

    boost::uint32_t get_parent_locality_id()
    {
        thread_self* self = get_self_ptr();
        return (0 != self) ?
            reinterpret_cast<thread_data*>(self->get_thread_id())->get_parent_locality_id() :
            naming::invalid_locality_id;
    }
#endif

    naming::address::address_type get_self_component_id()
    {
#if !HPX_THREAD_MAINTAIN_TARGET_ADDRESS
        return 0;
#else
        thread_self* self = get_self_ptr();
        return (0 != self) ?
            reinterpret_cast<thread_data*>(self->get_thread_id())->get_component_id() : 0;
#endif
    }
}}

///////////////////////////////////////////////////////////////////////////////
// explicit instantiation of the function thread_self::set_self
template HPX_EXPORT void
hpx::threads::thread_self::impl_type::set_self(hpx::threads::thread_self*);

template HPX_EXPORT hpx::threads::thread_self*
hpx::threads::thread_self::impl_type::get_self();

template HPX_EXPORT void
hpx::threads::thread_self::impl_type::init_self();

template HPX_EXPORT void
hpx::threads::thread_self::impl_type::reset_self();
