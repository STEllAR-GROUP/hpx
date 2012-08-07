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

#include <boost/coroutine/detail/coroutine_impl_impl.hpp>
#include <boost/assert.hpp>

///////////////////////////////////////////////////////////////////////////////
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::threads::detail::thread_data, hpx::components::component_thread)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace detail
{
    components::component_type thread_data::get_component_type()
    {
        return components::get_component_type<thread_data>();
    }

    void thread_data::set_component_type(components::component_type type)
    {
        components::set_component_type<thread_data>(type);
    }

    ///////////////////////////////////////////////////////////////////////////
    void *thread_data::operator new(std::size_t size, thread_pool& pool)
    {
        BOOST_ASSERT(sizeof(detail::thread_data) == size);

        void *ret = pool.detail_pool_.allocate();
        if (0 == ret)
            boost::throw_exception(std::bad_alloc());
        return ret;
    }

    void *thread_data::operator new(std::size_t size) throw()
    {
        return NULL;    // won't be ever used
    }

    void thread_data::operator delete(void *p, thread_pool& pool)
    {
        if (0 != p)
            pool.detail_pool_.deallocate(reinterpret_cast<detail::thread_data*>(p));
    }

    void thread_data::operator delete(void *p, std::size_t size)
    {
        BOOST_ASSERT(sizeof(detail::thread_data) == size);
        if (0 != p) {
            detail::thread_data* pt = reinterpret_cast<detail::thread_data*>(p);
            pt->pool_->detail_pool_.deallocate(pt);
        }
    }

    void thread_data::run_thread_exit_callbacks()
    {
        thread_mutex_type::scoped_lock l(this);
        while (exit_funcs_)
        {
            thread_exit_callback_node* const current_node = exit_funcs_;
            exit_funcs_ = current_node->next_;
            if (!current_node->f_.empty())
            {
                (current_node->f_)();
            }
            delete current_node;
        }
    }

    bool thread_data::add_thread_exit_callback(HPX_STD_FUNCTION<void()> const& f)
    {
        thread_mutex_type::scoped_lock l(this);
        if (get_state() == terminated)
            return false;

        thread_exit_callback_node* new_node =
            new thread_exit_callback_node(f, exit_funcs_);
        exit_funcs_ = new_node;
        return true;
    }

    void thread_data::free_thread_exit_callbacks()
    {
        thread_mutex_type::scoped_lock l(this);
        while (exit_funcs_)
        {
            thread_exit_callback_node* const current_node = exit_funcs_;
            exit_funcs_ = current_node->next_;
            delete current_node;
        }
    }
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    // This overload will be called by the ptr_map<> used in the thread_queue
    // whenever an instance of a threads::thread_data needs to be deleted. We
    // provide this overload as we need to extract the thread_pool from the
    // thread instance the moment before it gets deleted
    void delete_clone(threads::thread_data const* t)
    {
        if (0 != t) {
            threads::thread_pool* pool = t->get()->pool_;
            boost::checked_delete(t); // delete the normal way, memory does not get freed
            pool->pool_.free(const_cast<threads::thread_data*>(t));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void* thread_data::operator new(std::size_t size, thread_pool& pool)
    {
        BOOST_ASSERT(sizeof(thread_data) == size);

        void *ret = pool.pool_.alloc();
        if (0 == ret)
            boost::throw_exception(std::bad_alloc());
        return ret;
    }

    void thread_data::operator delete(void *p, thread_pool& pool)
    {
        if (0 != p)
            pool.pool_.free(reinterpret_cast<thread_data*>(p));
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
        return boost::coroutines::detail::coroutine_accessor::get_impl(get_self());
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
        return (0 != self) ? self->get_thread_id() : 0;
    }

    thread_id_type get_parent_id()
    {
        thread_self* self = get_self_ptr();
        return (0 != self) ?
            reinterpret_cast<thread_data*>(self->get_thread_id())->get_parent_thread_id() : 0;
    }

    std::size_t get_parent_phase()
    {
        thread_self* self = get_self_ptr();
        return (0 != self) ?
            reinterpret_cast<thread_data*>(self->get_thread_id())->get_parent_thread_phase() : 0;
    }

    boost::uint32_t get_parent_locality_id()
    {
        thread_self* self = get_self_ptr();
        return (0 != self) ?
            reinterpret_cast<thread_data*>(self->get_thread_id())->get_parent_locality_id() :
            naming::invalid_locality_id;
    }

    naming::address::address_type get_self_component_id()
    {
        thread_self* self = get_self_ptr();
        return (0 != self) ?
            reinterpret_cast<thread_data*>(self->get_thread_id())->get_component_id() : 0;
    }

    namespace detail
    {
        void thread_data::set_event()
        {
            // we need to reactivate the thread itself
            if (suspended == current_state_.load(boost::memory_order_acquire))
            {
                hpx::applier::get_applier().get_thread_manager().
                    set_state(get_thread_id(), pending);
            }
            // FIXME: implement functionality required for depleted state
        }
    }
}}

///////////////////////////////////////////////////////////////////////////////
// explicit instantiation of the function thread_self::set_self
template HPX_EXPORT void
hpx::threads::thread_self::impl_type::set_self(hpx::threads::thread_self*);

