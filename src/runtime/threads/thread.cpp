//  Copyright (c) 2008-2009 Chirag Dekate, Hartmut Kaiser, Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/threads/thread.hpp>

#include <boost/coroutine/detail/coroutine_impl_impl.hpp>
#include <boost/pool/singleton_pool.hpp>
#include <boost/assert.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace detail
{
    components::component_type thread::get_component_type()
    {
        return components::component_thread;
    }

    void thread::set_component_type(components::component_type type)
    {
        BOOST_ASSERT(false);    // shouldn't be called, ever
    }

    ///////////////////////////////////////////////////////////////////////////
    struct thread_tag {};

    // the used pool allocator doesn't need to be protected by a mutex as the
    // allocation always happens from inside the creation of the component
    // wrapper, which by itself is already protected by a mutex
    typedef boost::singleton_pool<
        thread_tag, sizeof(thread),
        boost::default_user_allocator_new_delete,
        boost::details::pool::default_mutex,
        0xFFFF                                      // start with 64k objects
    > pool_type;

    void *thread::operator new(std::size_t size)
    {
        BOOST_ASSERT(sizeof(thread) == size);

        void *ret = pool_type::malloc();
        if (0 == ret)
            boost::throw_exception(std::bad_alloc());
        return ret;
    }

    void thread::operator delete(void *p, std::size_t size)
    {
        BOOST_ASSERT(sizeof(thread) == size);
        if (0 != p) 
            pool_type::free(p);
    }

}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads 
{
    thread_self& get_self()
    {
        return *thread_self::impl_type::super_type::get_self();
    }

    thread_self* get_self_ptr()
    {
        return thread_self::impl_type::super_type::get_self();
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
            reinterpret_cast<thread*>(self->get_thread_id())->get_parent_thread_id() : 0;
    }

    boost::uint32_t get_parent_prefix()
    {
        thread_self* self = get_self_ptr();
        return (0 != self) ? 
            reinterpret_cast<thread*>(self->get_thread_id())->get_parent_locality_prefix() : 0;
    }

    naming::address::address_type get_self_component_id()
    {
        thread_self* self = get_self_ptr();
        return (0 != self) ? 
            reinterpret_cast<thread*>(self->get_thread_id())->get_component_id() : 0;
    }

    namespace detail
    {
        void thread::set_event()
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
template void hpx::threads::thread_self::impl_type::super_type::set_self(hpx::threads::thread_self*);

