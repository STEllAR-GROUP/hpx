//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#include <hpx/runtime.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/components/runtime_support.hpp>

namespace hpx { namespace threads
{
    thread::thread() BOOST_NOEXCEPT
      : id_(invalid_thread_id)
    {}

    thread::thread(BOOST_RV_REF(thread) rhs) BOOST_NOEXCEPT
      : id_(rhs.id_)
    {
        rhs.id_ = invalid_thread_id;
    }

    thread& thread::operator=(BOOST_RV_REF(thread) rhs) BOOST_NOEXCEPT
    {
        thread tmp(boost::move(rhs));
        swap(tmp);
        return *this;
    }

    thread::~thread()
    {
        // If the thread is still running, we terminate the whole application
        // as we have no chance of reporting this error (we can't throw)
        if (joinable()) {
            try {
                components::stubs::runtime_support::terminate_all(
                    naming::get_id_from_locality_id(HPX_AGAS_BOOTSTRAP_PREFIX));
            }
            catch(...) {
                /* nothing we can do */;
            }
        }
    }

    void thread::swap(thread& rhs) BOOST_NOEXCEPT
    {
        mutex_type::scoped_lock l(mtx_);
        std::swap(id_, rhs.id_);
    }

    static thread_state_enum
    thread_function_nullary(HPX_STD_FUNCTION<void()> const& func, thread* t)
    {
        try {
            func();
        }
        catch (hpx::exception const& e) {
            if (e.get_error() != hpx::thread_interrupted)
                throw;    // rethrow any exception except 'thread_interrupted'
        }

//        run_thread_exit_callbacks();

        // make sure our thread object knows that we're gone
        t->detach();
        return terminated;
    }

    thread::id thread::get_id() const BOOST_NOEXCEPT
    {
        return id(native_handle());
    }

    void thread::start_thread(BOOST_RV_REF(HPX_STD_FUNCTION<void()>) func)
    {
        threads::thread_init_data data(
            HPX_STD_BIND(&thread_function_nullary, boost::move(func), this),
            "<unknown>");

        error_code ec;
        thread_id_type id = hpx::get_runtime().get_thread_manager().
            register_thread(data, suspended, true, ec);
        if (ec) {
            HPX_THROWS_IF(ec, thread_resource_error, "thread::start_thread",
                "Could not create thread");
            return;
        }

        set_thread_state(id, pending, wait_signaled, thread_priority_normal, ec);
        if (ec) {
            HPX_THROWS_IF(ec, thread_resource_error, "thread::start_thread",
                "Could not start newly created thread");
            return;
        }

        mutex_type::scoped_lock l(mtx_);
        id_ = id;
    }

    void thread::join()
    {
        if (this_thread::get_id() == get_id())
        {
            HPX_THROW_EXCEPTION(thread_resource_error, "thread::join",
                "hpx::threads::thread: trying joining itself");
            return;
        }

        // wait for thread to be terminated, suspend for 10ms
        do {
            native_handle_type id = native_handle();
            if (id == invalid_thread_id || get_thread_state(id) == terminated)
                break;
            this_thread::suspend(boost::posix_time::milliseconds(10));
        } while (true);
    }

    void thread::yield() BOOST_NOEXCEPT
    {
        this_thread::yield();
    }

    void thread::sleep(boost::posix_time::ptime const& xt)
    {
        this_thread::suspend(xt);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace this_thread
    {
        void yield() BOOST_NOEXCEPT
        {
            threads::this_thread::suspend();
        }

        thread::id get_id() BOOST_NOEXCEPT
        {
            threads::thread_self& self = threads::get_self();
            return thread::id(self.get_thread_id());
        }
    }
}}

