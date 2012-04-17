//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#include <hpx/runtime.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/components/runtime_support.hpp>
#include <hpx/lcos/future.hpp>

namespace hpx
{
    thread::thread() BOOST_NOEXCEPT
      : id_(threads::invalid_thread_id)
    {}

    thread::thread(BOOST_RV_REF(thread) rhs) BOOST_NOEXCEPT
      : id_(rhs.id_)
    {
        rhs.id_ = threads::invalid_thread_id;
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

    static void run_thread_exit_callbacks()
    {
        threads::thread_self& self = threads::get_self();
        threads::thread_id_type id = self.get_thread_id();
        if (id == threads::invalid_thread_id) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "run_thread_exit_callbacks",
                "NULL thread id encountered");
        }
        threads::run_thread_exit_callbacks(id);
    }

    threads::thread_state_enum thread::thread_function_nullary(
        HPX_STD_FUNCTION<void()> const& func)
    {
        try {
            func();
        }
        catch (hpx::exception const& e) {
            if (e.get_error() != hpx::thread_interrupted)
                throw;    // rethrow any exception except 'thread_interrupted'
        }

        // run all callbacks attached to the exit event for this thread
        run_thread_exit_callbacks();
        return threads::terminated;
    }

    thread::id thread::get_id() const BOOST_NOEXCEPT
    {
        return id(native_handle());
    }

    void thread::start_thread(BOOST_RV_REF(HPX_STD_FUNCTION<void()>) func)
    {
        threads::thread_init_data data(
            HPX_STD_BIND(&thread::thread_function_nullary, boost::move(func)),
            "thread::thread_function_nullary");

        error_code ec;
        threads::thread_id_type id = hpx::get_runtime().get_thread_manager().
            register_thread(data, threads::suspended, true, ec);
        if (ec) {
            HPX_THROWS_IF(ec, thread_resource_error, "thread::start_thread",
                "Could not create thread");
            return;
        }

        set_thread_state(id, threads::pending, threads::wait_signaled,
            threads::thread_priority_normal, ec);
        if (ec) {
            HPX_THROWS_IF(ec, thread_resource_error, "thread::start_thread",
                "Could not start newly created thread");
            return;
        }

        mutex_type::scoped_lock l(mtx_);
        id_ = id;
    }

    static void resume_thread(threads::thread_id_type id)
    {
        threads::set_thread_state(id, threads::pending);
    }

    void thread::join()
    {
        if (this_thread::get_id() == get_id())
        {
            HPX_THROW_EXCEPTION(thread_resource_error, "thread::join",
                "hpx::thread: trying joining itself");
            return;
        }

        this_thread::interruption_point();

        native_handle_type id = native_handle();
        if (id != threads::invalid_thread_id)
        {
            // register callback function to be called when thread exits
            if (threads::add_thread_exit_callback(id, HPX_STD_BIND(&resume_thread, id)))
            {
                // wait for thread to be terminated
                this_thread::suspend();
            }
        }

        detach();   // invalidate this object
    }

    // extensions
    void thread::interrupt()
    {
        threads::interrupt_thread(native_handle());
    }

    bool thread::interruption_requested() const
    {
        return threads::get_thread_interruption_requested(native_handle());
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct thread_task_base
          : lcos::detail::future_data<void>
        {
            typedef boost::intrusive_ptr<lcos::detail::future_data<void> >
                future_base_type;

        public:
            thread_task_base(threads::thread_id_type id)
              : valid_(false)
            {
                if (threads::add_thread_exit_callback(id,
                        HPX_STD_BIND(&thread_task_base::thread_exit_function,
                        this, future_base_type(this))))
                {
                    valid_ = true;
                }
            }

            // retrieving the value
            void get(error_code& ec = throws)
            {
                this->get_data(ec);
            }

            // moving out the value
            void move(error_code& ec = throws)
            {
                this->move_data(ec);
            }

            bool is_valid() const
            {
                return valid_;
            }

        protected:
            void thread_exit_function(future_base_type base)
            {
                base->set_data(result_type());
            }

        private:
            bool valid_;
        };
    }

    lcos::future<void> thread::get_future(error_code& ec)
    {
        if (id_ == threads::invalid_thread_id) {
            HPX_THROWS_IF(ec, null_thread_id, "thread::get_future",
                "NULL thread id encountered");
            return lcos::future<void>();
        }

        detail::thread_task_base* p = new detail::thread_task_base(id_);
        boost::intrusive_ptr<lcos::detail::future_data_base<void> > base(p);
        if (!p->is_valid()) {
            HPX_THROWS_IF(ec, thread_resource_error, "thread::get_future",
                "Could not create future as thread has been terminated.");
            return lcos::future<void>();
        }

        return lcos::future<void>(base);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace this_thread
    {
        void yield() BOOST_NOEXCEPT
        {
            this_thread::suspend();
        }

        thread::id get_id() BOOST_NOEXCEPT
        {
            threads::thread_self& self = threads::get_self();
            return thread::id(self.get_thread_id());
        }

        // extensions
        void interruption_point()
        {
            if (interruption_enabled() && interruption_requested())
            {
                throw hpx::exception(thread_interrupted,
                    "hpx::this_thread::interruption_point: "
                    "thread aborts itself due to requested thread interruption");
            }
        }

        bool interruption_enabled()
        {
            threads::thread_self& self = threads::get_self();
            return threads::get_thread_interruption_enabled(self.get_thread_id());
        }

        bool interruption_requested()
        {
            threads::thread_self& self = threads::get_self();
            return threads::get_thread_interruption_requested(self.get_thread_id());
        }

        void sleep_until(boost::posix_time::ptime const& at)
        {
            this_thread::suspend(at);
        }

        void sleep_for(boost::posix_time::time_duration const& p)
        {
            this_thread::suspend(p);
        }

        ///////////////////////////////////////////////////////////////////////
        disable_interruption::disable_interruption()
          : interruption_was_enabled_(interruption_enabled())
        {
            if (interruption_was_enabled_) {
                threads::thread_self& self = threads::get_self();
                threads::set_thread_interruption_enabled(self.get_thread_id(), false);
            }
        }

        disable_interruption::~disable_interruption()
        {
            threads::thread_self* p = threads::get_self_ptr();
            if (p) {
                threads::set_thread_interruption_enabled(p->get_thread_id(),
                    interruption_was_enabled_);
            }
        }

        ///////////////////////////////////////////////////////////////////////
        restore_interruption::restore_interruption(disable_interruption& d)
        {
            if (d.interruption_was_enabled_)
            {
                threads::thread_self& self = threads::get_self();
                threads::set_thread_interruption_enabled(
                    self.get_thread_id(), true);
            }
        }

        restore_interruption::~restore_interruption()
        {
            threads::thread_self* p = threads::get_self_ptr();
            if (p) {
                threads::set_thread_interruption_enabled(
                    p->get_thread_id(), false);
            }
        }
    }
}

