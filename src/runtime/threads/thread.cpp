//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#include <hpx/runtime.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/components/runtime_support.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/util/register_locks.hpp>
#include <hpx/util/unlock_guard.hpp>

#include <mutex>

#if defined(__ANDROID__) || defined(ANDROID)
#include <cpu-features.h>
#endif

namespace hpx
{
    void thread::terminate(const char * function, const char * reason) const
    {
        try {
            // free all registered exit-callback functions
            threads::free_thread_exit_callbacks(id_);

            // report the error globally
            HPX_THROW_EXCEPTION(invalid_status, function, reason);
        }
        catch(...) {
            hpx::report_error(boost::current_exception());
            /* nothing else we can do */;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    thread::thread() HPX_NOEXCEPT
      : id_(threads::invalid_thread_id)
    {}

    thread::thread(thread && rhs) HPX_NOEXCEPT
      : id_(threads::invalid_thread_id)   // the rhs needs to end up with an invalid_id
    {
        std::lock_guard<mutex_type> l(rhs.mtx_);
        id_ = rhs.id_;
        rhs.id_ = threads::invalid_thread_id;
    }

    thread& thread::operator=(thread && rhs) HPX_NOEXCEPT
    {
        std::lock_guard<mutex_type> l(mtx_);
        std::lock_guard<mutex_type> l2(rhs.mtx_);
        // If our current thread is joinable, terminate
        if (joinable_locked())
        {
            terminate("thread::operator=", "destroying running thread");
        }
        id_ = rhs.id_;
        rhs.id_ = threads::invalid_thread_id;
        return *this;
    }

    thread::~thread()
    {
        // If the thread is still running, we terminate the whole application
        // as we have no chance of reporting this error (we can't throw)
        if (joinable_locked()) {
            terminate("thread::~thread", "destroying running thread");
        }
        threads::thread_id_type id = threads::invalid_thread_id;
        {
            std::lock_guard<mutex_type> l(mtx_);
            std::swap(id_, id);
        }
    }

    void thread::swap(thread& rhs) HPX_NOEXCEPT
    {
        std::lock_guard<mutex_type> l(mtx_);
        std::lock_guard<mutex_type> l2(rhs.mtx_);
        std::swap(id_, rhs.id_);
    }

    static void run_thread_exit_callbacks()
    {
        threads::thread_id_type id = threads::get_self_id();
        if (id == threads::invalid_thread_id) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "run_thread_exit_callbacks",
                "NULL thread id encountered");
        }
        threads::run_thread_exit_callbacks(id);
    }

    threads::thread_state_enum thread::thread_function_nullary(
        util::unique_function_nonser<void()> const& func)
    {
        try {
            // Now notify our calling thread that we started execution.
            func();
        }
        catch (hpx::thread_interrupted const&) { //-V565
            /* swallow this exception */
        }
        catch (hpx::exception const&) {
            // Verify that there are no more registered locks for this
            // OS-thread. This will throw if there are still any locks
            // held.
            util::force_error_on_lock();

            // run all callbacks attached to the exit event for this thread
            run_thread_exit_callbacks();

            throw;    // rethrow any exception except 'thread_interrupted'
        }

        // Verify that there are no more registered locks for this
        // OS-thread. This will throw if there are still any locks
        // held.
        util::force_error_on_lock();

        // run all callbacks attached to the exit event for this thread
        run_thread_exit_callbacks();
        return threads::terminated;
    }

    thread::id thread::get_id() const HPX_NOEXCEPT
    {
        return id(native_handle());
    }

    std::size_t thread::hardware_concurrency() HPX_NOEXCEPT
    {
        return hpx::threads::hardware_concurrency();
    }

    void thread::start_thread(util::unique_function_nonser<void()> && func)
    {
        threads::thread_init_data data(
            util::bind(util::one_shot(&thread::thread_function_nullary),
                std::move(func)),
            "thread::thread_function_nullary");

        // create the new thread, note that id_ is guaranteed to be valid
        // before the thread function is executed
        error_code ec(lightweight);
        hpx::get_runtime().get_thread_manager().
            register_thread(data, id_, threads::pending, true, ec);
        if (ec) {
            HPX_THROW_EXCEPTION(thread_resource_error, "thread::start_thread",
                "Could not create thread");
            return;
        }
    }

    static void resume_thread(threads::thread_id_type const& id)
    {
        threads::set_thread_state(id, threads::pending);
    }

    void thread::join()
    {
        std::unique_lock<mutex_type> l(mtx_);

        if(!joinable_locked())
        {
            terminate("thread::join", "trying to join a non joinable thread");
        }

        native_handle_type this_id = threads::get_self_id();
        if (this_id == id_)
        {
            HPX_THROW_EXCEPTION(thread_resource_error, "thread::join",
                "hpx::thread: trying joining itself");
            return;
        }
        this_thread::interruption_point();


        // register callback function to be called when thread exits
        if (threads::add_thread_exit_callback(id_,
                util::bind(&resume_thread, this_id)))
        {
            // wait for thread to be terminated
            util::unlock_guard<std::unique_lock<mutex_type> > ul(l);
            this_thread::suspend(threads::suspended, "thread::join");
        }

        detach_locked();   // invalidate this object
    }

    // extensions
    void thread::interrupt(bool flag)
    {
        threads::interrupt_thread(native_handle(), flag);
    }

    bool thread::interruption_requested() const
    {
        return threads::get_thread_interruption_requested(native_handle());
    }

    void thread::interrupt(thread::id id, bool flag)
    {
        threads::interrupt_thread(id.id_, flag);
    }

#ifdef HPX_HAVE_THREAD_LOCAL_STORAGE
    std::size_t thread::get_thread_data() const
    {
        return threads::get_thread_data(native_handle());
    }
    std::size_t thread::set_thread_data(std::size_t data)
    {
        return threads::set_thread_data(native_handle(), data);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct thread_task_base
          : lcos::detail::future_data<void>
        {
        private:
            typedef lcos::detail::future_data<void>::mutex_type mutex_type;
            typedef boost::intrusive_ptr<thread_task_base> future_base_type;

        protected:
            typedef lcos::detail::future_data<void>::result_type result_type;

        public:
            thread_task_base(threads::thread_id_type const& id)
              : id_(threads::invalid_thread_id)
            {
                if (threads::add_thread_exit_callback(id,
                        util::bind(&thread_task_base::thread_exit_function,
                            future_base_type(this))))
                {
                    id_ = id;
                }
            }

            bool valid() const
            {
                return id_ != threads::invalid_thread_id;
            }

            // cancellation support
            bool cancelable() const
            {
                return true;
            }

            void cancel()
            {
                std::lock_guard<mutex_type> l(this->mtx_);
                if (!this->is_ready()) {
                    threads::interrupt_thread(id_);
                    this->set_error(thread_cancelled,
                        "thread_task_base::cancel",
                        "future has been canceled");
                    id_ = threads::invalid_thread_id;
                }
            }

        protected:
            void thread_exit_function()
            {
                // might have been finished or canceled
                std::lock_guard<mutex_type> l(this->mtx_);
                if (!this->is_ready())
                    this->set_data(result_type());
                id_ = threads::invalid_thread_id;
            }

        private:
            threads::thread_id_type id_;
        };
    }

    lcos::future<void> thread::get_future(error_code& ec)
    {
        if (id_ == threads::invalid_thread_id)
        {
            HPX_THROWS_IF(ec, null_thread_id, "thread::get_future",
                "NULL thread id encountered");
            return lcos::future<void>();
        }

        detail::thread_task_base* p = new detail::thread_task_base(id_);
        boost::intrusive_ptr<lcos::detail::future_data<void> > base(p);
        if (!p->valid()) {
            HPX_THROWS_IF(ec, thread_resource_error, "thread::get_future",
                "Could not create future as thread has been terminated.");
            return lcos::future<void>();
        }

        using traits::future_access;
        return future_access<lcos::future<void> >::create(std::move(base));
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace this_thread
    {
        void yield() HPX_NOEXCEPT
        {
            this_thread::suspend(threads::pending, "this_thread::yield");
        }

        thread::id get_id() HPX_NOEXCEPT
        {
            return thread::id(threads::get_self_id());
        }

        // extensions
        threads::thread_priority get_priority()
        {
            return threads::get_thread_priority(threads::get_self_id());
        }

        std::ptrdiff_t get_stack_size()
        {
            return threads::get_stack_size(threads::get_self_id());
        }

        void interruption_point()
        {
            threads::interruption_point(threads::get_self_id());
        }

        bool interruption_enabled()
        {
            return threads::get_thread_interruption_enabled(threads::get_self_id());
        }

        bool interruption_requested()
        {
            return threads::get_thread_interruption_requested(threads::get_self_id());
        }

        void interrupt()
        {
            threads::interrupt_thread(threads::get_self_id());
            threads::interruption_point(threads::get_self_id());
        }

        void sleep_until(util::steady_time_point const& abs_time)
        {
            this_thread::suspend(abs_time, "this_thread::sleep_until");
        }

        ///////////////////////////////////////////////////////////////////////
        disable_interruption::disable_interruption()
          : interruption_was_enabled_(interruption_enabled())
        {
            if (interruption_was_enabled_) {
                interruption_was_enabled_ =
                    threads::set_thread_interruption_enabled(
                        threads::get_self_id(), false);
            }
        }

        disable_interruption::~disable_interruption()
        {
            threads::thread_self* p = threads::get_self_ptr();
            if (p) {
                threads::set_thread_interruption_enabled(
                    threads::get_self_id(), interruption_was_enabled_);
            }
        }

        ///////////////////////////////////////////////////////////////////////
        restore_interruption::restore_interruption(disable_interruption& d)
          : interruption_was_enabled_(d.interruption_was_enabled_)
        {
            if (!interruption_was_enabled_)
            {
                interruption_was_enabled_ =
                    threads::set_thread_interruption_enabled(
                        threads::get_self_id(), true);
            }
        }

        restore_interruption::~restore_interruption()
        {
            threads::thread_self* p = threads::get_self_ptr();
            if (p) {
                threads::set_thread_interruption_enabled(
                    threads::get_self_id(), interruption_was_enabled_);
            }
        }
    }
}

