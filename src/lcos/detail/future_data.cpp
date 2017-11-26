//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/lcos/future.hpp>
#include <hpx/lcos/detail/future_data.hpp>

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/annotated_function.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/detail/yield_k.hpp>
#include <hpx/util/unique_function.hpp>
#include <hpx/lcos/local/futures_factory.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include <boost/intrusive_ptr.hpp>

#include <exception>
#include <functional>
#include <mutex>
#include <utility>

namespace hpx { namespace lcos { namespace detail
{
    future_data_refcnt_base::~future_data_refcnt_base() = default;

    ///////////////////////////////////////////////////////////////////////////
    static bool run_on_completed_on_new_thread(
        util::unique_function_nonser<bool()> && f, error_code& ec)
    {
        lcos::local::futures_factory<bool()> p(std::move(f));

        bool is_hpx_thread = nullptr != hpx::threads::get_self_ptr();
        hpx::launch policy = launch::fork;
        if (!is_hpx_thread)
            policy = launch::async;

        // launch a new thread executing the given function
        threads::thread_id_type tid = p.apply(
            policy, threads::thread_priority_boost,
            threads::thread_stacksize_current, ec);
        if (ec) return false;

        // wait for the task to run
        if (is_hpx_thread)
        {
            // make sure this thread is executed last
            hpx::this_thread::yield_to(thread::id(std::move(tid)));
            return p.get_future().get(ec);
        }
        else
        {
            // If we are not on a HPX thread, we need to return immediately, to
            // allow the newly spawned thread to execute. This might swallow
            // possible exceptions bubbling up from the completion handler (which
            // shouldn't happen anyway...
            return true;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    future_data_base<traits::detail::future_data_void>::
        ~future_data_base()
    {}

    util::unused_type* future_data_base<traits::detail::future_data_void>::
        get_result_void(void const* storage, error_code& ec)
    {
        // yields control if needed
        wait(ec);
        if (ec) return nullptr;

        // No locking is required. Once a future has been made ready, which
        // is a postcondition of wait, either:
        //
        // - there is only one writer (future), or
        // - there are multiple readers only (shared_future, lock hurts
        //   concurrency)

        if (state_ == empty) {
            // the value has already been moved out of this future
            HPX_THROWS_IF(ec, no_state,
                "future_data_base::get_result",
                "this future has no valid shared state");
            return nullptr;
        }

        // the thread has been re-activated by one of the actions
        // supported by this promise (see promise::set_event
        // and promise::set_exception).
        if (state_ == exception)
        {
            std::exception_ptr const* exception_ptr =
                static_cast<std::exception_ptr const*>(storage);
            // an error has been reported in the meantime, throw or set
            // the error code
            if (&ec == &throws) {
                std::rethrow_exception(*exception_ptr);
                // never reached
            }
            else {
                ec = make_error_code(*exception_ptr);
            }
            return nullptr;
        }

        static util::unused_type unused_;
        return &unused_;
    }

    // deferred execution of a given continuation
    bool future_data_base<traits::detail::future_data_void>::
        run_on_completed(completed_callback_type && on_completed,
        std::exception_ptr& ptr)
    {
        try {
            hpx::util::annotate_function annotate(on_completed);
            on_completed();
        }
        catch (...) {
            ptr = std::current_exception();
            return false;
        }
        return true;
    }

    // make sure continuation invocation does not recurse deeper than
    // allowed
    void future_data_base<traits::detail::future_data_void>::
        handle_on_completed(completed_callback_type && on_completed)
    {
        // We need to run the completion on a new thread if we are on a
        // non HPX thread.
        bool recurse_asynchronously = hpx::threads::get_self_ptr() == nullptr;
#if defined(HPX_HAVE_THREADS_GET_STACK_POINTER)
        recurse_asynchronously =
            !this_thread::has_sufficient_stack_space();
#else
        handle_continuation_recursion_count cnt;
        recurse_asynchronously = recurse_asynchronously ||
            cnt.count_ > HPX_CONTINUATION_MAX_RECURSION_DEPTH;
#endif
        if (!recurse_asynchronously)
        {
            // directly execute continuation on this thread
            std::exception_ptr ptr;
            if (!run_on_completed(std::move(on_completed), ptr))
            {
                error_code ec(lightweight);
                set_exception(hpx::detail::access_exception(ec));
            }
        }
        else
        {
            // re-spawn continuation on a new thread
            boost::intrusive_ptr<future_data_base> this_(this);

            error_code ec(lightweight);
            std::exception_ptr ptr;
            if (!run_on_completed_on_new_thread(
                    util::deferred_call(
                        &future_data_base::run_on_completed,
                        std::move(this_), std::move(on_completed),
                        std::ref(ptr)),
                    ec))
            {
                // thread creation went wrong
                if (ec) {
                    set_exception(hpx::detail::access_exception(ec));
                    return;
                }

                // re-throw exception in this context
                HPX_ASSERT(ptr);        // exception should have been set
                std::rethrow_exception(ptr);
            }
        }
    }

    /// Set the callback which needs to be invoked when the future becomes
    /// ready. If the future is ready the function will be invoked
    /// immediately.
    void future_data_base<traits::detail::future_data_void>::
        set_on_completed(completed_callback_type data_sink)
    {
        if (!data_sink) return;

        std::unique_lock<mutex_type> l(mtx_);

        if (is_ready_locked(l)) {

            HPX_ASSERT(!on_completed_);

            // invoke the callback (continuation) function right away
            l.unlock();

            handle_on_completed(std::move(data_sink));
        }
        else {
            // store a combined callback wrapping the old and the new one
            // make sure continuations are evaluated in the order they are
            // attached
            on_completed_ = compose_cb(
                std::move(on_completed_), std::move(data_sink));
        }
    }

    void future_data_base<traits::detail::future_data_void>::
        wait(error_code& ec)
    {
        std::unique_lock<mutex_type> l(mtx_);

        // block if this entry is empty
        if (state_ == empty) {
            cond_.wait(l, "future_data_base::wait", ec);
            if (ec) return;
        }

        if (&ec != &throws)
            ec = make_success_code();
    }

    future_status future_data_base<traits::detail::future_data_void>::
        wait_until(util::steady_clock::time_point const& abs_time, error_code& ec)
    {
        std::unique_lock<mutex_type> l(mtx_);

        // block if this entry is empty
        if (state_ == empty) {
            threads::thread_state_ex_enum const reason =
                cond_.wait_until(l, abs_time,
                    "future_data_base::wait_until", ec);
            if (ec) return future_status::uninitialized;

            if (reason == threads::wait_timeout)
                return future_status::timeout;

            return future_status::ready;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return future_status::ready; //-V110
    }
}}}
