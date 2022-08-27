//  Copyright (c) 2015-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/futures/detail/future_data.hpp>
#include <hpx/futures/future.hpp>

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/move_only_function.hpp>
#include <hpx/futures/futures_factory.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <utility>

namespace hpx { namespace lcos { namespace detail {

    static run_on_completed_error_handler_type run_on_completed_error_handler;

    void set_run_on_completed_error_handler(
        run_on_completed_error_handler_type f)
    {
        run_on_completed_error_handler = f;
    }

    future_data_refcnt_base::~future_data_refcnt_base() = default;

    ///////////////////////////////////////////////////////////////////////////
    struct handle_continuation_recursion_count
    {
        handle_continuation_recursion_count() noexcept
          : count_(threads::get_continuation_recursion_count())
        {
            ++count_;
        }
        ~handle_continuation_recursion_count()
        {
            --count_;
        }

        std::size_t& count_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Callback>
    static void run_on_completed_on_new_thread(Callback&& f)
    {
        lcos::local::futures_factory<void()> p(HPX_FORWARD(Callback, f));

        bool is_hpx_thread = nullptr != hpx::threads::get_self_ptr();
        hpx::launch policy = launch::fork;
        if (!is_hpx_thread)
        {
            policy = launch::async;
        }

        policy.set_priority(threads::thread_priority::boost);
        policy.set_stacksize(threads::thread_stacksize::current);

        // launch a new thread executing the given function
        threads::thread_id_ref_type tid =
            p.apply("run_on_completed_on_new_thread", policy);

        // wait for the task to run
        if (is_hpx_thread)
        {
            // make sure this thread is executed last
            this_thread::suspend(
                threads::thread_schedule_state::pending, tid.noref());
            return p.get_future().get();
        }

        // If we are not on a HPX thread, we need to return immediately, to
        // allow the newly spawned thread to execute.
    }

    ///////////////////////////////////////////////////////////////////////////
    future_data_base<traits::detail::future_data_void>::~future_data_base() =
        default;

    static util::unused_type unused_;

    util::unused_type*
    future_data_base<traits::detail::future_data_void>::get_result_void(
        void const* storage, error_code& ec)
    {
        // yields control if needed
        state s = wait(ec);
        if (ec)
        {
            return nullptr;
        }

        // No locking is required. Once a future has been made ready, which
        // is a postcondition of wait, either:
        //
        // - there is only one writer (future), or
        // - there are multiple readers only (shared_future, lock hurts
        //   concurrency)

        // Avoid retrieving state twice. If wait() returns 'empty' then this
        // thread was suspended, in this case we need to load it again.
        if (s == empty)
        {
            s = state_.load(std::memory_order_relaxed);
        }

        if (s == value)
        {
            return &unused_;
        }

        if (s == empty)
        {
            // the value has already been moved out of this future
            HPX_THROWS_IF(ec, no_state, "future_data_base::get_result",
                "this future has no valid shared state");
            return nullptr;
        }

        // the thread has been re-activated by one of the actions
        // supported by this promise (see promise::set_event
        // and promise::set_exception).
        if (s == exception)
        {
            std::exception_ptr const* exception_ptr =
                static_cast<std::exception_ptr const*>(storage);

            // an error has been reported in the meantime, throw or set
            // the error code
            if (&ec == &throws)
            {
                std::rethrow_exception(*exception_ptr);
                // never reached
            }
            else
            {
                ec = make_error_code(*exception_ptr);
            }
        }

        return nullptr;
    }

    // deferred execution of a given continuation
    void future_data_base<traits::detail::future_data_void>::run_on_completed(
        completed_callback_type&& on_completed) noexcept
    {
        hpx::detail::try_catch_exception_ptr(
            [&]() {
                hpx::scoped_annotation annotate(on_completed);
                on_completed();
            },
            [&](std::exception_ptr ep) {
                // If the completion handler throws an exception, there's nothing
                // we can do, report the exception and terminate.
                if (run_on_completed_error_handler)
                {
                    run_on_completed_error_handler(HPX_MOVE(ep));
                }
                else
                {
                    std::terminate();
                }
            });
    }

    void future_data_base<traits::detail::future_data_void>::run_on_completed(
        completed_callback_vector_type&& on_completed) noexcept
    {
        for (auto&& func : on_completed)
        {
            run_on_completed(HPX_MOVE(func));
        }
    }

    // make sure continuation invocation does not recurse deeper than
    // allowed
    template <typename Callback>
    void
    future_data_base<traits::detail::future_data_void>::handle_on_completed(
        Callback&& on_completed)
    {
        // We need to run the completion on a new thread if we are on a
        // non HPX thread.
#if defined(HPX_HAVE_THREADS_GET_STACK_POINTER)
        bool recurse_asynchronously =
            !this_thread::has_sufficient_stack_space();
#else
        handle_continuation_recursion_count cnt;
        bool recurse_asynchronously =
            cnt.count_ > HPX_CONTINUATION_MAX_RECURSION_DEPTH ||
            (hpx::threads::get_self_ptr() == nullptr);
#endif
        if (!recurse_asynchronously)
        {
            // directly execute continuation on this thread
            run_on_completed(HPX_FORWARD(Callback, on_completed));
        }
        else
        {
            // re-spawn continuation on a new thread

            hpx::detail::try_catch_exception_ptr(
                [&]() {
                    constexpr void (*p)(Callback &&) =
                        &future_data_base::run_on_completed;
                    run_on_completed_on_new_thread(util::deferred_call(
                        p, HPX_FORWARD(Callback, on_completed)));
                },
                [&](std::exception_ptr ep) {
                    // If an exception while creating the new task or inside the
                    // completion handler is thrown, there is nothing we can do...
                    // ... but terminate and report the error
                    if (run_on_completed_error_handler)
                    {
                        run_on_completed_error_handler(HPX_MOVE(ep));
                    }
                    else
                    {
                        std::rethrow_exception(HPX_MOVE(ep));
                    }
                });
        }
    }

    // We need only one explicit instantiation here as the second version
    // (single callback) is implicitly instantiated below.
    using completed_callback_vector_type =
        future_data_refcnt_base::completed_callback_vector_type;

    template HPX_CORE_EXPORT void
    future_data_base<traits::detail::future_data_void>::handle_on_completed<
        completed_callback_vector_type>(completed_callback_vector_type&&);

    /// Set the callback which needs to be invoked when the future becomes
    /// ready. If the future is ready the function will be invoked
    /// immediately.
    void future_data_base<traits::detail::future_data_void>::set_on_completed(
        completed_callback_type data_sink)
    {
        if (!data_sink)
            return;

        if (is_ready())
        {
            // invoke the callback (continuation) function right away
            handle_on_completed(HPX_MOVE(data_sink));
        }
        else
        {
            std::unique_lock l(mtx_);
            if (is_ready())
            {
                l.unlock();

                // invoke the callback (continuation) function
                handle_on_completed(HPX_MOVE(data_sink));
            }
            else
            {
                on_completed_.push_back(HPX_MOVE(data_sink));
            }
        }
    }

    future_data_base<traits::detail::future_data_void>::state
    future_data_base<traits::detail::future_data_void>::wait(error_code& ec)
    {
        // block if this entry is empty
        state s = state_.load(std::memory_order_acquire);
        if (s == empty)
        {
            std::unique_lock l(mtx_);
            s = state_.load(std::memory_order_relaxed);
            if (s == empty)
            {
                cond_.wait(l, "future_data_base::wait", ec);
                if (ec)
                {
                    return s;
                }

                // reload the state, it's not empty anymore
                s = state_.load(std::memory_order_relaxed);
            }
        }

        if (&ec != &throws)
        {
            ec = make_success_code();
        }
        return s;
    }

    hpx::future_status
    future_data_base<traits::detail::future_data_void>::wait_until(
        std::chrono::steady_clock::time_point const& abs_time, error_code& ec)
    {
        // block if this entry is empty
        if (state_.load(std::memory_order_acquire) == empty)
        {
            std::unique_lock l(mtx_);
            if (state_.load(std::memory_order_relaxed) == empty)
            {
                threads::thread_restart_state const reason = cond_.wait_until(
                    l, abs_time, "future_data_base::wait_until", ec);
                if (ec)
                {
                    return hpx::future_status::uninitialized;
                }

                if (reason == threads::thread_restart_state::timeout &&
                    state_.load(std::memory_order_acquire) == empty)
                {
                    return hpx::future_status::timeout;
                }
            }
        }

        if (&ec != &throws)
        {
            ec = make_success_code();
        }
        return hpx::future_status::ready;    //-V110
    }
}}}    // namespace hpx::lcos::detail
