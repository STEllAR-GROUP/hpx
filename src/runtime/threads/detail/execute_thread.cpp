//  Copyright (c) 2019-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>

#include <hpx/assertion.hpp>
#include <hpx/basic_execution/this_thread.hpp>
#include <hpx/coroutines/detail/get_stack_pointer.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/custom_exception_info.hpp>
#include <hpx/errors.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/lcos/local/futures_factory.hpp>
#include <hpx/lcos_fwd.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/threads/detail/execute_thread.hpp>
#include <hpx/runtime/threads/detail/scheduling_loop.hpp>
#include <hpx/runtime/threads/policies/scheduler_base.hpp>
#include <hpx/runtime/threads/register_thread.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/threads/thread_data.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <utility>

namespace hpx { namespace threads { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    // reuse the continuation recursion count here as well
    struct execute_thread_recursion_count
    {
        execute_thread_recursion_count() noexcept
          : count_(threads::get_continuation_recursion_count())
        {
            ++count_;
        }
        ~execute_thread_recursion_count() noexcept
        {
            --count_;
        }

        std::size_t& count_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    coroutine_type::result_type execute_thread_new_thread(F&& f)
    {
        lcos::local::futures_factory<coroutine_type::result_type()> p(
            std::forward<F>(f));

        // launch a new thread executing the given function, preferably on the
        // same core
        threads::thread_schedule_hint hint(
            static_cast<std::int16_t>(get_worker_thread_num()));

        // wait for the task to run
        threads::thread_id_type tid =
            p.apply(threads::detail::get_self_or_default_pool(),
                "execute_thread_new_thread", launch::fork,
                threads::thread_priority_boost,
                threads::thread_stacksize_current, hint);

        // make sure this thread is executed last
        hpx::this_thread::yield_to(thread::id(std::move(tid)));
        return p.get_future().get();
    }

    // make sure thread invocation does not recurse deeper than allowed
    HPX_FORCEINLINE coroutine_type::result_type handle_execute_thread(
        thread_data* thrd)
    {
        // We need to run the completion on a new thread
        HPX_ASSERT(nullptr != hpx::threads::get_self_ptr());

#if defined(HPX_HAVE_THREADS_GET_STACK_POINTER)
        bool recurse_asynchronously =
            !this_thread::has_sufficient_stack_space();
#else
        execute_thread_recursion_count cnt;
        bool recurse_asynchronously =
            cnt.count_ > HPX_CONTINUATION_MAX_RECURSION_DEPTH;
#endif
        auto* agent = basic_execution::this_thread::detail::get_agent_storage();
        if (!recurse_asynchronously)
        {
            // directly execute continuation on this thread
            return thrd->invoke_directly(agent);
        }
        else
        {
            // re-spawn continuation on a new thread
            try
            {
                return execute_thread_new_thread(util::deferred_call(
                    &thread_data::invoke_directly, thrd, agent));
            }
            catch (...)
            {
                // If an exception while creating the new task or inside the
                // completion handler is thrown, there is nothing we can do...
                // ... but terminate and report the error
                hpx::detail::report_exception_and_terminate(
                    std::current_exception());
            }
        }
    }

    bool execute_thread(thread_data* thrd)
    {
        thread_state state = thrd->get_state();
        thread_state_enum state_val = state.state();

        if (state_val == pending)
        {
            // tries to set state to active (only if state is still
            // the same as 'state')
            switch_status thrd_stat(thrd, state);
            if (HPX_LIKELY(thrd_stat.is_valid() &&
                    thrd_stat.get_previous() == pending))
            {
                thrd_stat = handle_execute_thread(thrd);
            }

            // store and retrieve the new state in the thread
            if (HPX_UNLIKELY(!thrd_stat.store_state(state)))
            {
                // some other worker-thread got in between and changed
                // the state of this thread - we just continue, assuming this
                // thread is handled elsewhere
                return false;
            }

            state_val = state.state();

            // any exception thrown from the thread will reset its
            // state at this point
        }

        if (state_val == terminated)
        {
            // we are responsible for destroying terminated threads
            static std::int64_t fake_busy_count = 0;
            thrd->get_scheduler_base()->destroy_thread(thrd, fake_busy_count);
            return true;
        }

        return false;
    }

}}}    // namespace hpx::threads::detail
