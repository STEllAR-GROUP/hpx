//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/parallel_executor.hpp

#if !defined(HPX_ASYNC_LAUNCH_POLICY_DISPATCH_NOV_26_2017_1243PM)
#define HPX_ASYNC_LAUNCH_POLICY_DISPATCH_NOV_26_2017_1243PM

#include <hpx/config.hpp>
#include <hpx/lcos/async_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/futures_factory.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/util/deferred_call.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    HPX_FORCEINLINE
    typename std::enable_if<
        std::is_reference<
            typename util::detail::invoke_deferred_result<F>::type
        >::value
      , lcos::future<typename util::detail::invoke_deferred_result<F>::type>
    >::type
    call_sync(F && f, std::false_type)
    {
        typedef typename util::detail::invoke_deferred_result<F>::type result_type;
        try
        {
            return lcos::make_ready_future(std::ref(f()));
        }
        catch (...)
        {
            return lcos::make_exceptional_future<result_type>(
                std::current_exception());
        }
    }

    template <typename F>
    HPX_FORCEINLINE
    typename std::enable_if<
       !std::is_reference<
            typename util::detail::invoke_deferred_result<F>::type
        >::value
      , lcos::future<typename util::detail::invoke_deferred_result<F>::type>
    >::type
    call_sync(F && f, std::false_type) //-V659
    {
        typedef typename util::detail::invoke_deferred_result<F>::type result_type;
        try
        {
            return lcos::make_ready_future(f());
        }
        catch (...)
        {
            return lcos::make_exceptional_future<result_type>(
                std::current_exception());
        }
    }

    template <typename F>
    HPX_FORCEINLINE
    lcos::future<typename util::detail::invoke_deferred_result<F>::type>
    call_sync(F && f, std::true_type)
    {
        try
        {
            f();
            return lcos::make_ready_future();
        }
        catch (...)
        {
            return lcos::make_exceptional_future<void>(std::current_exception());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct async_launch_policy_dispatch<Action,
        typename std::enable_if<
            !traits::is_action<Action>::value
        >::type>
    {
        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            hpx::future<
                typename util::detail::invoke_deferred_result<F, Ts...>::type
            >
        >::type
        call(launch policy, F && f, Ts&&... ts)
        {
            typedef typename util::detail::invoke_deferred_result<F, Ts...>::type
                result_type;

            if (policy == launch::sync)
            {
                return detail::call_sync(
                    util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...),
                    typename std::is_void<result_type>::type());
            }

            lcos::local::futures_factory<result_type()> p(
                util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...));
            if (hpx::detail::has_async_policy(policy))
            {
                threads::thread_id_type tid = p.apply(policy, policy.priority());
                if (tid && policy == launch::fork)
                {
                    // make sure this thread is executed last
                    // yield_to
                    hpx::this_thread::suspend(threads::pending, tid,
                        "async_launch_policy_dispatch<fork>");
                }
            }
            return p.get_future();
        }

        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            hpx::future<
                typename util::detail::invoke_deferred_result<F, Ts...>::type
            >
        >::type
        call(hpx::detail::sync_policy, F && f, Ts&&... ts)
        {
            typedef typename util::detail::invoke_deferred_result<F, Ts...>::type
                result_type;

            return detail::call_sync(
                util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...),
                typename std::is_void<result_type>::type());
        }

        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            hpx::future<
                typename util::detail::invoke_deferred_result<F, Ts...>::type
            >
        >::type
        call(hpx::detail::async_policy policy, F && f, Ts&&... ts)
        {
            typedef typename util::detail::invoke_deferred_result<F, Ts...>::type
                result_type;

            lcos::local::futures_factory<result_type()> p(
                util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...));

            p.apply(policy, policy.priority());
            return p.get_future();
        }

        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            hpx::future<
                typename util::detail::invoke_deferred_result<F, Ts...>::type
            >
        >::type
        call(hpx::detail::fork_policy policy, F && f, Ts&&... ts)
        {
            typedef typename util::detail::invoke_deferred_result<F, Ts...>::type
                result_type;

            lcos::local::futures_factory<result_type()> p(
                util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...));

            // make sure this thread is executed last
            threads::thread_id_type tid = p.apply(policy, policy.priority());
            if (tid)
            {
                // yield_to
                hpx::this_thread::suspend(threads::pending, tid,
                    "async_launch_policy_dispatch<fork>");
            }
            return p.get_future();
        }

        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            hpx::future<
                typename util::detail::invoke_deferred_result<F, Ts...>::type
            >
        >::type
        call(hpx::detail::deferred_policy, F && f, Ts &&... ts)
        {
            typedef typename util::detail::invoke_deferred_result<F, Ts...>::type
                result_type;

            lcos::local::futures_factory<result_type()> p(
                util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...));

            return p.get_future();
        }
    };
}}

#endif
