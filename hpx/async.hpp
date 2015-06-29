//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_ASYNC_APR_16_20012_0225PM)
#define HPX_ASYNC_APR_16_20012_0225PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/lcos/async_continue.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/util/bind_action.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/traits/is_launch_policy.hpp>
#include <hpx/runtime/launch_policy.hpp>

#include <boost/type_traits/is_void.hpp>
#include <boost/utility/enable_if.hpp>

#include <type_traits>

namespace hpx { namespace detail
{
    // Defer the evaluation of result_of during the SFINAE checks below
#if defined(__clang__)
    template <typename F, typename Result =
        typename util::detail::deferred_result_of<F>::type>
    struct create_future
    {
        typedef lcos::future<Result> type;
    };
#else
    template <typename F, typename ResultOf = util::detail::deferred_result_of<F> >
    struct create_future
    {
        typedef lcos::future<typename ResultOf::type> type;
    };
#endif

    template <typename F>
    HPX_FORCEINLINE
    typename boost::lazy_enable_if<
        boost::is_reference<typename util::detail::deferred_result_of<F()>::type>
      , detail::create_future<F()>
    >::type
    call_sync(F&& f, boost::mpl::false_)
    {
        typedef typename util::detail::deferred_result_of<F()>::type result_type;
        try
        {
            return lcos::make_ready_future(boost::ref(f()));
        } catch (...) {
            return lcos::make_exceptional_future<result_type>
                (boost::current_exception());
        }
    }

    template <typename F>
    HPX_FORCEINLINE
    typename boost::lazy_disable_if<
        boost::is_reference<typename util::detail::deferred_result_of<F()>::type>
      , detail::create_future<F()>
    >::type
    call_sync(F&& f, boost::mpl::false_) //-V659
    {
        typedef typename util::detail::deferred_result_of<F()>::type result_type;
        try
        {
            return lcos::make_ready_future(f());
        } catch (...) {
            return lcos::make_exceptional_future<result_type>
                (boost::current_exception());
        }
    }

    template <typename F>
    HPX_FORCEINLINE typename detail::create_future<F()>::type
    call_sync(F&& f, boost::mpl::true_)
    {
        try
        {
            f();
            return lcos::make_ready_future();
        } catch (...) {
            return lcos::make_exceptional_future<void>(boost::current_exception());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct async_launch_policy_dispatch<Action,
        typename boost::enable_if_c<
            !traits::is_action<Action>::value
        >::type>
    {
        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename boost::enable_if_c<
            traits::detail::is_deferred_callable<F(Ts&&...)>::value,
            hpx::future<typename util::detail::deferred_result_of<F(Ts&&...)>::type>
        >::type
        call(launch launch_policy, F&& f, Ts&&... ts)
        {
            typedef typename util::detail::deferred_result_of<
                F(Ts&&...)
            >::type result_type;

            if (launch_policy == launch::sync) {
                return detail::call_sync(
                    util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...),
                    typename boost::is_void<result_type>::type());
            }
            lcos::local::futures_factory<result_type()> p(
                util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...));
            if (hpx::detail::has_async_policy(launch_policy))
            {
                p.apply(launch_policy);
                if (launch_policy == launch::fork)
                {
                    // make sure this thread is executed last
                    hpx::this_thread::yield();
                }
            }
            return p.get_future();
        }
    };

    // BOOST_SCOPED_ENUM(launch) requires one more level of dispatch
    template <typename Policy>
    struct async_dispatch<Policy,
        typename boost::enable_if_c<
            traits::is_launch_policy<Policy>::value
        >::type>
    {
        template <typename F, typename ...Ts>
        BOOST_FORCEINLINE static
        auto call(BOOST_SCOPED_ENUM(launch) const& launch_policy, F && f,
                Ts&&... ts)
         -> decltype(
                async_policy_dispatch<
                    typename util::decay<F>::type
                >::call(launch_policy, std::forward<F>(f),
                    std::forward<Ts>(ts)...)
            )
        {
            return async_policy_dispatch<
                    typename util::decay<F>::type
                >::call(launch_policy, std::forward<F>(f),
                    std::forward<Ts>(ts)...);
        }
    };

    // Launch the given function or function object asynchronously and return a
    // future allowing to synchronize with the returned result.
    template <typename Func, typename Enable>
    struct async_dispatch
    {
        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename boost::enable_if_c<
            traits::detail::is_deferred_callable<F(Ts&&...)>::value,
            hpx::future<typename util::detail::deferred_result_of<F(Ts&&...)>::type>
        >::type
        call(F&& f, Ts&&... ts)
        {
            return async_dispatch<launch>::call(
                launch::all, std::forward<F>(f), std::forward<Ts>(ts)...);
        }
    };

    // threads::executor
    template <typename Executor>
    struct async_dispatch<Executor,
        typename boost::enable_if_c<
            traits::is_threads_executor<Executor>::value
        >::type>
    {
        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename boost::enable_if_c<
            traits::detail::is_deferred_callable<F(Ts&&...)>::value,
            hpx::future<typename util::detail::deferred_result_of<F(Ts&&...)>::type>
        >::type
        call(Executor& sched, F&& f, Ts&&... ts)
        {
            typedef typename util::detail::deferred_result_of<
                    F(Ts&&...)
                >::type result_type;

            lcos::local::futures_factory<result_type()> p(sched,
                util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...));
            p.apply();
            return p.get_future();
        }
    };

    // parallel::executor
    template <typename Executor>
    struct async_dispatch<Executor,
        typename boost::enable_if_c<
            traits::is_executor<Executor>::value
        >::type>
    {
        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename boost::enable_if_c<
            traits::detail::is_deferred_callable<F(Ts&&...)>::value,
            hpx::future<typename util::detail::deferred_result_of<F(Ts&&...)>::type>
        >::type
        call(Executor& exec, F&& f, Ts&&... ts)
        {
            return parallel::executor_traits<Executor>::async_execute(exec,
                util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...));
        }
    };

    // bound action
    template <typename Bound>
    struct async_dispatch<Bound,
        typename boost::enable_if_c<
            traits::is_bound_action<Bound>::value
        >::type>
    {
        template <typename Action, typename BoundArgs, typename ...Ts>
        HPX_FORCEINLINE
        static hpx::future<typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type>
        call(hpx::util::detail::bound_action<Action, BoundArgs> const& bound,
            Ts&&... ts)
        {
            return bound.async(std::forward<Ts>(ts)...);
        }
    };
}}

namespace hpx
{
    template <typename F, typename ...Ts>
    HPX_FORCEINLINE auto async(F&& f, Ts&&... ts)
    ->  decltype(detail::async_dispatch<typename util::decay<F>::type>::call(
            std::forward<F>(f), std::forward<Ts>(ts)...
        ))
    {
        return detail::async_dispatch<
                typename util::decay<F>::type
            >::call(std::forward<F>(f), std::forward<Ts>(ts)...);
    }
}

#endif
