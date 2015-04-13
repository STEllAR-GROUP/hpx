//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_ASYNC_APR_16_20012_0225PM)
#define HPX_ASYNC_APR_16_20012_0225PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/lcos/async_continue.hpp>
#include <hpx/lcos/detail/async_implementations.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/util/bind_action.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/move.hpp>
#include <hpx/traits/is_callable.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_void.hpp>

namespace hpx { namespace detail
{
    // Defer the evaluation of result_of during the SFINAE checks below
#if defined(__clang__)
    template <typename F, typename Result = typename util::deferred_call_result_of<F>::type>
    struct create_future
    {
        typedef lcos::future<Result> type;
    };
#else
    template <typename F, typename ResultOf = util::deferred_call_result_of<F> >
    struct create_future
    {
        typedef lcos::future<typename ResultOf::type> type;
    };
#endif

    template <typename F>
    BOOST_FORCEINLINE
    typename boost::lazy_enable_if<
        boost::is_reference<typename util::deferred_call_result_of<F()>::type>
      , detail::create_future<F()>
    >::type
    call_sync(F&& f, boost::mpl::false_)
    {
        typedef typename util::deferred_call_result_of<F()>::type result_type;
        try
        {
            return lcos::make_ready_future(boost::ref(f()));
        } catch (...) {
            return lcos::make_exceptional_future<result_type>(boost::current_exception());
        }
    }

    template <typename F>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        boost::is_reference<typename util::deferred_call_result_of<F()>::type>
      , detail::create_future<F()>
    >::type
    call_sync(F&& f, boost::mpl::false_) //-V659
    {
        typedef typename util::deferred_call_result_of<F()>::type result_type;
        try
        {
            return lcos::make_ready_future(f());
        } catch (...) {
            return lcos::make_exceptional_future<result_type>(boost::current_exception());
        }
    }

    template <typename F>
    BOOST_FORCEINLINE typename detail::create_future<F()>::type
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
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    // Define async() overloads for plain local functions and function objects.

    // Launch the given function or function object asynchronously and return a
    // future allowing to synchronize with the returned result.
    template <typename F, typename ...Ts>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(typename util::decay<Ts>::type...)
        >::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<F(Ts...)>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, F&& f, Ts&&... vs)
    {
        typedef typename util::deferred_call_result_of<
            F(Ts...)
        >::type result_type;

        if (policy == launch::sync) {
            return detail::call_sync(
                util::deferred_call(std::forward<F>(f), std::forward<Ts>(vs)...),
                typename boost::is_void<result_type>::type());
        }
        lcos::local::futures_factory<result_type()> p(
            util::deferred_call(std::forward<F>(f), std::forward<Ts>(vs)...));
        if (hpx::detail::has_async_policy(policy))
        {
            p.apply(policy);
            if (policy == launch::fork)
            {
                // make sure this thread is executed last
                hpx::this_thread::yield();
            }
        }
        return p.get_future();
    }

    template <typename F, typename ...Ts>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(typename util::decay<Ts>::type...)
        >::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<F(Ts...)>
    >::type
    async(threads::executor& sched, F&& f, Ts&&... vs)
    {
        typedef typename util::deferred_call_result_of<
            F(Ts...)
        >::type result_type;

        lcos::local::futures_factory<result_type()> p(sched,
            util::deferred_call(std::forward<F>(f), std::forward<Ts>(vs)...));
        p.apply();
        return p.get_future();
    }

    template <typename F, typename ...Ts>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(typename util::decay<Ts>::type...)
        >::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<F(Ts...)>
    >::type
    async(F&& f, Ts&&... vs)
    {
        return async(launch::all, std::forward<F>(f), std::forward<Ts>(vs)...);
    }

    // define async() overloads for bound actions
    template <typename Action, typename BoundArgs, typename ...Ts>
    lcos::future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(hpx::util::detail::bound_action<Action, BoundArgs> const& bound,
        Ts&&... vs)
    {
        return bound.async(std::forward<Ts>(vs)...);
    }
}

#endif
