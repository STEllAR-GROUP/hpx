// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx
{
    
    
    
    template <typename F>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type()>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type()>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f)
    {
        typedef typename util::result_of<
            typename util::decay<F>::type()
        >::type result_type;
        if (policy == launch::sync)
        {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(boost::forward<F>(f), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            boost::forward<F>(f));
        if (detail::has_async_policy(policy))
            p.apply();
        return p.get_future();
    }
    template <typename F>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type()>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type()>
    >::type
    async(threads::executor& sched, BOOST_FWD_REF(F) f)
    {
        typedef typename util::result_of<
            typename util::decay<F>::type()
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            boost::forward<F>(f));
        p.apply();
        return p.get_future();
    }
    template <typename F>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type()>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type()>
    >::type
    async(BOOST_FWD_REF(F) f)
    {
        return async(launch::all, boost::forward<F>(f));
    }
    
    
    
    template <typename Action, typename BoundArgs>
    lcos::unique_future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(hpx::util::detail::bound_action<Action, BoundArgs> const& bound)
    {
        return bound.async();
    }
}
namespace hpx
{
    
    
    
    
    template <typename F, typename A0>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type)>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0)
    {
        typedef typename util::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(
                util::deferred_call(
                    boost::forward<F>(f),
                    boost::forward<A0>( a0 )
                ), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::deferred_call(
                boost::forward<F>(f),
                boost::forward<A0>( a0 )
            ));
        if (detail::has_async_policy(policy))
            p.apply();
        return p.get_future();
    }
    template <typename F, typename A0>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type)>
    >::type
    async(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0)
    {
        typedef typename util::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::deferred_call(
                boost::forward<F>(f),
                boost::forward<A0>( a0 )
            ));
        p.apply();
        return p.get_future();
    }
    template <typename F, typename A0>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type)>
    >::type
    async(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0)
    {
        return async(launch::all, boost::forward<F>(f),
            boost::forward<A0>( a0 ));
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0
    >
    lcos::unique_future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0
    )
    {
        return bound.async(boost::forward<A0>( a0 ));
    }
}
namespace hpx
{
    
    
    
    
    template <typename F, typename A0 , typename A1>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type)>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1)
    {
        typedef typename util::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(
                util::deferred_call(
                    boost::forward<F>(f),
                    boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )
                ), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::deferred_call(
                boost::forward<F>(f),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )
            ));
        if (detail::has_async_policy(policy))
            p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type)>
    >::type
    async(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1)
    {
        typedef typename util::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::deferred_call(
                boost::forward<F>(f),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )
            ));
        p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type)>
    >::type
    async(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1)
    {
        return async(launch::all, boost::forward<F>(f),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1
    >
    lcos::unique_future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1
    )
    {
        return bound.async(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
    }
}
namespace hpx
{
    
    
    
    
    template <typename F, typename A0 , typename A1 , typename A2>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type)>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2)
    {
        typedef typename util::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(
                util::deferred_call(
                    boost::forward<F>(f),
                    boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )
                ), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::deferred_call(
                boost::forward<F>(f),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )
            ));
        if (detail::has_async_policy(policy))
            p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type)>
    >::type
    async(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2)
    {
        typedef typename util::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::deferred_call(
                boost::forward<F>(f),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )
            ));
        p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type)>
    >::type
    async(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2)
    {
        return async(launch::all, boost::forward<F>(f),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1 , typename A2
    >
    lcos::unique_future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2
    )
    {
        return bound.async(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
    }
}
namespace hpx
{
    
    
    
    
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type)>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3)
    {
        typedef typename util::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(
                util::deferred_call(
                    boost::forward<F>(f),
                    boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )
                ), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::deferred_call(
                boost::forward<F>(f),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )
            ));
        if (detail::has_async_policy(policy))
            p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type)>
    >::type
    async(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3)
    {
        typedef typename util::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::deferred_call(
                boost::forward<F>(f),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )
            ));
        p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type)>
    >::type
    async(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3)
    {
        return async(launch::all, boost::forward<F>(f),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1 , typename A2 , typename A3
    >
    lcos::unique_future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3
    )
    {
        return bound.async(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
    }
}
namespace hpx
{
    
    
    
    
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type)>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4)
    {
        typedef typename util::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(
                util::deferred_call(
                    boost::forward<F>(f),
                    boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )
                ), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::deferred_call(
                boost::forward<F>(f),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )
            ));
        if (detail::has_async_policy(policy))
            p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type)>
    >::type
    async(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4)
    {
        typedef typename util::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::deferred_call(
                boost::forward<F>(f),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )
            ));
        p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type)>
    >::type
    async(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4)
    {
        return async(launch::all, boost::forward<F>(f),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
    >
    lcos::unique_future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4
    )
    {
        return bound.async(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
    }
}
