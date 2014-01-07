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
        typedef typename boost::result_of<
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
        typedef typename boost::result_of<
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
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(util::bind(
                util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 )), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 )));
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
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 )));
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
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(util::bind(
                util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )));
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
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )));
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
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(util::bind(
                util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )));
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
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )));
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
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(util::bind(
                util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )));
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
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )));
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
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(util::bind(
                util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )));
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
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )));
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
namespace hpx
{
    
    
    
    
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type)>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(util::bind(
                util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )));
        if (detail::has_async_policy(policy))
            p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type)>
    >::type
    async(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )));
        p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type)>
    >::type
    async(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5)
    {
        return async(launch::all, boost::forward<F>(f),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ));
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
    >
    lcos::unique_future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5
    )
    {
        return bound.async(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ));
    }
}
namespace hpx
{
    
    
    
    
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type)>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(util::bind(
                util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )));
        if (detail::has_async_policy(policy))
            p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type)>
    >::type
    async(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )));
        p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type)>
    >::type
    async(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6)
    {
        return async(launch::all, boost::forward<F>(f),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ));
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
    >
    lcos::unique_future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6
    )
    {
        return bound.async(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ));
    }
}
namespace hpx
{
    
    
    
    
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type)>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(util::bind(
                util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 )), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 )));
        if (detail::has_async_policy(policy))
            p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type)>
    >::type
    async(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 )));
        p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type)>
    >::type
    async(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7)
    {
        return async(launch::all, boost::forward<F>(f),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ));
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
    >
    lcos::unique_future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7
    )
    {
        return bound.async(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ));
    }
}
namespace hpx
{
    
    
    
    
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type)>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(util::bind(
                util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 )), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 )));
        if (detail::has_async_policy(policy))
            p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type)>
    >::type
    async(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 )));
        p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type)>
    >::type
    async(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8)
    {
        return async(launch::all, boost::forward<F>(f),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ));
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
    >
    lcos::unique_future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8
    )
    {
        return bound.async(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ));
    }
}
namespace hpx
{
    
    
    
    
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type)>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(util::bind(
                util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 )), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 )));
        if (detail::has_async_policy(policy))
            p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type)>
    >::type
    async(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 )));
        p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type)>
    >::type
    async(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9)
    {
        return async(launch::all, boost::forward<F>(f),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ));
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
    >
    lcos::unique_future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9
    )
    {
        return bound.async(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ));
    }
}
namespace hpx
{
    
    
    
    
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type)>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(util::bind(
                util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 )), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 )));
        if (detail::has_async_policy(policy))
            p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type)>
    >::type
    async(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 )));
        p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type)>
    >::type
    async(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10)
    {
        return async(launch::all, boost::forward<F>(f),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ));
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
    >
    lcos::unique_future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10
    )
    {
        return bound.async(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ));
    }
}
namespace hpx
{
    
    
    
    
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type)>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(util::bind(
                util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 )), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 )));
        if (detail::has_async_policy(policy))
            p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type)>
    >::type
    async(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 )));
        p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type)>
    >::type
    async(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11)
    {
        return async(launch::all, boost::forward<F>(f),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ));
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
    >
    lcos::unique_future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11
    )
    {
        return bound.async(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ));
    }
}
namespace hpx
{
    
    
    
    
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type)>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(util::bind(
                util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 )), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 )));
        if (detail::has_async_policy(policy))
            p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type)>
    >::type
    async(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 )));
        p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type)>
    >::type
    async(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12)
    {
        return async(launch::all, boost::forward<F>(f),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ));
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
    >
    lcos::unique_future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12
    )
    {
        return bound.async(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ));
    }
}
namespace hpx
{
    
    
    
    
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type)>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(util::bind(
                util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 )), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 )));
        if (detail::has_async_policy(policy))
            p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type)>
    >::type
    async(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 )));
        p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type)>
    >::type
    async(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13)
    {
        return async(launch::all, boost::forward<F>(f),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ));
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
    >
    lcos::unique_future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13
    )
    {
        return bound.async(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ));
    }
}
namespace hpx
{
    
    
    
    
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type)>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(util::bind(
                util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 )), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 )));
        if (detail::has_async_policy(policy))
            p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type)>
    >::type
    async(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 )));
        p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type)>
    >::type
    async(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14)
    {
        return async(launch::all, boost::forward<F>(f),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ));
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
    >
    lcos::unique_future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14
    )
    {
        return bound.async(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ));
    }
}
namespace hpx
{
    
    
    
    
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type)>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(util::bind(
                util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 )), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 )));
        if (detail::has_async_policy(policy))
            p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type)>
    >::type
    async(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 )));
        p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type)>
    >::type
    async(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15)
    {
        return async(launch::all, boost::forward<F>(f),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ));
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
    >
    lcos::unique_future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15
    )
    {
        return bound.async(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ));
    }
}
namespace hpx
{
    
    
    
    
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type)>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(util::bind(
                util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 )), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 )));
        if (detail::has_async_policy(policy))
            p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type)>
    >::type
    async(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 )));
        p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type)>
    >::type
    async(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16)
    {
        return async(launch::all, boost::forward<F>(f),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ));
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
    >
    lcos::unique_future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16
    )
    {
        return bound.async(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ));
    }
}
namespace hpx
{
    
    
    
    
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type)>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(util::bind(
                util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 )), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 )));
        if (detail::has_async_policy(policy))
            p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type)>
    >::type
    async(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 )));
        p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type)>
    >::type
    async(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17)
    {
        return async(launch::all, boost::forward<F>(f),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ));
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17
    >
    lcos::unique_future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17
    )
    {
        return bound.async(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ));
    }
}
namespace hpx
{
    
    
    
    
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type , typename util::decay<A18>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type , typename util::decay<A18>::type)>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type , typename util::decay<A18>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(util::bind(
                util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 )), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 )));
        if (detail::has_async_policy(policy))
            p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type , typename util::decay<A18>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type , typename util::decay<A18>::type)>
    >::type
    async(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type , typename util::decay<A18>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 )));
        p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type , typename util::decay<A18>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type , typename util::decay<A18>::type)>
    >::type
    async(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18)
    {
        return async(launch::all, boost::forward<F>(f),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 ));
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18
    >
    lcos::unique_future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18
    )
    {
        return bound.async(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 ));
    }
}
namespace hpx
{
    
    
    
    
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type , typename util::decay<A18>::type , typename util::decay<A19>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type , typename util::decay<A18>::type , typename util::decay<A19>::type)>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18 , BOOST_FWD_REF(A19) a19)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type , typename util::decay<A18>::type , typename util::decay<A19>::type)
        >::type result_type;
        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(util::bind(
                util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 ) , boost::forward<A19>( a19 )), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 ) , boost::forward<A19>( a19 )));
        if (detail::has_async_policy(policy))
            p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type , typename util::decay<A18>::type , typename util::decay<A19>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type , typename util::decay<A18>::type , typename util::decay<A19>::type)>
    >::type
    async(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18 , BOOST_FWD_REF(A19) a19)
    {
        typedef typename boost::result_of<
            typename util::decay<F>::type
                (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type , typename util::decay<A18>::type , typename util::decay<A19>::type)
        >::type result_type;
        lcos::local::futures_factory<result_type()> p(sched,
            util::bind(util::one_shot(boost::forward<F>(f)),
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 ) , boost::forward<A19>( a19 )));
        p.apply();
        return p.get_future();
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type , typename util::decay<A18>::type , typename util::decay<A19>::type)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<typename util::decay<F>::type
            (typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type , typename util::decay<A15>::type , typename util::decay<A16>::type , typename util::decay<A17>::type , typename util::decay<A18>::type , typename util::decay<A19>::type)>
    >::type
    async(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18 , BOOST_FWD_REF(A19) a19)
    {
        return async(launch::all, boost::forward<F>(f),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 ) , boost::forward<A19>( a19 ));
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19
    >
    lcos::unique_future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18 , BOOST_FWD_REF(A19) a19
    )
    {
        return bound.async(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 ) , boost::forward<A19>( a19 ));
    }
}
