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
    bool apply(threads::executor& sched, BOOST_FWD_REF(F) f)
    {
        sched.add(boost::forward<F>(f), "hpx::apply");
        return false; 
    }
    template <typename F>
    typename boost::enable_if_c<
        traits::detail::is_callable_not_action<F>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , bool
    >::type
    apply(BOOST_FWD_REF(F) f)
    {
        threads::register_thread(boost::forward<F>(f), "hpx::apply");
        return false; 
    }
    
    
    template <typename Action, typename BoundArgs>
    bool apply(hpx::util::detail::bound_action<Action, BoundArgs> const& bound)
    {
        return bound.apply();
    }
}
namespace hpx
{
    
    
    
    template <typename F, typename A0>
    typename boost::enable_if_c<
        traits::detail::is_callable_not_action<F
          , BOOST_FWD_REF(A0)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , bool
    >::type
    apply(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0)
    {
        sched.add(util::bind(util::protect(boost::forward<F>(f)),
            boost::forward<A0>( a0 )), "hpx::apply");
        return false;
    }
    template <typename F, typename A0>
    typename boost::enable_if_c<
        traits::detail::is_callable_not_action<F
          , BOOST_FWD_REF(A0)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , bool
    >::type
    apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0)
    {
        threads::register_thread(util::bind(
            util::protect(boost::forward<F>(f)),
            boost::forward<A0>( a0 )), "hpx::apply");
        return false;
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0
    >
    bool apply(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0
    )
    {
        return bound.apply(boost::forward<A0>( a0 ));
    }
}
namespace hpx
{
    
    
    
    template <typename F, typename A0 , typename A1>
    typename boost::enable_if_c<
        traits::detail::is_callable_not_action<F
          , BOOST_FWD_REF(A0) , BOOST_FWD_REF(A1)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , bool
    >::type
    apply(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1)
    {
        sched.add(util::bind(util::protect(boost::forward<F>(f)),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )), "hpx::apply");
        return false;
    }
    template <typename F, typename A0 , typename A1>
    typename boost::enable_if_c<
        traits::detail::is_callable_not_action<F
          , BOOST_FWD_REF(A0) , BOOST_FWD_REF(A1)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , bool
    >::type
    apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1)
    {
        threads::register_thread(util::bind(
            util::protect(boost::forward<F>(f)),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )), "hpx::apply");
        return false;
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1
    >
    bool apply(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1
    )
    {
        return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
    }
}
namespace hpx
{
    
    
    
    template <typename F, typename A0 , typename A1 , typename A2>
    typename boost::enable_if_c<
        traits::detail::is_callable_not_action<F
          , BOOST_FWD_REF(A0) , BOOST_FWD_REF(A1) , BOOST_FWD_REF(A2)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , bool
    >::type
    apply(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2)
    {
        sched.add(util::bind(util::protect(boost::forward<F>(f)),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )), "hpx::apply");
        return false;
    }
    template <typename F, typename A0 , typename A1 , typename A2>
    typename boost::enable_if_c<
        traits::detail::is_callable_not_action<F
          , BOOST_FWD_REF(A0) , BOOST_FWD_REF(A1) , BOOST_FWD_REF(A2)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , bool
    >::type
    apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2)
    {
        threads::register_thread(util::bind(
            util::protect(boost::forward<F>(f)),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )), "hpx::apply");
        return false;
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1 , typename A2
    >
    bool apply(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2
    )
    {
        return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
    }
}
namespace hpx
{
    
    
    
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3>
    typename boost::enable_if_c<
        traits::detail::is_callable_not_action<F
          , BOOST_FWD_REF(A0) , BOOST_FWD_REF(A1) , BOOST_FWD_REF(A2) , BOOST_FWD_REF(A3)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , bool
    >::type
    apply(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3)
    {
        sched.add(util::bind(util::protect(boost::forward<F>(f)),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )), "hpx::apply");
        return false;
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3>
    typename boost::enable_if_c<
        traits::detail::is_callable_not_action<F
          , BOOST_FWD_REF(A0) , BOOST_FWD_REF(A1) , BOOST_FWD_REF(A2) , BOOST_FWD_REF(A3)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , bool
    >::type
    apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3)
    {
        threads::register_thread(util::bind(
            util::protect(boost::forward<F>(f)),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )), "hpx::apply");
        return false;
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1 , typename A2 , typename A3
    >
    bool apply(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3
    )
    {
        return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
    }
}
namespace hpx
{
    
    
    
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    typename boost::enable_if_c<
        traits::detail::is_callable_not_action<F
          , BOOST_FWD_REF(A0) , BOOST_FWD_REF(A1) , BOOST_FWD_REF(A2) , BOOST_FWD_REF(A3) , BOOST_FWD_REF(A4)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , bool
    >::type
    apply(threads::executor& sched, BOOST_FWD_REF(F) f,
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4)
    {
        sched.add(util::bind(util::protect(boost::forward<F>(f)),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )), "hpx::apply");
        return false;
    }
    template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    typename boost::enable_if_c<
        traits::detail::is_callable_not_action<F
          , BOOST_FWD_REF(A0) , BOOST_FWD_REF(A1) , BOOST_FWD_REF(A2) , BOOST_FWD_REF(A3) , BOOST_FWD_REF(A4)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , bool
    >::type
    apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4)
    {
        threads::register_thread(util::bind(
            util::protect(boost::forward<F>(f)),
            boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )), "hpx::apply");
        return false;
    }
    
    template <
        typename Action, typename BoundArgs
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
    >
    bool apply(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4
    )
    {
        return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
    }
}
