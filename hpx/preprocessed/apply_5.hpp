// Copyright (c) 2007-2012 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx
{
    
    
    template <typename F>
    bool apply(BOOST_FWD_REF(F) f)
    {
        thread t(boost::forward<F>(f));
        t.detach(); 
        return false; 
    }
    
    
    
    
    
    
    
    
    template <typename F, typename A0> typename boost::enable_if< traits::supports_result_of<F> , bool >::type apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0) { thread t(boost::forward<F>(f), boost::forward<A0>( a0 )); t.detach(); return false; } template <typename F, typename A0 , typename A1> typename boost::enable_if< traits::supports_result_of<F> , bool >::type apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) { thread t(boost::forward<F>(f), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); t.detach(); return false; } template <typename F, typename A0 , typename A1 , typename A2> typename boost::enable_if< traits::supports_result_of<F> , bool >::type apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) { thread t(boost::forward<F>(f), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); t.detach(); return false; } template <typename F, typename A0 , typename A1 , typename A2 , typename A3> typename boost::enable_if< traits::supports_result_of<F> , bool >::type apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) { thread t(boost::forward<F>(f), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); t.detach(); return false; } template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4> typename boost::enable_if< traits::supports_result_of<F> , bool >::type apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) { thread t(boost::forward<F>(f), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); t.detach(); return false; } template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> typename boost::enable_if< traits::supports_result_of<F> , bool >::type apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) { thread t(boost::forward<F>(f), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); t.detach(); return false; } template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> typename boost::enable_if< traits::supports_result_of<F> , bool >::type apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) { thread t(boost::forward<F>(f), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); t.detach(); return false; }
}
namespace hpx
{
    
    
    template <
        typename R
      , typename T0
      , typename Arg0
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            hpx::util::detail::bound_function1<
                R
              , T0
              , Arg0
            >
        ))) bound)
    {
        thread t(boost::move(bound));
        t.detach(); 
        return false; 
    }
    
    template < typename R , typename T0 , typename Arg0 , typename A0 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function1< R , T0 , Arg0 >))) bound , BOOST_FWD_REF(A0) a0 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 )); t.detach(); return false; } template < typename R , typename T0 , typename Arg0 , typename A0 , typename A1 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function1< R , T0 , Arg0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); t.detach(); return false; } template < typename R , typename T0 , typename Arg0 , typename A0 , typename A1 , typename A2 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function1< R , T0 , Arg0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); t.detach(); return false; } template < typename R , typename T0 , typename Arg0 , typename A0 , typename A1 , typename A2 , typename A3 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function1< R , T0 , Arg0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); t.detach(); return false; } template < typename R , typename T0 , typename Arg0 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function1< R , T0 , Arg0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); t.detach(); return false; } template < typename R , typename T0 , typename Arg0 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function1< R , T0 , Arg0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); t.detach(); return false; } template < typename R , typename T0 , typename Arg0 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function1< R , T0 , Arg0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); t.detach(); return false; }
    
    
    template <
        typename R
      , typename C
      
            
      , typename Arg0
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            hpx::util::detail::bound_member_function1<
                R
              , C
              
                    
              , Arg0
            >
        ))) bound)
    {
        thread t(boost::move(bound));
        t.detach(); 
        return false; 
    }
    
    template < typename R , typename C , typename Arg0 , typename A0 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function1< R , C , Arg0 >))) bound , BOOST_FWD_REF(A0) a0 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 )); t.detach(); return false; } template < typename R , typename C , typename Arg0 , typename A0 , typename A1 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function1< R , C , Arg0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); t.detach(); return false; } template < typename R , typename C , typename Arg0 , typename A0 , typename A1 , typename A2 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function1< R , C , Arg0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); t.detach(); return false; } template < typename R , typename C , typename Arg0 , typename A0 , typename A1 , typename A2 , typename A3 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function1< R , C , Arg0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); t.detach(); return false; } template < typename R , typename C , typename Arg0 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function1< R , C , Arg0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); t.detach(); return false; } template < typename R , typename C , typename Arg0 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function1< R , C , Arg0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); t.detach(); return false; } template < typename R , typename C , typename Arg0 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function1< R , C , Arg0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); t.detach(); return false; }
    
    
    template <
        typename F
      , typename Arg0
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            hpx::util::detail::bound_functor1<
                F
              , Arg0
            >
        ))) bound)
    {
        thread t(boost::move(bound));
        t.detach(); 
        return false; 
    }
    
    template < typename F , typename Arg0 , typename A0 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor1< F , Arg0 >))) bound , BOOST_FWD_REF(A0) a0 ) { thread t(boost::move(bound), boost::forward<A0>( a0 )); t.detach(); return false; } template < typename F , typename Arg0 , typename A0 , typename A1 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor1< F , Arg0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); t.detach(); return false; } template < typename F , typename Arg0 , typename A0 , typename A1 , typename A2 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor1< F , Arg0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); t.detach(); return false; } template < typename F , typename Arg0 , typename A0 , typename A1 , typename A2 , typename A3 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor1< F , Arg0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); t.detach(); return false; } template < typename F , typename Arg0 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor1< F , Arg0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); t.detach(); return false; } template < typename F , typename Arg0 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor1< F , Arg0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); t.detach(); return false; } template < typename F , typename Arg0 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor1< F , Arg0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); t.detach(); return false; }
    
    
    template <
        typename Action
      , typename T0
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            hpx::util::detail::bound_action1<
                Action
              , T0
            >))) bound)
    {
        return bound.apply();
    }
    
    template < typename Action , typename T0 , typename A0 , typename A0 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action1< Action , T0 >))) bound , BOOST_FWD_REF(A0) a0 ) { return bound.apply(boost::forward<A0>( a0 )); } template < typename Action , typename T0 , typename A0 , typename A1 , typename A0 , typename A1 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action1< Action , T0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); } template < typename Action , typename T0 , typename A0 , typename A1 , typename A2 , typename A0 , typename A1 , typename A2 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action1< Action , T0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); } template < typename Action , typename T0 , typename A0 , typename A1 , typename A2 , typename A3 , typename A0 , typename A1 , typename A2 , typename A3 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action1< Action , T0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); } template < typename Action , typename T0 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action1< Action , T0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); } template < typename Action , typename T0 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action1< Action , T0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); } template < typename Action , typename T0 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action1< Action , T0 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); }
}
namespace hpx
{
    
    
    template <
        typename R
      , typename T0 , typename T1
      , typename Arg0 , typename Arg1
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            hpx::util::detail::bound_function2<
                R
              , T0 , T1
              , Arg0 , Arg1
            >
        ))) bound)
    {
        thread t(boost::move(bound));
        t.detach(); 
        return false; 
    }
    
    template < typename R , typename T0 , typename T1 , typename Arg0 , typename Arg1 , typename A0 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function2< R , T0 , T1 , Arg0 , Arg1 >))) bound , BOOST_FWD_REF(A0) a0 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename Arg0 , typename Arg1 , typename A0 , typename A1 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function2< R , T0 , T1 , Arg0 , Arg1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename Arg0 , typename Arg1 , typename A0 , typename A1 , typename A2 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function2< R , T0 , T1 , Arg0 , Arg1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename Arg0 , typename Arg1 , typename A0 , typename A1 , typename A2 , typename A3 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function2< R , T0 , T1 , Arg0 , Arg1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename Arg0 , typename Arg1 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function2< R , T0 , T1 , Arg0 , Arg1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename Arg0 , typename Arg1 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function2< R , T0 , T1 , Arg0 , Arg1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename Arg0 , typename Arg1 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function2< R , T0 , T1 , Arg0 , Arg1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); t.detach(); return false; }
    
    
    template <
        typename R
      , typename C
      ,
            typename T0
      , typename Arg0 , typename Arg1
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            hpx::util::detail::bound_member_function2<
                R
              , C
              ,
                    T0
              , Arg0 , Arg1
            >
        ))) bound)
    {
        thread t(boost::move(bound));
        t.detach(); 
        return false; 
    }
    
    template < typename R , typename C , typename T0 , typename Arg0 , typename Arg1 , typename A0 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function2< R , C , T0 , Arg0 , Arg1 >))) bound , BOOST_FWD_REF(A0) a0 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename Arg0 , typename Arg1 , typename A0 , typename A1 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function2< R , C , T0 , Arg0 , Arg1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename Arg0 , typename Arg1 , typename A0 , typename A1 , typename A2 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function2< R , C , T0 , Arg0 , Arg1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename Arg0 , typename Arg1 , typename A0 , typename A1 , typename A2 , typename A3 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function2< R , C , T0 , Arg0 , Arg1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename Arg0 , typename Arg1 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function2< R , C , T0 , Arg0 , Arg1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename Arg0 , typename Arg1 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function2< R , C , T0 , Arg0 , Arg1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename Arg0 , typename Arg1 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function2< R , C , T0 , Arg0 , Arg1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); t.detach(); return false; }
    
    
    template <
        typename F
      , typename Arg0 , typename Arg1
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            hpx::util::detail::bound_functor2<
                F
              , Arg0 , Arg1
            >
        ))) bound)
    {
        thread t(boost::move(bound));
        t.detach(); 
        return false; 
    }
    
    template < typename F , typename Arg0 , typename Arg1 , typename A0 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor2< F , Arg0 , Arg1 >))) bound , BOOST_FWD_REF(A0) a0 ) { thread t(boost::move(bound), boost::forward<A0>( a0 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename A0 , typename A1 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor2< F , Arg0 , Arg1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename A0 , typename A1 , typename A2 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor2< F , Arg0 , Arg1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename A0 , typename A1 , typename A2 , typename A3 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor2< F , Arg0 , Arg1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor2< F , Arg0 , Arg1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor2< F , Arg0 , Arg1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor2< F , Arg0 , Arg1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); t.detach(); return false; }
    
    
    template <
        typename Action
      , typename T0 , typename T1
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            hpx::util::detail::bound_action2<
                Action
              , T0 , T1
            >))) bound)
    {
        return bound.apply();
    }
    
    template < typename Action , typename T0 , typename T1 , typename A0 , typename A0 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action2< Action , T0 , T1 >))) bound , BOOST_FWD_REF(A0) a0 ) { return bound.apply(boost::forward<A0>( a0 )); } template < typename Action , typename T0 , typename T1 , typename A0 , typename A1 , typename A0 , typename A1 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action2< Action , T0 , T1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); } template < typename Action , typename T0 , typename T1 , typename A0 , typename A1 , typename A2 , typename A0 , typename A1 , typename A2 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action2< Action , T0 , T1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); } template < typename Action , typename T0 , typename T1 , typename A0 , typename A1 , typename A2 , typename A3 , typename A0 , typename A1 , typename A2 , typename A3 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action2< Action , T0 , T1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); } template < typename Action , typename T0 , typename T1 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action2< Action , T0 , T1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); } template < typename Action , typename T0 , typename T1 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action2< Action , T0 , T1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); } template < typename Action , typename T0 , typename T1 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action2< Action , T0 , T1 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); }
}
namespace hpx
{
    
    
    template <
        typename R
      , typename T0 , typename T1 , typename T2
      , typename Arg0 , typename Arg1 , typename Arg2
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            hpx::util::detail::bound_function3<
                R
              , T0 , T1 , T2
              , Arg0 , Arg1 , Arg2
            >
        ))) bound)
    {
        thread t(boost::move(bound));
        t.detach(); 
        return false; 
    }
    
    template < typename R , typename T0 , typename T1 , typename T2 , typename Arg0 , typename Arg1 , typename Arg2 , typename A0 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function3< R , T0 , T1 , T2 , Arg0 , Arg1 , Arg2 >))) bound , BOOST_FWD_REF(A0) a0 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename T2 , typename Arg0 , typename Arg1 , typename Arg2 , typename A0 , typename A1 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function3< R , T0 , T1 , T2 , Arg0 , Arg1 , Arg2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename T2 , typename Arg0 , typename Arg1 , typename Arg2 , typename A0 , typename A1 , typename A2 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function3< R , T0 , T1 , T2 , Arg0 , Arg1 , Arg2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename T2 , typename Arg0 , typename Arg1 , typename Arg2 , typename A0 , typename A1 , typename A2 , typename A3 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function3< R , T0 , T1 , T2 , Arg0 , Arg1 , Arg2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename T2 , typename Arg0 , typename Arg1 , typename Arg2 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function3< R , T0 , T1 , T2 , Arg0 , Arg1 , Arg2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename T2 , typename Arg0 , typename Arg1 , typename Arg2 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function3< R , T0 , T1 , T2 , Arg0 , Arg1 , Arg2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename T2 , typename Arg0 , typename Arg1 , typename Arg2 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function3< R , T0 , T1 , T2 , Arg0 , Arg1 , Arg2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); t.detach(); return false; }
    
    
    template <
        typename R
      , typename C
      ,
            typename T0 , typename T1
      , typename Arg0 , typename Arg1 , typename Arg2
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            hpx::util::detail::bound_member_function3<
                R
              , C
              ,
                    T0 , T1
              , Arg0 , Arg1 , Arg2
            >
        ))) bound)
    {
        thread t(boost::move(bound));
        t.detach(); 
        return false; 
    }
    
    template < typename R , typename C , typename T0 , typename T1 , typename Arg0 , typename Arg1 , typename Arg2 , typename A0 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function3< R , C , T0 , T1 , Arg0 , Arg1 , Arg2 >))) bound , BOOST_FWD_REF(A0) a0 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename T1 , typename Arg0 , typename Arg1 , typename Arg2 , typename A0 , typename A1 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function3< R , C , T0 , T1 , Arg0 , Arg1 , Arg2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename T1 , typename Arg0 , typename Arg1 , typename Arg2 , typename A0 , typename A1 , typename A2 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function3< R , C , T0 , T1 , Arg0 , Arg1 , Arg2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename T1 , typename Arg0 , typename Arg1 , typename Arg2 , typename A0 , typename A1 , typename A2 , typename A3 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function3< R , C , T0 , T1 , Arg0 , Arg1 , Arg2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename T1 , typename Arg0 , typename Arg1 , typename Arg2 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function3< R , C , T0 , T1 , Arg0 , Arg1 , Arg2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename T1 , typename Arg0 , typename Arg1 , typename Arg2 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function3< R , C , T0 , T1 , Arg0 , Arg1 , Arg2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename T1 , typename Arg0 , typename Arg1 , typename Arg2 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function3< R , C , T0 , T1 , Arg0 , Arg1 , Arg2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); t.detach(); return false; }
    
    
    template <
        typename F
      , typename Arg0 , typename Arg1 , typename Arg2
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            hpx::util::detail::bound_functor3<
                F
              , Arg0 , Arg1 , Arg2
            >
        ))) bound)
    {
        thread t(boost::move(bound));
        t.detach(); 
        return false; 
    }
    
    template < typename F , typename Arg0 , typename Arg1 , typename Arg2 , typename A0 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor3< F , Arg0 , Arg1 , Arg2 >))) bound , BOOST_FWD_REF(A0) a0 ) { thread t(boost::move(bound), boost::forward<A0>( a0 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename Arg2 , typename A0 , typename A1 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor3< F , Arg0 , Arg1 , Arg2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename Arg2 , typename A0 , typename A1 , typename A2 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor3< F , Arg0 , Arg1 , Arg2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename Arg2 , typename A0 , typename A1 , typename A2 , typename A3 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor3< F , Arg0 , Arg1 , Arg2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename Arg2 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor3< F , Arg0 , Arg1 , Arg2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename Arg2 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor3< F , Arg0 , Arg1 , Arg2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename Arg2 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor3< F , Arg0 , Arg1 , Arg2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); t.detach(); return false; }
    
    
    template <
        typename Action
      , typename T0 , typename T1 , typename T2
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            hpx::util::detail::bound_action3<
                Action
              , T0 , T1 , T2
            >))) bound)
    {
        return bound.apply();
    }
    
    template < typename Action , typename T0 , typename T1 , typename T2 , typename A0 , typename A0 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action3< Action , T0 , T1 , T2 >))) bound , BOOST_FWD_REF(A0) a0 ) { return bound.apply(boost::forward<A0>( a0 )); } template < typename Action , typename T0 , typename T1 , typename T2 , typename A0 , typename A1 , typename A0 , typename A1 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action3< Action , T0 , T1 , T2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); } template < typename Action , typename T0 , typename T1 , typename T2 , typename A0 , typename A1 , typename A2 , typename A0 , typename A1 , typename A2 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action3< Action , T0 , T1 , T2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); } template < typename Action , typename T0 , typename T1 , typename T2 , typename A0 , typename A1 , typename A2 , typename A3 , typename A0 , typename A1 , typename A2 , typename A3 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action3< Action , T0 , T1 , T2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); } template < typename Action , typename T0 , typename T1 , typename T2 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action3< Action , T0 , T1 , T2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); } template < typename Action , typename T0 , typename T1 , typename T2 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action3< Action , T0 , T1 , T2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); } template < typename Action , typename T0 , typename T1 , typename T2 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action3< Action , T0 , T1 , T2 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); }
}
namespace hpx
{
    
    
    template <
        typename R
      , typename T0 , typename T1 , typename T2 , typename T3
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            hpx::util::detail::bound_function4<
                R
              , T0 , T1 , T2 , T3
              , Arg0 , Arg1 , Arg2 , Arg3
            >
        ))) bound)
    {
        thread t(boost::move(bound));
        t.detach(); 
        return false; 
    }
    
    template < typename R , typename T0 , typename T1 , typename T2 , typename T3 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename A0 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function4< R , T0 , T1 , T2 , T3 , Arg0 , Arg1 , Arg2 , Arg3 >))) bound , BOOST_FWD_REF(A0) a0 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename T2 , typename T3 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename A0 , typename A1 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function4< R , T0 , T1 , T2 , T3 , Arg0 , Arg1 , Arg2 , Arg3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename T2 , typename T3 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename A0 , typename A1 , typename A2 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function4< R , T0 , T1 , T2 , T3 , Arg0 , Arg1 , Arg2 , Arg3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename T2 , typename T3 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename A0 , typename A1 , typename A2 , typename A3 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function4< R , T0 , T1 , T2 , T3 , Arg0 , Arg1 , Arg2 , Arg3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename T2 , typename T3 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function4< R , T0 , T1 , T2 , T3 , Arg0 , Arg1 , Arg2 , Arg3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename T2 , typename T3 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function4< R , T0 , T1 , T2 , T3 , Arg0 , Arg1 , Arg2 , Arg3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename T2 , typename T3 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function4< R , T0 , T1 , T2 , T3 , Arg0 , Arg1 , Arg2 , Arg3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); t.detach(); return false; }
    
    
    template <
        typename R
      , typename C
      ,
            typename T0 , typename T1 , typename T2
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            hpx::util::detail::bound_member_function4<
                R
              , C
              ,
                    T0 , T1 , T2
              , Arg0 , Arg1 , Arg2 , Arg3
            >
        ))) bound)
    {
        thread t(boost::move(bound));
        t.detach(); 
        return false; 
    }
    
    template < typename R , typename C , typename T0 , typename T1 , typename T2 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename A0 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function4< R , C , T0 , T1 , T2 , Arg0 , Arg1 , Arg2 , Arg3 >))) bound , BOOST_FWD_REF(A0) a0 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename T1 , typename T2 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename A0 , typename A1 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function4< R , C , T0 , T1 , T2 , Arg0 , Arg1 , Arg2 , Arg3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename T1 , typename T2 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename A0 , typename A1 , typename A2 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function4< R , C , T0 , T1 , T2 , Arg0 , Arg1 , Arg2 , Arg3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename T1 , typename T2 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename A0 , typename A1 , typename A2 , typename A3 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function4< R , C , T0 , T1 , T2 , Arg0 , Arg1 , Arg2 , Arg3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename T1 , typename T2 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function4< R , C , T0 , T1 , T2 , Arg0 , Arg1 , Arg2 , Arg3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename T1 , typename T2 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function4< R , C , T0 , T1 , T2 , Arg0 , Arg1 , Arg2 , Arg3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename T1 , typename T2 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function4< R , C , T0 , T1 , T2 , Arg0 , Arg1 , Arg2 , Arg3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); t.detach(); return false; }
    
    
    template <
        typename F
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            hpx::util::detail::bound_functor4<
                F
              , Arg0 , Arg1 , Arg2 , Arg3
            >
        ))) bound)
    {
        thread t(boost::move(bound));
        t.detach(); 
        return false; 
    }
    
    template < typename F , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename A0 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor4< F , Arg0 , Arg1 , Arg2 , Arg3 >))) bound , BOOST_FWD_REF(A0) a0 ) { thread t(boost::move(bound), boost::forward<A0>( a0 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename A0 , typename A1 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor4< F , Arg0 , Arg1 , Arg2 , Arg3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename A0 , typename A1 , typename A2 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor4< F , Arg0 , Arg1 , Arg2 , Arg3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename A0 , typename A1 , typename A2 , typename A3 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor4< F , Arg0 , Arg1 , Arg2 , Arg3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor4< F , Arg0 , Arg1 , Arg2 , Arg3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor4< F , Arg0 , Arg1 , Arg2 , Arg3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor4< F , Arg0 , Arg1 , Arg2 , Arg3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); t.detach(); return false; }
    
    
    template <
        typename Action
      , typename T0 , typename T1 , typename T2 , typename T3
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            hpx::util::detail::bound_action4<
                Action
              , T0 , T1 , T2 , T3
            >))) bound)
    {
        return bound.apply();
    }
    
    template < typename Action , typename T0 , typename T1 , typename T2 , typename T3 , typename A0 , typename A0 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action4< Action , T0 , T1 , T2 , T3 >))) bound , BOOST_FWD_REF(A0) a0 ) { return bound.apply(boost::forward<A0>( a0 )); } template < typename Action , typename T0 , typename T1 , typename T2 , typename T3 , typename A0 , typename A1 , typename A0 , typename A1 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action4< Action , T0 , T1 , T2 , T3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); } template < typename Action , typename T0 , typename T1 , typename T2 , typename T3 , typename A0 , typename A1 , typename A2 , typename A0 , typename A1 , typename A2 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action4< Action , T0 , T1 , T2 , T3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); } template < typename Action , typename T0 , typename T1 , typename T2 , typename T3 , typename A0 , typename A1 , typename A2 , typename A3 , typename A0 , typename A1 , typename A2 , typename A3 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action4< Action , T0 , T1 , T2 , T3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); } template < typename Action , typename T0 , typename T1 , typename T2 , typename T3 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action4< Action , T0 , T1 , T2 , T3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); } template < typename Action , typename T0 , typename T1 , typename T2 , typename T3 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action4< Action , T0 , T1 , T2 , T3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); } template < typename Action , typename T0 , typename T1 , typename T2 , typename T3 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action4< Action , T0 , T1 , T2 , T3 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); }
}
namespace hpx
{
    
    
    template <
        typename R
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            hpx::util::detail::bound_function5<
                R
              , T0 , T1 , T2 , T3 , T4
              , Arg0 , Arg1 , Arg2 , Arg3 , Arg4
            >
        ))) bound)
    {
        thread t(boost::move(bound));
        t.detach(); 
        return false; 
    }
    
    template < typename R , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename A0 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function5< R , T0 , T1 , T2 , T3 , T4 , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 >))) bound , BOOST_FWD_REF(A0) a0 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename A0 , typename A1 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function5< R , T0 , T1 , T2 , T3 , T4 , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename A0 , typename A1 , typename A2 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function5< R , T0 , T1 , T2 , T3 , T4 , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename A0 , typename A1 , typename A2 , typename A3 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function5< R , T0 , T1 , T2 , T3 , T4 , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function5< R , T0 , T1 , T2 , T3 , T4 , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function5< R , T0 , T1 , T2 , T3 , T4 , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); t.detach(); return false; } template < typename R , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_function5< R , T0 , T1 , T2 , T3 , T4 , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); t.detach(); return false; }
    
    
    template <
        typename R
      , typename C
      ,
            typename T0 , typename T1 , typename T2 , typename T3
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            hpx::util::detail::bound_member_function5<
                R
              , C
              ,
                    T0 , T1 , T2 , T3
              , Arg0 , Arg1 , Arg2 , Arg3 , Arg4
            >
        ))) bound)
    {
        thread t(boost::move(bound));
        t.detach(); 
        return false; 
    }
    
    template < typename R , typename C , typename T0 , typename T1 , typename T2 , typename T3 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename A0 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function5< R , C , T0 , T1 , T2 , T3 , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 >))) bound , BOOST_FWD_REF(A0) a0 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename T1 , typename T2 , typename T3 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename A0 , typename A1 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function5< R , C , T0 , T1 , T2 , T3 , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename T1 , typename T2 , typename T3 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename A0 , typename A1 , typename A2 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function5< R , C , T0 , T1 , T2 , T3 , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename T1 , typename T2 , typename T3 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename A0 , typename A1 , typename A2 , typename A3 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function5< R , C , T0 , T1 , T2 , T3 , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename T1 , typename T2 , typename T3 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function5< R , C , T0 , T1 , T2 , T3 , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename T1 , typename T2 , typename T3 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function5< R , C , T0 , T1 , T2 , T3 , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); t.detach(); return false; } template < typename R , typename C , typename T0 , typename T1 , typename T2 , typename T3 , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_member_function5< R , C , T0 , T1 , T2 , T3 , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 ) { thread t(boost::move(bound) , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); t.detach(); return false; }
    
    
    template <
        typename F
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            hpx::util::detail::bound_functor5<
                F
              , Arg0 , Arg1 , Arg2 , Arg3 , Arg4
            >
        ))) bound)
    {
        thread t(boost::move(bound));
        t.detach(); 
        return false; 
    }
    
    template < typename F , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename A0 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor5< F , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 >))) bound , BOOST_FWD_REF(A0) a0 ) { thread t(boost::move(bound), boost::forward<A0>( a0 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename A0 , typename A1 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor5< F , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename A0 , typename A1 , typename A2 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor5< F , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename A0 , typename A1 , typename A2 , typename A3 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor5< F , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor5< F , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor5< F , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); t.detach(); return false; } template < typename F , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_functor5< F , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 ) { thread t(boost::move(bound), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); t.detach(); return false; }
    
    
    template <
        typename Action
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            hpx::util::detail::bound_action5<
                Action
              , T0 , T1 , T2 , T3 , T4
            >))) bound)
    {
        return bound.apply();
    }
    
    template < typename Action , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename A0 , typename A0 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action5< Action , T0 , T1 , T2 , T3 , T4 >))) bound , BOOST_FWD_REF(A0) a0 ) { return bound.apply(boost::forward<A0>( a0 )); } template < typename Action , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename A0 , typename A1 , typename A0 , typename A1 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action5< Action , T0 , T1 , T2 , T3 , T4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); } template < typename Action , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename A0 , typename A1 , typename A2 , typename A0 , typename A1 , typename A2 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action5< Action , T0 , T1 , T2 , T3 , T4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); } template < typename Action , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename A0 , typename A1 , typename A2 , typename A3 , typename A0 , typename A1 , typename A2 , typename A3 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action5< Action , T0 , T1 , T2 , T3 , T4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); } template < typename Action , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action5< Action , T0 , T1 , T2 , T3 , T4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); } template < typename Action , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action5< Action , T0 , T1 , T2 , T3 , T4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); } template < typename Action , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 > bool apply( BOOST_RV_REF(HPX_UTIL_STRIP(( hpx::util::detail::bound_action5< Action , T0 , T1 , T2 , T3 , T4 >))) bound , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 ) { return bound.apply(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); }
}
