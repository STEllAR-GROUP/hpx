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
    
    
    
    
    
    
    
    
    template <typename F, typename A0> typename boost::enable_if< traits::supports_result_of<F> , bool >::type apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0) { thread t(boost::forward<F>(f), boost::forward<A0>( a0 )); t.detach(); return false; } template <typename F, typename A0 , typename A1> typename boost::enable_if< traits::supports_result_of<F> , bool >::type apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) { thread t(boost::forward<F>(f), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); t.detach(); return false; } template <typename F, typename A0 , typename A1 , typename A2> typename boost::enable_if< traits::supports_result_of<F> , bool >::type apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) { thread t(boost::forward<F>(f), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); t.detach(); return false; } template <typename F, typename A0 , typename A1 , typename A2 , typename A3> typename boost::enable_if< traits::supports_result_of<F> , bool >::type apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) { thread t(boost::forward<F>(f), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); t.detach(); return false; } template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4> typename boost::enable_if< traits::supports_result_of<F> , bool >::type apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) { thread t(boost::forward<F>(f), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); t.detach(); return false; } template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> typename boost::enable_if< traits::supports_result_of<F> , bool >::type apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) { thread t(boost::forward<F>(f), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); t.detach(); return false; } template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> typename boost::enable_if< traits::supports_result_of<F> , bool >::type apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) { thread t(boost::forward<F>(f), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); t.detach(); return false; } template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7> typename boost::enable_if< traits::supports_result_of<F> , bool >::type apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7) { thread t(boost::forward<F>(f), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 )); t.detach(); return false; } template <typename F, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8> typename boost::enable_if< traits::supports_result_of<F> , bool >::type apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8) { thread t(boost::forward<F>(f), boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 )); t.detach(); return false; }
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
    
    