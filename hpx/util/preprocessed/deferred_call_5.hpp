// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


        
namespace hpx { namespace util
{
    template <typename F, typename T0>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename util::decay<T0>::type>
    >
    deferred_call(BOOST_FWD_REF(F) f, BOOST_FWD_REF(T0) t0)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<T0>::type>
        > result_type;
        return result_type(boost::forward<F>(f)
          , util::forward_as_tuple(boost::forward<T0>( t0 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type>
    >
    deferred_call(BOOST_FWD_REF(F) f, BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type>
        > result_type;
        return result_type(boost::forward<F>(f)
          , util::forward_as_tuple(boost::forward<T0>( t0 ) , boost::forward<T1>( t1 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type>
    >
    deferred_call(BOOST_FWD_REF(F) f, BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type>
        > result_type;
        return result_type(boost::forward<F>(f)
          , util::forward_as_tuple(boost::forward<T0>( t0 ) , boost::forward<T1>( t1 ) , boost::forward<T2>( t2 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type>
    >
    deferred_call(BOOST_FWD_REF(F) f, BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type>
        > result_type;
        return result_type(boost::forward<F>(f)
          , util::forward_as_tuple(boost::forward<T0>( t0 ) , boost::forward<T1>( t1 ) , boost::forward<T2>( t2 ) , boost::forward<T3>( t3 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type>
    >
    deferred_call(BOOST_FWD_REF(F) f, BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type>
        > result_type;
        return result_type(boost::forward<F>(f)
          , util::forward_as_tuple(boost::forward<T0>( t0 ) , boost::forward<T1>( t1 ) , boost::forward<T2>( t2 ) , boost::forward<T3>( t3 ) , boost::forward<T4>( t4 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type>
    >
    deferred_call(BOOST_FWD_REF(F) f, BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type>
        > result_type;
        return result_type(boost::forward<F>(f)
          , util::forward_as_tuple(boost::forward<T0>( t0 ) , boost::forward<T1>( t1 ) , boost::forward<T2>( t2 ) , boost::forward<T3>( t3 ) , boost::forward<T4>( t4 ) , boost::forward<T5>( t5 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type>
    >
    deferred_call(BOOST_FWD_REF(F) f, BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5 , BOOST_FWD_REF(T6) t6)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type>
        > result_type;
        return result_type(boost::forward<F>(f)
          , util::forward_as_tuple(boost::forward<T0>( t0 ) , boost::forward<T1>( t1 ) , boost::forward<T2>( t2 ) , boost::forward<T3>( t3 ) , boost::forward<T4>( t4 ) , boost::forward<T5>( t5 ) , boost::forward<T6>( t6 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type>
    >
    deferred_call(BOOST_FWD_REF(F) f, BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5 , BOOST_FWD_REF(T6) t6 , BOOST_FWD_REF(T7) t7)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type>
        > result_type;
        return result_type(boost::forward<F>(f)
          , util::forward_as_tuple(boost::forward<T0>( t0 ) , boost::forward<T1>( t1 ) , boost::forward<T2>( t2 ) , boost::forward<T3>( t3 ) , boost::forward<T4>( t4 ) , boost::forward<T5>( t5 ) , boost::forward<T6>( t6 ) , boost::forward<T7>( t7 )));
    }
}}
