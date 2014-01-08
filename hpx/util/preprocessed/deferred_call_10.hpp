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
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type>
    >
    deferred_call(BOOST_FWD_REF(F) f, BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5 , BOOST_FWD_REF(T6) t6 , BOOST_FWD_REF(T7) t7 , BOOST_FWD_REF(T8) t8)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type>
        > result_type;
        return result_type(boost::forward<F>(f)
          , util::forward_as_tuple(boost::forward<T0>( t0 ) , boost::forward<T1>( t1 ) , boost::forward<T2>( t2 ) , boost::forward<T3>( t3 ) , boost::forward<T4>( t4 ) , boost::forward<T5>( t5 ) , boost::forward<T6>( t6 ) , boost::forward<T7>( t7 ) , boost::forward<T8>( t8 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type>
    >
    deferred_call(BOOST_FWD_REF(F) f, BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5 , BOOST_FWD_REF(T6) t6 , BOOST_FWD_REF(T7) t7 , BOOST_FWD_REF(T8) t8 , BOOST_FWD_REF(T9) t9)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type>
        > result_type;
        return result_type(boost::forward<F>(f)
          , util::forward_as_tuple(boost::forward<T0>( t0 ) , boost::forward<T1>( t1 ) , boost::forward<T2>( t2 ) , boost::forward<T3>( t3 ) , boost::forward<T4>( t4 ) , boost::forward<T5>( t5 ) , boost::forward<T6>( t6 ) , boost::forward<T7>( t7 ) , boost::forward<T8>( t8 ) , boost::forward<T9>( t9 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type>
    >
    deferred_call(BOOST_FWD_REF(F) f, BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5 , BOOST_FWD_REF(T6) t6 , BOOST_FWD_REF(T7) t7 , BOOST_FWD_REF(T8) t8 , BOOST_FWD_REF(T9) t9 , BOOST_FWD_REF(T10) t10)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type>
        > result_type;
        return result_type(boost::forward<F>(f)
          , util::forward_as_tuple(boost::forward<T0>( t0 ) , boost::forward<T1>( t1 ) , boost::forward<T2>( t2 ) , boost::forward<T3>( t3 ) , boost::forward<T4>( t4 ) , boost::forward<T5>( t5 ) , boost::forward<T6>( t6 ) , boost::forward<T7>( t7 ) , boost::forward<T8>( t8 ) , boost::forward<T9>( t9 ) , boost::forward<T10>( t10 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type>
    >
    deferred_call(BOOST_FWD_REF(F) f, BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5 , BOOST_FWD_REF(T6) t6 , BOOST_FWD_REF(T7) t7 , BOOST_FWD_REF(T8) t8 , BOOST_FWD_REF(T9) t9 , BOOST_FWD_REF(T10) t10 , BOOST_FWD_REF(T11) t11)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type>
        > result_type;
        return result_type(boost::forward<F>(f)
          , util::forward_as_tuple(boost::forward<T0>( t0 ) , boost::forward<T1>( t1 ) , boost::forward<T2>( t2 ) , boost::forward<T3>( t3 ) , boost::forward<T4>( t4 ) , boost::forward<T5>( t5 ) , boost::forward<T6>( t6 ) , boost::forward<T7>( t7 ) , boost::forward<T8>( t8 ) , boost::forward<T9>( t9 ) , boost::forward<T10>( t10 ) , boost::forward<T11>( t11 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type>
    >
    deferred_call(BOOST_FWD_REF(F) f, BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5 , BOOST_FWD_REF(T6) t6 , BOOST_FWD_REF(T7) t7 , BOOST_FWD_REF(T8) t8 , BOOST_FWD_REF(T9) t9 , BOOST_FWD_REF(T10) t10 , BOOST_FWD_REF(T11) t11 , BOOST_FWD_REF(T12) t12)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type>
        > result_type;
        return result_type(boost::forward<F>(f)
          , util::forward_as_tuple(boost::forward<T0>( t0 ) , boost::forward<T1>( t1 ) , boost::forward<T2>( t2 ) , boost::forward<T3>( t3 ) , boost::forward<T4>( t4 ) , boost::forward<T5>( t5 ) , boost::forward<T6>( t6 ) , boost::forward<T7>( t7 ) , boost::forward<T8>( t8 ) , boost::forward<T9>( t9 ) , boost::forward<T10>( t10 ) , boost::forward<T11>( t11 ) , boost::forward<T12>( t12 )));
    }
}}
