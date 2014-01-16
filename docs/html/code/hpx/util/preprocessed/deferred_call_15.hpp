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
    struct deferred_call_result_of<F(T0)>
      : util::result_of<typename util::decay<F>::type(
            typename detail::decay_unwrap<T0>::type)>
    {};
    template <typename F, typename T0>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename detail::decay_unwrap<T0>::type>
    >
    deferred_call(F && f, T0 && t0)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename detail::decay_unwrap<T0>::type>
        > result_type;
        return result_type(std::forward<F>(f)
          , util::forward_as_tuple(std::forward<T0>( t0 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1>
    struct deferred_call_result_of<F(T0 , T1)>
      : util::result_of<typename util::decay<F>::type(
            typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type)>
    {};
    template <typename F, typename T0 , typename T1>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type>
    >
    deferred_call(F && f, T0 && t0 , T1 && t1)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type>
        > result_type;
        return result_type(std::forward<F>(f)
          , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2>
    struct deferred_call_result_of<F(T0 , T1 , T2)>
      : util::result_of<typename util::decay<F>::type(
            typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type)>
    {};
    template <typename F, typename T0 , typename T1 , typename T2>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type>
    >
    deferred_call(F && f, T0 && t0 , T1 && t1 , T2 && t2)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type>
        > result_type;
        return result_type(std::forward<F>(f)
          , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3>
    struct deferred_call_result_of<F(T0 , T1 , T2 , T3)>
      : util::result_of<typename util::decay<F>::type(
            typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type)>
    {};
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type>
    >
    deferred_call(F && f, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type>
        > result_type;
        return result_type(std::forward<F>(f)
          , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    struct deferred_call_result_of<F(T0 , T1 , T2 , T3 , T4)>
      : util::result_of<typename util::decay<F>::type(
            typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type)>
    {};
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type>
    >
    deferred_call(F && f, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type>
        > result_type;
        return result_type(std::forward<F>(f)
          , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    struct deferred_call_result_of<F(T0 , T1 , T2 , T3 , T4 , T5)>
      : util::result_of<typename util::decay<F>::type(
            typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type)>
    {};
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type>
    >
    deferred_call(F && f, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type>
        > result_type;
        return result_type(std::forward<F>(f)
          , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    struct deferred_call_result_of<F(T0 , T1 , T2 , T3 , T4 , T5 , T6)>
      : util::result_of<typename util::decay<F>::type(
            typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type)>
    {};
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type>
    >
    deferred_call(F && f, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type>
        > result_type;
        return result_type(std::forward<F>(f)
          , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    struct deferred_call_result_of<F(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7)>
      : util::result_of<typename util::decay<F>::type(
            typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type)>
    {};
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type>
    >
    deferred_call(F && f, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type>
        > result_type;
        return result_type(std::forward<F>(f)
          , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    struct deferred_call_result_of<F(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8)>
      : util::result_of<typename util::decay<F>::type(
            typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type)>
    {};
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type>
    >
    deferred_call(F && f, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type>
        > result_type;
        return result_type(std::forward<F>(f)
          , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    struct deferred_call_result_of<F(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9)>
      : util::result_of<typename util::decay<F>::type(
            typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type)>
    {};
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type>
    >
    deferred_call(F && f, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type>
        > result_type;
        return result_type(std::forward<F>(f)
          , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    struct deferred_call_result_of<F(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10)>
      : util::result_of<typename util::decay<F>::type(
            typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type)>
    {};
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type>
    >
    deferred_call(F && f, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type>
        > result_type;
        return result_type(std::forward<F>(f)
          , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    struct deferred_call_result_of<F(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11)>
      : util::result_of<typename util::decay<F>::type(
            typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type , typename detail::decay_unwrap<T11>::type)>
    {};
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type , typename detail::decay_unwrap<T11>::type>
    >
    deferred_call(F && f, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type , typename detail::decay_unwrap<T11>::type>
        > result_type;
        return result_type(std::forward<F>(f)
          , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    struct deferred_call_result_of<F(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12)>
      : util::result_of<typename util::decay<F>::type(
            typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type , typename detail::decay_unwrap<T11>::type , typename detail::decay_unwrap<T12>::type)>
    {};
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type , typename detail::decay_unwrap<T11>::type , typename detail::decay_unwrap<T12>::type>
    >
    deferred_call(F && f, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type , typename detail::decay_unwrap<T11>::type , typename detail::decay_unwrap<T12>::type>
        > result_type;
        return result_type(std::forward<F>(f)
          , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    struct deferred_call_result_of<F(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13)>
      : util::result_of<typename util::decay<F>::type(
            typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type , typename detail::decay_unwrap<T11>::type , typename detail::decay_unwrap<T12>::type , typename detail::decay_unwrap<T13>::type)>
    {};
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type , typename detail::decay_unwrap<T11>::type , typename detail::decay_unwrap<T12>::type , typename detail::decay_unwrap<T13>::type>
    >
    deferred_call(F && f, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type , typename detail::decay_unwrap<T11>::type , typename detail::decay_unwrap<T12>::type , typename detail::decay_unwrap<T13>::type>
        > result_type;
        return result_type(std::forward<F>(f)
          , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    struct deferred_call_result_of<F(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14)>
      : util::result_of<typename util::decay<F>::type(
            typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type , typename detail::decay_unwrap<T11>::type , typename detail::decay_unwrap<T12>::type , typename detail::decay_unwrap<T13>::type , typename detail::decay_unwrap<T14>::type)>
    {};
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type , typename detail::decay_unwrap<T11>::type , typename detail::decay_unwrap<T12>::type , typename detail::decay_unwrap<T13>::type , typename detail::decay_unwrap<T14>::type>
    >
    deferred_call(F && f, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type , typename detail::decay_unwrap<T11>::type , typename detail::decay_unwrap<T12>::type , typename detail::decay_unwrap<T13>::type , typename detail::decay_unwrap<T14>::type>
        > result_type;
        return result_type(std::forward<F>(f)
          , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ) , std::forward<T14>( t14 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
    struct deferred_call_result_of<F(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15)>
      : util::result_of<typename util::decay<F>::type(
            typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type , typename detail::decay_unwrap<T11>::type , typename detail::decay_unwrap<T12>::type , typename detail::decay_unwrap<T13>::type , typename detail::decay_unwrap<T14>::type , typename detail::decay_unwrap<T15>::type)>
    {};
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type , typename detail::decay_unwrap<T11>::type , typename detail::decay_unwrap<T12>::type , typename detail::decay_unwrap<T13>::type , typename detail::decay_unwrap<T14>::type , typename detail::decay_unwrap<T15>::type>
    >
    deferred_call(F && f, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14 , T15 && t15)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type , typename detail::decay_unwrap<T11>::type , typename detail::decay_unwrap<T12>::type , typename detail::decay_unwrap<T13>::type , typename detail::decay_unwrap<T14>::type , typename detail::decay_unwrap<T15>::type>
        > result_type;
        return result_type(std::forward<F>(f)
          , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ) , std::forward<T14>( t14 ) , std::forward<T15>( t15 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
    struct deferred_call_result_of<F(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16)>
      : util::result_of<typename util::decay<F>::type(
            typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type , typename detail::decay_unwrap<T11>::type , typename detail::decay_unwrap<T12>::type , typename detail::decay_unwrap<T13>::type , typename detail::decay_unwrap<T14>::type , typename detail::decay_unwrap<T15>::type , typename detail::decay_unwrap<T16>::type)>
    {};
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type , typename detail::decay_unwrap<T11>::type , typename detail::decay_unwrap<T12>::type , typename detail::decay_unwrap<T13>::type , typename detail::decay_unwrap<T14>::type , typename detail::decay_unwrap<T15>::type , typename detail::decay_unwrap<T16>::type>
    >
    deferred_call(F && f, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14 , T15 && t15 , T16 && t16)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type , typename detail::decay_unwrap<T11>::type , typename detail::decay_unwrap<T12>::type , typename detail::decay_unwrap<T13>::type , typename detail::decay_unwrap<T14>::type , typename detail::decay_unwrap<T15>::type , typename detail::decay_unwrap<T16>::type>
        > result_type;
        return result_type(std::forward<F>(f)
          , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ) , std::forward<T14>( t14 ) , std::forward<T15>( t15 ) , std::forward<T16>( t16 )));
    }
}}
namespace hpx { namespace util
{
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    struct deferred_call_result_of<F(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17)>
      : util::result_of<typename util::decay<F>::type(
            typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type , typename detail::decay_unwrap<T11>::type , typename detail::decay_unwrap<T12>::type , typename detail::decay_unwrap<T13>::type , typename detail::decay_unwrap<T14>::type , typename detail::decay_unwrap<T15>::type , typename detail::decay_unwrap<T16>::type , typename detail::decay_unwrap<T17>::type)>
    {};
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    detail::deferred_call_impl<
        typename util::decay<F>::type
      , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type , typename detail::decay_unwrap<T11>::type , typename detail::decay_unwrap<T12>::type , typename detail::decay_unwrap<T13>::type , typename detail::decay_unwrap<T14>::type , typename detail::decay_unwrap<T15>::type , typename detail::decay_unwrap<T16>::type , typename detail::decay_unwrap<T17>::type>
    >
    deferred_call(F && f, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14 , T15 && t15 , T16 && t16 , T17 && t17)
    {
        typedef detail::deferred_call_impl<
            typename util::decay<F>::type
          , util::tuple<typename detail::decay_unwrap<T0>::type , typename detail::decay_unwrap<T1>::type , typename detail::decay_unwrap<T2>::type , typename detail::decay_unwrap<T3>::type , typename detail::decay_unwrap<T4>::type , typename detail::decay_unwrap<T5>::type , typename detail::decay_unwrap<T6>::type , typename detail::decay_unwrap<T7>::type , typename detail::decay_unwrap<T8>::type , typename detail::decay_unwrap<T9>::type , typename detail::decay_unwrap<T10>::type , typename detail::decay_unwrap<T11>::type , typename detail::decay_unwrap<T12>::type , typename detail::decay_unwrap<T13>::type , typename detail::decay_unwrap<T14>::type , typename detail::decay_unwrap<T15>::type , typename detail::decay_unwrap<T16>::type , typename detail::decay_unwrap<T17>::type>
        > result_type;
        return result_type(std::forward<F>(f)
          , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ) , std::forward<T14>( t14 ) , std::forward<T15>( t15 ) , std::forward<T16>( t16 ) , std::forward<T17>( t17 )));
    }
}}
