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
