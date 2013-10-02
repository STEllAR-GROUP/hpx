// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace util
{
    
    template <typename R, typename F, typename Arg0>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , args.a0);
    }
    template <typename R, typename F, typename Arg0>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0));
    }
    template <typename F, typename Arg0>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename boost::add_const<Arg0>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , args.a0);
    }
    template <typename F, typename Arg0>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename util::add_rvalue_reference<Arg0>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0));
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename F, typename Arg0 , typename Arg1>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , args.a0 , args.a1);
    }
    template <typename R, typename F, typename Arg0 , typename Arg1>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1));
    }
    template <typename F, typename Arg0 , typename Arg1>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , args.a0 , args.a1);
    }
    template <typename F, typename Arg0 , typename Arg1>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename util::add_rvalue_reference<Arg0>::type , typename util::add_rvalue_reference<Arg1>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1));
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2);
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2);
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename util::add_rvalue_reference<Arg0>::type , typename util::add_rvalue_reference<Arg1>::type , typename util::add_rvalue_reference<Arg2>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2));
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3);
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3);
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename util::add_rvalue_reference<Arg0>::type , typename util::add_rvalue_reference<Arg1>::type , typename util::add_rvalue_reference<Arg2>::type , typename util::add_rvalue_reference<Arg3>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3));
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4);
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4);
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename util::add_rvalue_reference<Arg0>::type , typename util::add_rvalue_reference<Arg1>::type , typename util::add_rvalue_reference<Arg2>::type , typename util::add_rvalue_reference<Arg3>::type , typename util::add_rvalue_reference<Arg4>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4));
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5);
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5);
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename util::add_rvalue_reference<Arg0>::type , typename util::add_rvalue_reference<Arg1>::type , typename util::add_rvalue_reference<Arg2>::type , typename util::add_rvalue_reference<Arg3>::type , typename util::add_rvalue_reference<Arg4>::type , typename util::add_rvalue_reference<Arg5>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5));
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6);
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6);
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename util::add_rvalue_reference<Arg0>::type , typename util::add_rvalue_reference<Arg1>::type , typename util::add_rvalue_reference<Arg2>::type , typename util::add_rvalue_reference<Arg3>::type , typename util::add_rvalue_reference<Arg4>::type , typename util::add_rvalue_reference<Arg5>::type , typename util::add_rvalue_reference<Arg6>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6));
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6 , args.a7);
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6) , boost::forward<Arg7>(args.a7));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6 , args.a7);
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename util::add_rvalue_reference<Arg0>::type , typename util::add_rvalue_reference<Arg1>::type , typename util::add_rvalue_reference<Arg2>::type , typename util::add_rvalue_reference<Arg3>::type , typename util::add_rvalue_reference<Arg4>::type , typename util::add_rvalue_reference<Arg5>::type , typename util::add_rvalue_reference<Arg6>::type , typename util::add_rvalue_reference<Arg7>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6) , boost::forward<Arg7>(args.a7));
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6 , args.a7 , args.a8);
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6) , boost::forward<Arg7>(args.a7) , boost::forward<Arg8>(args.a8));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type , typename boost::add_const<Arg8>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6 , args.a7 , args.a8);
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename util::add_rvalue_reference<Arg0>::type , typename util::add_rvalue_reference<Arg1>::type , typename util::add_rvalue_reference<Arg2>::type , typename util::add_rvalue_reference<Arg3>::type , typename util::add_rvalue_reference<Arg4>::type , typename util::add_rvalue_reference<Arg5>::type , typename util::add_rvalue_reference<Arg6>::type , typename util::add_rvalue_reference<Arg7>::type , typename util::add_rvalue_reference<Arg8>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6) , boost::forward<Arg7>(args.a7) , boost::forward<Arg8>(args.a8));
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6 , args.a7 , args.a8 , args.a9);
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6) , boost::forward<Arg7>(args.a7) , boost::forward<Arg8>(args.a8) , boost::forward<Arg9>(args.a9));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type , typename boost::add_const<Arg8>::type , typename boost::add_const<Arg9>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6 , args.a7 , args.a8 , args.a9);
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename util::add_rvalue_reference<Arg0>::type , typename util::add_rvalue_reference<Arg1>::type , typename util::add_rvalue_reference<Arg2>::type , typename util::add_rvalue_reference<Arg3>::type , typename util::add_rvalue_reference<Arg4>::type , typename util::add_rvalue_reference<Arg5>::type , typename util::add_rvalue_reference<Arg6>::type , typename util::add_rvalue_reference<Arg7>::type , typename util::add_rvalue_reference<Arg8>::type , typename util::add_rvalue_reference<Arg9>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6) , boost::forward<Arg7>(args.a7) , boost::forward<Arg8>(args.a8) , boost::forward<Arg9>(args.a9));
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6 , args.a7 , args.a8 , args.a9 , args.a10);
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6) , boost::forward<Arg7>(args.a7) , boost::forward<Arg8>(args.a8) , boost::forward<Arg9>(args.a9) , boost::forward<Arg10>(args.a10));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type , typename boost::add_const<Arg8>::type , typename boost::add_const<Arg9>::type , typename boost::add_const<Arg10>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6 , args.a7 , args.a8 , args.a9 , args.a10);
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename util::add_rvalue_reference<Arg0>::type , typename util::add_rvalue_reference<Arg1>::type , typename util::add_rvalue_reference<Arg2>::type , typename util::add_rvalue_reference<Arg3>::type , typename util::add_rvalue_reference<Arg4>::type , typename util::add_rvalue_reference<Arg5>::type , typename util::add_rvalue_reference<Arg6>::type , typename util::add_rvalue_reference<Arg7>::type , typename util::add_rvalue_reference<Arg8>::type , typename util::add_rvalue_reference<Arg9>::type , typename util::add_rvalue_reference<Arg10>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6) , boost::forward<Arg7>(args.a7) , boost::forward<Arg8>(args.a8) , boost::forward<Arg9>(args.a9) , boost::forward<Arg10>(args.a10));
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6 , args.a7 , args.a8 , args.a9 , args.a10 , args.a11);
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6) , boost::forward<Arg7>(args.a7) , boost::forward<Arg8>(args.a8) , boost::forward<Arg9>(args.a9) , boost::forward<Arg10>(args.a10) , boost::forward<Arg11>(args.a11));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type , typename boost::add_const<Arg8>::type , typename boost::add_const<Arg9>::type , typename boost::add_const<Arg10>::type , typename boost::add_const<Arg11>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6 , args.a7 , args.a8 , args.a9 , args.a10 , args.a11);
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename util::add_rvalue_reference<Arg0>::type , typename util::add_rvalue_reference<Arg1>::type , typename util::add_rvalue_reference<Arg2>::type , typename util::add_rvalue_reference<Arg3>::type , typename util::add_rvalue_reference<Arg4>::type , typename util::add_rvalue_reference<Arg5>::type , typename util::add_rvalue_reference<Arg6>::type , typename util::add_rvalue_reference<Arg7>::type , typename util::add_rvalue_reference<Arg8>::type , typename util::add_rvalue_reference<Arg9>::type , typename util::add_rvalue_reference<Arg10>::type , typename util::add_rvalue_reference<Arg11>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6) , boost::forward<Arg7>(args.a7) , boost::forward<Arg8>(args.a8) , boost::forward<Arg9>(args.a9) , boost::forward<Arg10>(args.a10) , boost::forward<Arg11>(args.a11));
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6 , args.a7 , args.a8 , args.a9 , args.a10 , args.a11 , args.a12);
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6) , boost::forward<Arg7>(args.a7) , boost::forward<Arg8>(args.a8) , boost::forward<Arg9>(args.a9) , boost::forward<Arg10>(args.a10) , boost::forward<Arg11>(args.a11) , boost::forward<Arg12>(args.a12));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type , typename boost::add_const<Arg8>::type , typename boost::add_const<Arg9>::type , typename boost::add_const<Arg10>::type , typename boost::add_const<Arg11>::type , typename boost::add_const<Arg12>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6 , args.a7 , args.a8 , args.a9 , args.a10 , args.a11 , args.a12);
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename util::add_rvalue_reference<Arg0>::type , typename util::add_rvalue_reference<Arg1>::type , typename util::add_rvalue_reference<Arg2>::type , typename util::add_rvalue_reference<Arg3>::type , typename util::add_rvalue_reference<Arg4>::type , typename util::add_rvalue_reference<Arg5>::type , typename util::add_rvalue_reference<Arg6>::type , typename util::add_rvalue_reference<Arg7>::type , typename util::add_rvalue_reference<Arg8>::type , typename util::add_rvalue_reference<Arg9>::type , typename util::add_rvalue_reference<Arg10>::type , typename util::add_rvalue_reference<Arg11>::type , typename util::add_rvalue_reference<Arg12>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6) , boost::forward<Arg7>(args.a7) , boost::forward<Arg8>(args.a8) , boost::forward<Arg9>(args.a9) , boost::forward<Arg10>(args.a10) , boost::forward<Arg11>(args.a11) , boost::forward<Arg12>(args.a12));
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6 , args.a7 , args.a8 , args.a9 , args.a10 , args.a11 , args.a12 , args.a13);
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6) , boost::forward<Arg7>(args.a7) , boost::forward<Arg8>(args.a8) , boost::forward<Arg9>(args.a9) , boost::forward<Arg10>(args.a10) , boost::forward<Arg11>(args.a11) , boost::forward<Arg12>(args.a12) , boost::forward<Arg13>(args.a13));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type , typename boost::add_const<Arg8>::type , typename boost::add_const<Arg9>::type , typename boost::add_const<Arg10>::type , typename boost::add_const<Arg11>::type , typename boost::add_const<Arg12>::type , typename boost::add_const<Arg13>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6 , args.a7 , args.a8 , args.a9 , args.a10 , args.a11 , args.a12 , args.a13);
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename util::add_rvalue_reference<Arg0>::type , typename util::add_rvalue_reference<Arg1>::type , typename util::add_rvalue_reference<Arg2>::type , typename util::add_rvalue_reference<Arg3>::type , typename util::add_rvalue_reference<Arg4>::type , typename util::add_rvalue_reference<Arg5>::type , typename util::add_rvalue_reference<Arg6>::type , typename util::add_rvalue_reference<Arg7>::type , typename util::add_rvalue_reference<Arg8>::type , typename util::add_rvalue_reference<Arg9>::type , typename util::add_rvalue_reference<Arg10>::type , typename util::add_rvalue_reference<Arg11>::type , typename util::add_rvalue_reference<Arg12>::type , typename util::add_rvalue_reference<Arg13>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6) , boost::forward<Arg7>(args.a7) , boost::forward<Arg8>(args.a8) , boost::forward<Arg9>(args.a9) , boost::forward<Arg10>(args.a10) , boost::forward<Arg11>(args.a11) , boost::forward<Arg12>(args.a12) , boost::forward<Arg13>(args.a13));
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6 , args.a7 , args.a8 , args.a9 , args.a10 , args.a11 , args.a12 , args.a13 , args.a14);
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6) , boost::forward<Arg7>(args.a7) , boost::forward<Arg8>(args.a8) , boost::forward<Arg9>(args.a9) , boost::forward<Arg10>(args.a10) , boost::forward<Arg11>(args.a11) , boost::forward<Arg12>(args.a12) , boost::forward<Arg13>(args.a13) , boost::forward<Arg14>(args.a14));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type , typename boost::add_const<Arg8>::type , typename boost::add_const<Arg9>::type , typename boost::add_const<Arg10>::type , typename boost::add_const<Arg11>::type , typename boost::add_const<Arg12>::type , typename boost::add_const<Arg13>::type , typename boost::add_const<Arg14>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6 , args.a7 , args.a8 , args.a9 , args.a10 , args.a11 , args.a12 , args.a13 , args.a14);
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename util::add_rvalue_reference<Arg0>::type , typename util::add_rvalue_reference<Arg1>::type , typename util::add_rvalue_reference<Arg2>::type , typename util::add_rvalue_reference<Arg3>::type , typename util::add_rvalue_reference<Arg4>::type , typename util::add_rvalue_reference<Arg5>::type , typename util::add_rvalue_reference<Arg6>::type , typename util::add_rvalue_reference<Arg7>::type , typename util::add_rvalue_reference<Arg8>::type , typename util::add_rvalue_reference<Arg9>::type , typename util::add_rvalue_reference<Arg10>::type , typename util::add_rvalue_reference<Arg11>::type , typename util::add_rvalue_reference<Arg12>::type , typename util::add_rvalue_reference<Arg13>::type , typename util::add_rvalue_reference<Arg14>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6) , boost::forward<Arg7>(args.a7) , boost::forward<Arg8>(args.a8) , boost::forward<Arg9>(args.a9) , boost::forward<Arg10>(args.a10) , boost::forward<Arg11>(args.a11) , boost::forward<Arg12>(args.a12) , boost::forward<Arg13>(args.a13) , boost::forward<Arg14>(args.a14));
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6 , args.a7 , args.a8 , args.a9 , args.a10 , args.a11 , args.a12 , args.a13 , args.a14 , args.a15);
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6) , boost::forward<Arg7>(args.a7) , boost::forward<Arg8>(args.a8) , boost::forward<Arg9>(args.a9) , boost::forward<Arg10>(args.a10) , boost::forward<Arg11>(args.a11) , boost::forward<Arg12>(args.a12) , boost::forward<Arg13>(args.a13) , boost::forward<Arg14>(args.a14) , boost::forward<Arg15>(args.a15));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type , typename boost::add_const<Arg8>::type , typename boost::add_const<Arg9>::type , typename boost::add_const<Arg10>::type , typename boost::add_const<Arg11>::type , typename boost::add_const<Arg12>::type , typename boost::add_const<Arg13>::type , typename boost::add_const<Arg14>::type , typename boost::add_const<Arg15>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6 , args.a7 , args.a8 , args.a9 , args.a10 , args.a11 , args.a12 , args.a13 , args.a14 , args.a15);
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename util::add_rvalue_reference<Arg0>::type , typename util::add_rvalue_reference<Arg1>::type , typename util::add_rvalue_reference<Arg2>::type , typename util::add_rvalue_reference<Arg3>::type , typename util::add_rvalue_reference<Arg4>::type , typename util::add_rvalue_reference<Arg5>::type , typename util::add_rvalue_reference<Arg6>::type , typename util::add_rvalue_reference<Arg7>::type , typename util::add_rvalue_reference<Arg8>::type , typename util::add_rvalue_reference<Arg9>::type , typename util::add_rvalue_reference<Arg10>::type , typename util::add_rvalue_reference<Arg11>::type , typename util::add_rvalue_reference<Arg12>::type , typename util::add_rvalue_reference<Arg13>::type , typename util::add_rvalue_reference<Arg14>::type , typename util::add_rvalue_reference<Arg15>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6) , boost::forward<Arg7>(args.a7) , boost::forward<Arg8>(args.a8) , boost::forward<Arg9>(args.a9) , boost::forward<Arg10>(args.a10) , boost::forward<Arg11>(args.a11) , boost::forward<Arg12>(args.a12) , boost::forward<Arg13>(args.a13) , boost::forward<Arg14>(args.a14) , boost::forward<Arg15>(args.a15));
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6 , args.a7 , args.a8 , args.a9 , args.a10 , args.a11 , args.a12 , args.a13 , args.a14 , args.a15 , args.a16);
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6) , boost::forward<Arg7>(args.a7) , boost::forward<Arg8>(args.a8) , boost::forward<Arg9>(args.a9) , boost::forward<Arg10>(args.a10) , boost::forward<Arg11>(args.a11) , boost::forward<Arg12>(args.a12) , boost::forward<Arg13>(args.a13) , boost::forward<Arg14>(args.a14) , boost::forward<Arg15>(args.a15) , boost::forward<Arg16>(args.a16));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type , typename boost::add_const<Arg8>::type , typename boost::add_const<Arg9>::type , typename boost::add_const<Arg10>::type , typename boost::add_const<Arg11>::type , typename boost::add_const<Arg12>::type , typename boost::add_const<Arg13>::type , typename boost::add_const<Arg14>::type , typename boost::add_const<Arg15>::type , typename boost::add_const<Arg16>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6 , args.a7 , args.a8 , args.a9 , args.a10 , args.a11 , args.a12 , args.a13 , args.a14 , args.a15 , args.a16);
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename util::add_rvalue_reference<Arg0>::type , typename util::add_rvalue_reference<Arg1>::type , typename util::add_rvalue_reference<Arg2>::type , typename util::add_rvalue_reference<Arg3>::type , typename util::add_rvalue_reference<Arg4>::type , typename util::add_rvalue_reference<Arg5>::type , typename util::add_rvalue_reference<Arg6>::type , typename util::add_rvalue_reference<Arg7>::type , typename util::add_rvalue_reference<Arg8>::type , typename util::add_rvalue_reference<Arg9>::type , typename util::add_rvalue_reference<Arg10>::type , typename util::add_rvalue_reference<Arg11>::type , typename util::add_rvalue_reference<Arg12>::type , typename util::add_rvalue_reference<Arg13>::type , typename util::add_rvalue_reference<Arg14>::type , typename util::add_rvalue_reference<Arg15>::type , typename util::add_rvalue_reference<Arg16>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6) , boost::forward<Arg7>(args.a7) , boost::forward<Arg8>(args.a8) , boost::forward<Arg9>(args.a9) , boost::forward<Arg10>(args.a10) , boost::forward<Arg11>(args.a11) , boost::forward<Arg12>(args.a12) , boost::forward<Arg13>(args.a13) , boost::forward<Arg14>(args.a14) , boost::forward<Arg15>(args.a15) , boost::forward<Arg16>(args.a16));
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6 , args.a7 , args.a8 , args.a9 , args.a10 , args.a11 , args.a12 , args.a13 , args.a14 , args.a15 , args.a16 , args.a17);
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6) , boost::forward<Arg7>(args.a7) , boost::forward<Arg8>(args.a8) , boost::forward<Arg9>(args.a9) , boost::forward<Arg10>(args.a10) , boost::forward<Arg11>(args.a11) , boost::forward<Arg12>(args.a12) , boost::forward<Arg13>(args.a13) , boost::forward<Arg14>(args.a14) , boost::forward<Arg15>(args.a15) , boost::forward<Arg16>(args.a16) , boost::forward<Arg17>(args.a17));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type , typename boost::add_const<Arg8>::type , typename boost::add_const<Arg9>::type , typename boost::add_const<Arg10>::type , typename boost::add_const<Arg11>::type , typename boost::add_const<Arg12>::type , typename boost::add_const<Arg13>::type , typename boost::add_const<Arg14>::type , typename boost::add_const<Arg15>::type , typename boost::add_const<Arg16>::type , typename boost::add_const<Arg17>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , args.a0 , args.a1 , args.a2 , args.a3 , args.a4 , args.a5 , args.a6 , args.a7 , args.a8 , args.a9 , args.a10 , args.a11 , args.a12 , args.a13 , args.a14 , args.a15 , args.a16 , args.a17);
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(typename util::add_rvalue_reference<Arg0>::type , typename util::add_rvalue_reference<Arg1>::type , typename util::add_rvalue_reference<Arg2>::type , typename util::add_rvalue_reference<Arg3>::type , typename util::add_rvalue_reference<Arg4>::type , typename util::add_rvalue_reference<Arg5>::type , typename util::add_rvalue_reference<Arg6>::type , typename util::add_rvalue_reference<Arg7>::type , typename util::add_rvalue_reference<Arg8>::type , typename util::add_rvalue_reference<Arg9>::type , typename util::add_rvalue_reference<Arg10>::type , typename util::add_rvalue_reference<Arg11>::type , typename util::add_rvalue_reference<Arg12>::type , typename util::add_rvalue_reference<Arg13>::type , typename util::add_rvalue_reference<Arg14>::type , typename util::add_rvalue_reference<Arg15>::type , typename util::add_rvalue_reference<Arg16>::type , typename util::add_rvalue_reference<Arg17>::type)
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , boost::forward<Arg0>(args.a0) , boost::forward<Arg1>(args.a1) , boost::forward<Arg2>(args.a2) , boost::forward<Arg3>(args.a3) , boost::forward<Arg4>(args.a4) , boost::forward<Arg5>(args.a5) , boost::forward<Arg6>(args.a6) , boost::forward<Arg7>(args.a7) , boost::forward<Arg8>(args.a8) , boost::forward<Arg9>(args.a9) , boost::forward<Arg10>(args.a10) , boost::forward<Arg11>(args.a11) , boost::forward<Arg12>(args.a12) , boost::forward<Arg13>(args.a13) , boost::forward<Arg14>(args.a14) , boost::forward<Arg15>(args.a15) , boost::forward<Arg16>(args.a16) , boost::forward<Arg17>(args.a17));
    }
}}
