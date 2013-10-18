// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace util
{
    namespace detail
    {
        template <typename FD, typename F, typename Tuple>
        struct invoke_fused_result_of_impl<
            FD, F(Tuple)
          , typename boost::enable_if_c<
                util::tuple_size<Tuple>::value == 1
            >::type
        > : invoke_result_of<
                F(typename detail::qualify_as< typename util::tuple_element< 0 , typename util::decay<Tuple>::type >::type , Tuple >::type)
            >
        {};
    }
    
    template <typename R, typename F, typename Arg0>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0> >::value == 1
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f
      , util::tuple<Arg0> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(args));
    }
    template <typename R, typename F, typename Arg0>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0> >::value == 1
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)));
    }
    template <typename F, typename Arg0>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0> >::value == 1
      , typename invoke_result_of<
            F(typename boost::add_const<Arg0>::type)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , util::tuple<Arg0> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(args));
    }
    template <typename F, typename Arg0>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0> >::value == 1
      , typename invoke_result_of<
            F(Arg0)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename FD, typename F, typename Tuple>
        struct invoke_fused_result_of_impl<
            FD, F(Tuple)
          , typename boost::enable_if_c<
                util::tuple_size<Tuple>::value == 2
            >::type
        > : invoke_result_of<
                F(typename detail::qualify_as< typename util::tuple_element< 0 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 1 , typename util::decay<Tuple>::type >::type , Tuple >::type)
            >
        {};
    }
    
    template <typename R, typename F, typename Arg0 , typename Arg1>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1> >::value == 2
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1> >::value == 2
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)));
    }
    template <typename F, typename Arg0 , typename Arg1>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1> >::value == 2
      , typename invoke_result_of<
            F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args));
    }
    template <typename F, typename Arg0 , typename Arg1>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1> >::value == 2
      , typename invoke_result_of<
            F(Arg0 , Arg1)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename FD, typename F, typename Tuple>
        struct invoke_fused_result_of_impl<
            FD, F(Tuple)
          , typename boost::enable_if_c<
                util::tuple_size<Tuple>::value == 3
            >::type
        > : invoke_result_of<
                F(typename detail::qualify_as< typename util::tuple_element< 0 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 1 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 2 , typename util::decay<Tuple>::type >::type , Tuple >::type)
            >
        {};
    }
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2> >::value == 3
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2> >::value == 3
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2> >::value == 3
      , typename invoke_result_of<
            F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2> >::value == 3
      , typename invoke_result_of<
            F(Arg0 , Arg1 , Arg2)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename FD, typename F, typename Tuple>
        struct invoke_fused_result_of_impl<
            FD, F(Tuple)
          , typename boost::enable_if_c<
                util::tuple_size<Tuple>::value == 4
            >::type
        > : invoke_result_of<
                F(typename detail::qualify_as< typename util::tuple_element< 0 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 1 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 2 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 3 , typename util::decay<Tuple>::type >::type , Tuple >::type)
            >
        {};
    }
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3> >::value == 4
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3> >::value == 4
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3> >::value == 4
      , typename invoke_result_of<
            F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3> >::value == 4
      , typename invoke_result_of<
            F(Arg0 , Arg1 , Arg2 , Arg3)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename FD, typename F, typename Tuple>
        struct invoke_fused_result_of_impl<
            FD, F(Tuple)
          , typename boost::enable_if_c<
                util::tuple_size<Tuple>::value == 5
            >::type
        > : invoke_result_of<
                F(typename detail::qualify_as< typename util::tuple_element< 0 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 1 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 2 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 3 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 4 , typename util::decay<Tuple>::type >::type , Tuple >::type)
            >
        {};
    }
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4> >::value == 5
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4> >::value == 5
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4> >::value == 5
      , typename invoke_result_of<
            F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4> >::value == 5
      , typename invoke_result_of<
            F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename FD, typename F, typename Tuple>
        struct invoke_fused_result_of_impl<
            FD, F(Tuple)
          , typename boost::enable_if_c<
                util::tuple_size<Tuple>::value == 6
            >::type
        > : invoke_result_of<
                F(typename detail::qualify_as< typename util::tuple_element< 0 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 1 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 2 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 3 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 4 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 5 , typename util::decay<Tuple>::type >::type , Tuple >::type)
            >
        {};
    }
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5> >::value == 6
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5> >::value == 6
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5> >::value == 6
      , typename invoke_result_of<
            F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5> >::value == 6
      , typename invoke_result_of<
            F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename FD, typename F, typename Tuple>
        struct invoke_fused_result_of_impl<
            FD, F(Tuple)
          , typename boost::enable_if_c<
                util::tuple_size<Tuple>::value == 7
            >::type
        > : invoke_result_of<
                F(typename detail::qualify_as< typename util::tuple_element< 0 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 1 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 2 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 3 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 4 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 5 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 6 , typename util::decay<Tuple>::type >::type , Tuple >::type)
            >
        {};
    }
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6> >::value == 7
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6> >::value == 7
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6> >::value == 7
      , typename invoke_result_of<
            F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6> >::value == 7
      , typename invoke_result_of<
            F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename FD, typename F, typename Tuple>
        struct invoke_fused_result_of_impl<
            FD, F(Tuple)
          , typename boost::enable_if_c<
                util::tuple_size<Tuple>::value == 8
            >::type
        > : invoke_result_of<
                F(typename detail::qualify_as< typename util::tuple_element< 0 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 1 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 2 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 3 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 4 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 5 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 6 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 7 , typename util::decay<Tuple>::type >::type , Tuple >::type)
            >
        {};
    }
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7> >::value == 8
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7> >::value == 8
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7> >::value == 8
      , typename invoke_result_of<
            F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7> >::value == 8
      , typename invoke_result_of<
            F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename FD, typename F, typename Tuple>
        struct invoke_fused_result_of_impl<
            FD, F(Tuple)
          , typename boost::enable_if_c<
                util::tuple_size<Tuple>::value == 9
            >::type
        > : invoke_result_of<
                F(typename detail::qualify_as< typename util::tuple_element< 0 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 1 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 2 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 3 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 4 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 5 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 6 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 7 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 8 , typename util::decay<Tuple>::type >::type , Tuple >::type)
            >
        {};
    }
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8> >::value == 9
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8> >::value == 9
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8> >::value == 9
      , typename invoke_result_of<
            F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type , typename boost::add_const<Arg8>::type)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8> >::value == 9
      , typename invoke_result_of<
            F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename FD, typename F, typename Tuple>
        struct invoke_fused_result_of_impl<
            FD, F(Tuple)
          , typename boost::enable_if_c<
                util::tuple_size<Tuple>::value == 10
            >::type
        > : invoke_result_of<
                F(typename detail::qualify_as< typename util::tuple_element< 0 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 1 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 2 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 3 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 4 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 5 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 6 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 7 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 8 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 9 , typename util::decay<Tuple>::type >::type , Tuple >::type)
            >
        {};
    }
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9> >::value == 10
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args) , util::get< 9>(args));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9> >::value == 10
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)) , util::get< 9>(boost::move(args)));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9> >::value == 10
      , typename invoke_result_of<
            F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type , typename boost::add_const<Arg8>::type , typename boost::add_const<Arg9>::type)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args) , util::get< 9>(args));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9> >::value == 10
      , typename invoke_result_of<
            F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)) , util::get< 9>(boost::move(args)));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename FD, typename F, typename Tuple>
        struct invoke_fused_result_of_impl<
            FD, F(Tuple)
          , typename boost::enable_if_c<
                util::tuple_size<Tuple>::value == 11
            >::type
        > : invoke_result_of<
                F(typename detail::qualify_as< typename util::tuple_element< 0 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 1 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 2 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 3 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 4 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 5 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 6 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 7 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 8 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 9 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 10 , typename util::decay<Tuple>::type >::type , Tuple >::type)
            >
        {};
    }
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10> >::value == 11
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args) , util::get< 9>(args) , util::get< 10>(args));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10> >::value == 11
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)) , util::get< 9>(boost::move(args)) , util::get< 10>(boost::move(args)));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10> >::value == 11
      , typename invoke_result_of<
            F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type , typename boost::add_const<Arg8>::type , typename boost::add_const<Arg9>::type , typename boost::add_const<Arg10>::type)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args) , util::get< 9>(args) , util::get< 10>(args));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10> >::value == 11
      , typename invoke_result_of<
            F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)) , util::get< 9>(boost::move(args)) , util::get< 10>(boost::move(args)));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename FD, typename F, typename Tuple>
        struct invoke_fused_result_of_impl<
            FD, F(Tuple)
          , typename boost::enable_if_c<
                util::tuple_size<Tuple>::value == 12
            >::type
        > : invoke_result_of<
                F(typename detail::qualify_as< typename util::tuple_element< 0 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 1 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 2 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 3 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 4 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 5 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 6 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 7 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 8 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 9 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 10 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 11 , typename util::decay<Tuple>::type >::type , Tuple >::type)
            >
        {};
    }
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11> >::value == 12
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args) , util::get< 9>(args) , util::get< 10>(args) , util::get< 11>(args));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11> >::value == 12
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)) , util::get< 9>(boost::move(args)) , util::get< 10>(boost::move(args)) , util::get< 11>(boost::move(args)));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11> >::value == 12
      , typename invoke_result_of<
            F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type , typename boost::add_const<Arg8>::type , typename boost::add_const<Arg9>::type , typename boost::add_const<Arg10>::type , typename boost::add_const<Arg11>::type)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args) , util::get< 9>(args) , util::get< 10>(args) , util::get< 11>(args));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11> >::value == 12
      , typename invoke_result_of<
            F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)) , util::get< 9>(boost::move(args)) , util::get< 10>(boost::move(args)) , util::get< 11>(boost::move(args)));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename FD, typename F, typename Tuple>
        struct invoke_fused_result_of_impl<
            FD, F(Tuple)
          , typename boost::enable_if_c<
                util::tuple_size<Tuple>::value == 13
            >::type
        > : invoke_result_of<
                F(typename detail::qualify_as< typename util::tuple_element< 0 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 1 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 2 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 3 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 4 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 5 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 6 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 7 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 8 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 9 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 10 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 11 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 12 , typename util::decay<Tuple>::type >::type , Tuple >::type)
            >
        {};
    }
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12> >::value == 13
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args) , util::get< 9>(args) , util::get< 10>(args) , util::get< 11>(args) , util::get< 12>(args));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12> >::value == 13
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)) , util::get< 9>(boost::move(args)) , util::get< 10>(boost::move(args)) , util::get< 11>(boost::move(args)) , util::get< 12>(boost::move(args)));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12> >::value == 13
      , typename invoke_result_of<
            F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type , typename boost::add_const<Arg8>::type , typename boost::add_const<Arg9>::type , typename boost::add_const<Arg10>::type , typename boost::add_const<Arg11>::type , typename boost::add_const<Arg12>::type)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args) , util::get< 9>(args) , util::get< 10>(args) , util::get< 11>(args) , util::get< 12>(args));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12> >::value == 13
      , typename invoke_result_of<
            F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)) , util::get< 9>(boost::move(args)) , util::get< 10>(boost::move(args)) , util::get< 11>(boost::move(args)) , util::get< 12>(boost::move(args)));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename FD, typename F, typename Tuple>
        struct invoke_fused_result_of_impl<
            FD, F(Tuple)
          , typename boost::enable_if_c<
                util::tuple_size<Tuple>::value == 14
            >::type
        > : invoke_result_of<
                F(typename detail::qualify_as< typename util::tuple_element< 0 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 1 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 2 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 3 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 4 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 5 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 6 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 7 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 8 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 9 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 10 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 11 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 12 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 13 , typename util::decay<Tuple>::type >::type , Tuple >::type)
            >
        {};
    }
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13> >::value == 14
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args) , util::get< 9>(args) , util::get< 10>(args) , util::get< 11>(args) , util::get< 12>(args) , util::get< 13>(args));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13> >::value == 14
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)) , util::get< 9>(boost::move(args)) , util::get< 10>(boost::move(args)) , util::get< 11>(boost::move(args)) , util::get< 12>(boost::move(args)) , util::get< 13>(boost::move(args)));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13> >::value == 14
      , typename invoke_result_of<
            F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type , typename boost::add_const<Arg8>::type , typename boost::add_const<Arg9>::type , typename boost::add_const<Arg10>::type , typename boost::add_const<Arg11>::type , typename boost::add_const<Arg12>::type , typename boost::add_const<Arg13>::type)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args) , util::get< 9>(args) , util::get< 10>(args) , util::get< 11>(args) , util::get< 12>(args) , util::get< 13>(args));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13> >::value == 14
      , typename invoke_result_of<
            F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)) , util::get< 9>(boost::move(args)) , util::get< 10>(boost::move(args)) , util::get< 11>(boost::move(args)) , util::get< 12>(boost::move(args)) , util::get< 13>(boost::move(args)));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename FD, typename F, typename Tuple>
        struct invoke_fused_result_of_impl<
            FD, F(Tuple)
          , typename boost::enable_if_c<
                util::tuple_size<Tuple>::value == 15
            >::type
        > : invoke_result_of<
                F(typename detail::qualify_as< typename util::tuple_element< 0 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 1 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 2 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 3 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 4 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 5 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 6 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 7 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 8 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 9 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 10 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 11 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 12 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 13 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 14 , typename util::decay<Tuple>::type >::type , Tuple >::type)
            >
        {};
    }
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14> >::value == 15
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args) , util::get< 9>(args) , util::get< 10>(args) , util::get< 11>(args) , util::get< 12>(args) , util::get< 13>(args) , util::get< 14>(args));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14> >::value == 15
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)) , util::get< 9>(boost::move(args)) , util::get< 10>(boost::move(args)) , util::get< 11>(boost::move(args)) , util::get< 12>(boost::move(args)) , util::get< 13>(boost::move(args)) , util::get< 14>(boost::move(args)));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14> >::value == 15
      , typename invoke_result_of<
            F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type , typename boost::add_const<Arg8>::type , typename boost::add_const<Arg9>::type , typename boost::add_const<Arg10>::type , typename boost::add_const<Arg11>::type , typename boost::add_const<Arg12>::type , typename boost::add_const<Arg13>::type , typename boost::add_const<Arg14>::type)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args) , util::get< 9>(args) , util::get< 10>(args) , util::get< 11>(args) , util::get< 12>(args) , util::get< 13>(args) , util::get< 14>(args));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14> >::value == 15
      , typename invoke_result_of<
            F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)) , util::get< 9>(boost::move(args)) , util::get< 10>(boost::move(args)) , util::get< 11>(boost::move(args)) , util::get< 12>(boost::move(args)) , util::get< 13>(boost::move(args)) , util::get< 14>(boost::move(args)));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename FD, typename F, typename Tuple>
        struct invoke_fused_result_of_impl<
            FD, F(Tuple)
          , typename boost::enable_if_c<
                util::tuple_size<Tuple>::value == 16
            >::type
        > : invoke_result_of<
                F(typename detail::qualify_as< typename util::tuple_element< 0 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 1 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 2 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 3 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 4 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 5 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 6 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 7 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 8 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 9 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 10 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 11 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 12 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 13 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 14 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 15 , typename util::decay<Tuple>::type >::type , Tuple >::type)
            >
        {};
    }
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15> >::value == 16
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args) , util::get< 9>(args) , util::get< 10>(args) , util::get< 11>(args) , util::get< 12>(args) , util::get< 13>(args) , util::get< 14>(args) , util::get< 15>(args));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15> >::value == 16
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)) , util::get< 9>(boost::move(args)) , util::get< 10>(boost::move(args)) , util::get< 11>(boost::move(args)) , util::get< 12>(boost::move(args)) , util::get< 13>(boost::move(args)) , util::get< 14>(boost::move(args)) , util::get< 15>(boost::move(args)));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15> >::value == 16
      , typename invoke_result_of<
            F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type , typename boost::add_const<Arg8>::type , typename boost::add_const<Arg9>::type , typename boost::add_const<Arg10>::type , typename boost::add_const<Arg11>::type , typename boost::add_const<Arg12>::type , typename boost::add_const<Arg13>::type , typename boost::add_const<Arg14>::type , typename boost::add_const<Arg15>::type)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args) , util::get< 9>(args) , util::get< 10>(args) , util::get< 11>(args) , util::get< 12>(args) , util::get< 13>(args) , util::get< 14>(args) , util::get< 15>(args));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15> >::value == 16
      , typename invoke_result_of<
            F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)) , util::get< 9>(boost::move(args)) , util::get< 10>(boost::move(args)) , util::get< 11>(boost::move(args)) , util::get< 12>(boost::move(args)) , util::get< 13>(boost::move(args)) , util::get< 14>(boost::move(args)) , util::get< 15>(boost::move(args)));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename FD, typename F, typename Tuple>
        struct invoke_fused_result_of_impl<
            FD, F(Tuple)
          , typename boost::enable_if_c<
                util::tuple_size<Tuple>::value == 17
            >::type
        > : invoke_result_of<
                F(typename detail::qualify_as< typename util::tuple_element< 0 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 1 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 2 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 3 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 4 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 5 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 6 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 7 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 8 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 9 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 10 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 11 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 12 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 13 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 14 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 15 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 16 , typename util::decay<Tuple>::type >::type , Tuple >::type)
            >
        {};
    }
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16> >::value == 17
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args) , util::get< 9>(args) , util::get< 10>(args) , util::get< 11>(args) , util::get< 12>(args) , util::get< 13>(args) , util::get< 14>(args) , util::get< 15>(args) , util::get< 16>(args));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16> >::value == 17
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)) , util::get< 9>(boost::move(args)) , util::get< 10>(boost::move(args)) , util::get< 11>(boost::move(args)) , util::get< 12>(boost::move(args)) , util::get< 13>(boost::move(args)) , util::get< 14>(boost::move(args)) , util::get< 15>(boost::move(args)) , util::get< 16>(boost::move(args)));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16> >::value == 17
      , typename invoke_result_of<
            F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type , typename boost::add_const<Arg8>::type , typename boost::add_const<Arg9>::type , typename boost::add_const<Arg10>::type , typename boost::add_const<Arg11>::type , typename boost::add_const<Arg12>::type , typename boost::add_const<Arg13>::type , typename boost::add_const<Arg14>::type , typename boost::add_const<Arg15>::type , typename boost::add_const<Arg16>::type)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args) , util::get< 9>(args) , util::get< 10>(args) , util::get< 11>(args) , util::get< 12>(args) , util::get< 13>(args) , util::get< 14>(args) , util::get< 15>(args) , util::get< 16>(args));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16> >::value == 17
      , typename invoke_result_of<
            F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)) , util::get< 9>(boost::move(args)) , util::get< 10>(boost::move(args)) , util::get< 11>(boost::move(args)) , util::get< 12>(boost::move(args)) , util::get< 13>(boost::move(args)) , util::get< 14>(boost::move(args)) , util::get< 15>(boost::move(args)) , util::get< 16>(boost::move(args)));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename FD, typename F, typename Tuple>
        struct invoke_fused_result_of_impl<
            FD, F(Tuple)
          , typename boost::enable_if_c<
                util::tuple_size<Tuple>::value == 18
            >::type
        > : invoke_result_of<
                F(typename detail::qualify_as< typename util::tuple_element< 0 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 1 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 2 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 3 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 4 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 5 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 6 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 7 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 8 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 9 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 10 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 11 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 12 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 13 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 14 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 15 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 16 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 17 , typename util::decay<Tuple>::type >::type , Tuple >::type)
            >
        {};
    }
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17> >::value == 18
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args) , util::get< 9>(args) , util::get< 10>(args) , util::get< 11>(args) , util::get< 12>(args) , util::get< 13>(args) , util::get< 14>(args) , util::get< 15>(args) , util::get< 16>(args) , util::get< 17>(args));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17> >::value == 18
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)) , util::get< 9>(boost::move(args)) , util::get< 10>(boost::move(args)) , util::get< 11>(boost::move(args)) , util::get< 12>(boost::move(args)) , util::get< 13>(boost::move(args)) , util::get< 14>(boost::move(args)) , util::get< 15>(boost::move(args)) , util::get< 16>(boost::move(args)) , util::get< 17>(boost::move(args)));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17> >::value == 18
      , typename invoke_result_of<
            F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type , typename boost::add_const<Arg8>::type , typename boost::add_const<Arg9>::type , typename boost::add_const<Arg10>::type , typename boost::add_const<Arg11>::type , typename boost::add_const<Arg12>::type , typename boost::add_const<Arg13>::type , typename boost::add_const<Arg14>::type , typename boost::add_const<Arg15>::type , typename boost::add_const<Arg16>::type , typename boost::add_const<Arg17>::type)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args) , util::get< 9>(args) , util::get< 10>(args) , util::get< 11>(args) , util::get< 12>(args) , util::get< 13>(args) , util::get< 14>(args) , util::get< 15>(args) , util::get< 16>(args) , util::get< 17>(args));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17> >::value == 18
      , typename invoke_result_of<
            F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)) , util::get< 9>(boost::move(args)) , util::get< 10>(boost::move(args)) , util::get< 11>(boost::move(args)) , util::get< 12>(boost::move(args)) , util::get< 13>(boost::move(args)) , util::get< 14>(boost::move(args)) , util::get< 15>(boost::move(args)) , util::get< 16>(boost::move(args)) , util::get< 17>(boost::move(args)));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename FD, typename F, typename Tuple>
        struct invoke_fused_result_of_impl<
            FD, F(Tuple)
          , typename boost::enable_if_c<
                util::tuple_size<Tuple>::value == 19
            >::type
        > : invoke_result_of<
                F(typename detail::qualify_as< typename util::tuple_element< 0 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 1 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 2 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 3 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 4 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 5 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 6 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 7 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 8 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 9 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 10 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 11 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 12 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 13 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 14 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 15 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 16 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 17 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 18 , typename util::decay<Tuple>::type >::type , Tuple >::type)
            >
        {};
    }
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18> >::value == 19
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args) , util::get< 9>(args) , util::get< 10>(args) , util::get< 11>(args) , util::get< 12>(args) , util::get< 13>(args) , util::get< 14>(args) , util::get< 15>(args) , util::get< 16>(args) , util::get< 17>(args) , util::get< 18>(args));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18> >::value == 19
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)) , util::get< 9>(boost::move(args)) , util::get< 10>(boost::move(args)) , util::get< 11>(boost::move(args)) , util::get< 12>(boost::move(args)) , util::get< 13>(boost::move(args)) , util::get< 14>(boost::move(args)) , util::get< 15>(boost::move(args)) , util::get< 16>(boost::move(args)) , util::get< 17>(boost::move(args)) , util::get< 18>(boost::move(args)));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18> >::value == 19
      , typename invoke_result_of<
            F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type , typename boost::add_const<Arg8>::type , typename boost::add_const<Arg9>::type , typename boost::add_const<Arg10>::type , typename boost::add_const<Arg11>::type , typename boost::add_const<Arg12>::type , typename boost::add_const<Arg13>::type , typename boost::add_const<Arg14>::type , typename boost::add_const<Arg15>::type , typename boost::add_const<Arg16>::type , typename boost::add_const<Arg17>::type , typename boost::add_const<Arg18>::type)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args) , util::get< 9>(args) , util::get< 10>(args) , util::get< 11>(args) , util::get< 12>(args) , util::get< 13>(args) , util::get< 14>(args) , util::get< 15>(args) , util::get< 16>(args) , util::get< 17>(args) , util::get< 18>(args));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18> >::value == 19
      , typename invoke_result_of<
            F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)) , util::get< 9>(boost::move(args)) , util::get< 10>(boost::move(args)) , util::get< 11>(boost::move(args)) , util::get< 12>(boost::move(args)) , util::get< 13>(boost::move(args)) , util::get< 14>(boost::move(args)) , util::get< 15>(boost::move(args)) , util::get< 16>(boost::move(args)) , util::get< 17>(boost::move(args)) , util::get< 18>(boost::move(args)));
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename FD, typename F, typename Tuple>
        struct invoke_fused_result_of_impl<
            FD, F(Tuple)
          , typename boost::enable_if_c<
                util::tuple_size<Tuple>::value == 20
            >::type
        > : invoke_result_of<
                F(typename detail::qualify_as< typename util::tuple_element< 0 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 1 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 2 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 3 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 4 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 5 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 6 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 7 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 8 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 9 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 10 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 11 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 12 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 13 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 14 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 15 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 16 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 17 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 18 , typename util::decay<Tuple>::type >::type , Tuple >::type , typename detail::qualify_as< typename util::tuple_element< 19 , typename util::decay<Tuple>::type >::type , Tuple >::type)
            >
        {};
    }
    
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18 , Arg19> >::value == 20
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18 , Arg19> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args) , util::get< 9>(args) , util::get< 10>(args) , util::get< 11>(args) , util::get< 12>(args) , util::get< 13>(args) , util::get< 14>(args) , util::get< 15>(args) , util::get< 16>(args) , util::get< 17>(args) , util::get< 18>(args) , util::get< 19>(args));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18 , Arg19> >::value == 20
      , R
    >::type
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18 , Arg19>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)) , util::get< 9>(boost::move(args)) , util::get< 10>(boost::move(args)) , util::get< 11>(boost::move(args)) , util::get< 12>(boost::move(args)) , util::get< 13>(boost::move(args)) , util::get< 14>(boost::move(args)) , util::get< 15>(boost::move(args)) , util::get< 16>(boost::move(args)) , util::get< 17>(boost::move(args)) , util::get< 18>(boost::move(args)) , util::get< 19>(boost::move(args)));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18 , Arg19> >::value == 20
      , typename invoke_result_of<
            F(typename boost::add_const<Arg0>::type , typename boost::add_const<Arg1>::type , typename boost::add_const<Arg2>::type , typename boost::add_const<Arg3>::type , typename boost::add_const<Arg4>::type , typename boost::add_const<Arg5>::type , typename boost::add_const<Arg6>::type , typename boost::add_const<Arg7>::type , typename boost::add_const<Arg8>::type , typename boost::add_const<Arg9>::type , typename boost::add_const<Arg10>::type , typename boost::add_const<Arg11>::type , typename boost::add_const<Arg12>::type , typename boost::add_const<Arg13>::type , typename boost::add_const<Arg14>::type , typename boost::add_const<Arg15>::type , typename boost::add_const<Arg16>::type , typename boost::add_const<Arg17>::type , typename boost::add_const<Arg18>::type , typename boost::add_const<Arg19>::type)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18 , Arg19> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(args) , util::get< 1>(args) , util::get< 2>(args) , util::get< 3>(args) , util::get< 4>(args) , util::get< 5>(args) , util::get< 6>(args) , util::get< 7>(args) , util::get< 8>(args) , util::get< 9>(args) , util::get< 10>(args) , util::get< 11>(args) , util::get< 12>(args) , util::get< 13>(args) , util::get< 14>(args) , util::get< 15>(args) , util::get< 16>(args) , util::get< 17>(args) , util::get< 18>(args) , util::get< 19>(args));
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19>
    BOOST_FORCEINLINE
    typename boost::enable_if_c<
        util::tuple_size<util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18 , Arg19> >::value == 20
      , typename invoke_result_of<
            F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18 , Arg19)
        >::type
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        util::tuple<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18 , Arg19>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , util::get< 0>(boost::move(args)) , util::get< 1>(boost::move(args)) , util::get< 2>(boost::move(args)) , util::get< 3>(boost::move(args)) , util::get< 4>(boost::move(args)) , util::get< 5>(boost::move(args)) , util::get< 6>(boost::move(args)) , util::get< 7>(boost::move(args)) , util::get< 8>(boost::move(args)) , util::get< 9>(boost::move(args)) , util::get< 10>(boost::move(args)) , util::get< 11>(boost::move(args)) , util::get< 12>(boost::move(args)) , util::get< 13>(boost::move(args)) , util::get< 14>(boost::move(args)) , util::get< 15>(boost::move(args)) , util::get< 16>(boost::move(args)) , util::get< 17>(boost::move(args)) , util::get< 18>(boost::move(args)) , util::get< 19>(boost::move(args)));
    }
}}
