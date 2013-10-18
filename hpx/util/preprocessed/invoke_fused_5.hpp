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
