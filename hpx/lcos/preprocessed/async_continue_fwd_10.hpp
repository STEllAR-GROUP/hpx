// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx
{
    
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<Arguments>::value == 0>
      , lcos::future<typename traits::promise_local_result<Result>::type
    >::type
    async_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, BOOST_FWD_REF(F) f);
}
namespace hpx
{
    
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0, typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<Arguments>::value == 1>
      , lcos::future<typename traits::promise_local_result<Result>::type
    >::type
    async_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0,
        BOOST_FWD_REF(F) f);
}
namespace hpx
{
    
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1, typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<Arguments>::value == 2>
      , lcos::future<typename traits::promise_local_result<Result>::type
    >::type
    async_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1,
        BOOST_FWD_REF(F) f);
}
namespace hpx
{
    
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1 , typename Arg2, typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<Arguments>::value == 3>
      , lcos::future<typename traits::promise_local_result<Result>::type
    >::type
    async_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2,
        BOOST_FWD_REF(F) f);
}
namespace hpx
{
    
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3, typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<Arguments>::value == 4>
      , lcos::future<typename traits::promise_local_result<Result>::type
    >::type
    async_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3,
        BOOST_FWD_REF(F) f);
}
namespace hpx
{
    
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4, typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<Arguments>::value == 5>
      , lcos::future<typename traits::promise_local_result<Result>::type
    >::type
    async_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4,
        BOOST_FWD_REF(F) f);
}
namespace hpx
{
    
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5, typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<Arguments>::value == 6>
      , lcos::future<typename traits::promise_local_result<Result>::type
    >::type
    async_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5,
        BOOST_FWD_REF(F) f);
}
namespace hpx
{
    
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6, typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<Arguments>::value == 7>
      , lcos::future<typename traits::promise_local_result<Result>::type
    >::type
    async_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6,
        BOOST_FWD_REF(F) f);
}
namespace hpx
{
    
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7, typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<Arguments>::value == 8>
      , lcos::future<typename traits::promise_local_result<Result>::type
    >::type
    async_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7,
        BOOST_FWD_REF(F) f);
}
namespace hpx
{
    
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8, typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<Arguments>::value == 9>
      , lcos::future<typename traits::promise_local_result<Result>::type
    >::type
    async_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8,
        BOOST_FWD_REF(F) f);
}
namespace hpx
{
    
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9, typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<Arguments>::value == 10>
      , lcos::future<typename traits::promise_local_result<Result>::type
    >::type
    async_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9,
        BOOST_FWD_REF(F) f);
}
