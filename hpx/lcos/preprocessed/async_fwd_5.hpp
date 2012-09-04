// Copyright (c) 2007-2012 Hartmut Kaiser
// Copyright (c)      2012 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx
{
    
    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived>
    lcos::future<typename traits::promise_local_result<Result>::type, Result>
    async (
        hpx::actions::action<
            Component, Action, Result, Arguments, Derived
        > , naming::id_type const& gid);
}
namespace hpx
{
    
    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived, 
        typename Arg0>
    lcos::future<typename traits::promise_local_result<Result>::type, Result>
    async (
        hpx::actions::action<
            Component, Action, Result, Arguments, Derived
        > , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0);
}
namespace hpx
{
    
    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived, 
        typename Arg0 , typename Arg1>
    lcos::future<typename traits::promise_local_result<Result>::type, Result>
    async (
        hpx::actions::action<
            Component, Action, Result, Arguments, Derived
        > , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1);
}
namespace hpx
{
    
    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived, 
        typename Arg0 , typename Arg1 , typename Arg2>
    lcos::future<typename traits::promise_local_result<Result>::type, Result>
    async (
        hpx::actions::action<
            Component, Action, Result, Arguments, Derived
        > , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2);
}
namespace hpx
{
    
    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived, 
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    lcos::future<typename traits::promise_local_result<Result>::type, Result>
    async (
        hpx::actions::action<
            Component, Action, Result, Arguments, Derived
        > , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3);
}
namespace hpx
{
    
    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived, 
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    lcos::future<typename traits::promise_local_result<Result>::type, Result>
    async (
        hpx::actions::action<
            Component, Action, Result, Arguments, Derived
        > , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4);
}
