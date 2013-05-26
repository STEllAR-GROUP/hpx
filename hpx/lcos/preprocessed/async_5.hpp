// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx
{
    
    template <typename Action, typename Arg0>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type
    >
    async (BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid,
        BOOST_FWD_REF(Arg0) arg0)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::remote_result_type
        >::type result_type;
        typedef lcos::packaged_action<action_type, result_type>
            packaged_action_type;
        packaged_action_type p;
        if (policy == launch::sync || detail::has_async_policy(policy))
            p.apply(policy, gid, boost::forward<Arg0>( arg0 ));
        return p.get_future();
    }
    template <typename Action, typename Arg0>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type
    >
    async (naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0)
    {
        return async<Action>(launch::all, gid,
            boost::forward<Arg0>( arg0 ));
    }
    
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Arg0>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async (BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const & , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0)
    {
        return async<Derived>(policy, gid,
            boost::forward<Arg0>( arg0 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Arg0>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const &, naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0)
    {
        return async<Derived>(launch::all, gid,
            boost::forward<Arg0>( arg0 ));
    }
}
namespace hpx
{
    
    template <typename Action, typename Arg0 , typename Arg1>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type
    >
    async (BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid,
        BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::remote_result_type
        >::type result_type;
        typedef lcos::packaged_action<action_type, result_type>
            packaged_action_type;
        packaged_action_type p;
        if (policy == launch::sync || detail::has_async_policy(policy))
            p.apply(policy, gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
        return p.get_future();
    }
    template <typename Action, typename Arg0 , typename Arg1>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type
    >
    async (naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
    {
        return async<Action>(launch::all, gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
    
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Arg0 , typename Arg1>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async (BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const & , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
    {
        return async<Derived>(policy, gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Arg0 , typename Arg1>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const &, naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
    {
        return async<Derived>(launch::all, gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
}
namespace hpx
{
    
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type
    >
    async (BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid,
        BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::remote_result_type
        >::type result_type;
        typedef lcos::packaged_action<action_type, result_type>
            packaged_action_type;
        packaged_action_type p;
        if (policy == launch::sync || detail::has_async_policy(policy))
            p.apply(policy, gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
        return p.get_future();
    }
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type
    >
    async (naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
    {
        return async<Action>(launch::all, gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
    
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Arg0 , typename Arg1 , typename Arg2>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async (BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const & , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
    {
        return async<Derived>(policy, gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Arg0 , typename Arg1 , typename Arg2>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const &, naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
    {
        return async<Derived>(launch::all, gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
}
namespace hpx
{
    
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type
    >
    async (BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid,
        BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::remote_result_type
        >::type result_type;
        typedef lcos::packaged_action<action_type, result_type>
            packaged_action_type;
        packaged_action_type p;
        if (policy == launch::sync || detail::has_async_policy(policy))
            p.apply(policy, gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
        return p.get_future();
    }
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type
    >
    async (naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
    {
        return async<Action>(launch::all, gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
    
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async (BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const & , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
    {
        return async<Derived>(policy, gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const &, naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
    {
        return async<Derived>(launch::all, gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
}
namespace hpx
{
    
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type
    >
    async (BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid,
        BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::remote_result_type
        >::type result_type;
        typedef lcos::packaged_action<action_type, result_type>
            packaged_action_type;
        packaged_action_type p;
        if (policy == launch::sync || detail::has_async_policy(policy))
            p.apply(policy, gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
        return p.get_future();
    }
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type
    >
    async (naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
    {
        return async<Action>(launch::all, gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
    
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async (BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const & , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
    {
        return async<Derived>(policy, gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const &, naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
    {
        return async<Derived>(launch::all, gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
}
