// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx
{
    
    namespace detail
    {
        template <
            typename Action
          , typename RemoteResult
           
          , typename F>
        typename boost::enable_if_c<
            util::tuple_size<typename Action::arguments_type>::value == 0
          , lcos::future<
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            >
        >::type
        async_continue_r(
            naming::id_type const& gid
           
          , F && f);
    }
    
    template <
        typename Action
       
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 0
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        >
    >::type
    async_continue(
        naming::id_type const& gid
       
      , F && f);
    
    template <
        typename Component, typename Result, typename Arguments, typename Derived
       
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 0
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Derived, F>::type
            >::type
        >
    >::type
    async_continue(
        hpx::actions::action<Component, Result, Arguments, Derived> 
      , naming::id_type const& gid
       
      , F && f);
}
namespace hpx
{
    
    namespace detail
    {
        template <
            typename Action
          , typename RemoteResult
          , typename Arg0
          , typename F>
        typename boost::enable_if_c<
            util::tuple_size<typename Action::arguments_type>::value == 1
          , lcos::future<
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            >
        >::type
        async_continue_r(
            naming::id_type const& gid
          , Arg0 && arg0
          , F && f);
    }
    
    template <
        typename Action
      , typename Arg0
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 1
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        >
    >::type
    async_continue(
        naming::id_type const& gid
      , Arg0 && arg0
      , F && f);
    
    template <
        typename Component, typename Result, typename Arguments, typename Derived
      , typename Arg0
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 1
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Derived, F>::type
            >::type
        >
    >::type
    async_continue(
        hpx::actions::action<Component, Result, Arguments, Derived> 
      , naming::id_type const& gid
      , Arg0 && arg0
      , F && f);
}
namespace hpx
{
    
    namespace detail
    {
        template <
            typename Action
          , typename RemoteResult
          , typename Arg0 , typename Arg1
          , typename F>
        typename boost::enable_if_c<
            util::tuple_size<typename Action::arguments_type>::value == 2
          , lcos::future<
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            >
        >::type
        async_continue_r(
            naming::id_type const& gid
          , Arg0 && arg0 , Arg1 && arg1
          , F && f);
    }
    
    template <
        typename Action
      , typename Arg0 , typename Arg1
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 2
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        >
    >::type
    async_continue(
        naming::id_type const& gid
      , Arg0 && arg0 , Arg1 && arg1
      , F && f);
    
    template <
        typename Component, typename Result, typename Arguments, typename Derived
      , typename Arg0 , typename Arg1
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 2
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Derived, F>::type
            >::type
        >
    >::type
    async_continue(
        hpx::actions::action<Component, Result, Arguments, Derived> 
      , naming::id_type const& gid
      , Arg0 && arg0 , Arg1 && arg1
      , F && f);
}
namespace hpx
{
    
    namespace detail
    {
        template <
            typename Action
          , typename RemoteResult
          , typename Arg0 , typename Arg1 , typename Arg2
          , typename F>
        typename boost::enable_if_c<
            util::tuple_size<typename Action::arguments_type>::value == 3
          , lcos::future<
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            >
        >::type
        async_continue_r(
            naming::id_type const& gid
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2
          , F && f);
    }
    
    template <
        typename Action
      , typename Arg0 , typename Arg1 , typename Arg2
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 3
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        >
    >::type
    async_continue(
        naming::id_type const& gid
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2
      , F && f);
    
    template <
        typename Component, typename Result, typename Arguments, typename Derived
      , typename Arg0 , typename Arg1 , typename Arg2
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 3
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Derived, F>::type
            >::type
        >
    >::type
    async_continue(
        hpx::actions::action<Component, Result, Arguments, Derived> 
      , naming::id_type const& gid
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2
      , F && f);
}
namespace hpx
{
    
    namespace detail
    {
        template <
            typename Action
          , typename RemoteResult
          , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3
          , typename F>
        typename boost::enable_if_c<
            util::tuple_size<typename Action::arguments_type>::value == 4
          , lcos::future<
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            >
        >::type
        async_continue_r(
            naming::id_type const& gid
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3
          , F && f);
    }
    
    template <
        typename Action
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 4
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        >
    >::type
    async_continue(
        naming::id_type const& gid
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3
      , F && f);
    
    template <
        typename Component, typename Result, typename Arguments, typename Derived
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 4
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Derived, F>::type
            >::type
        >
    >::type
    async_continue(
        hpx::actions::action<Component, Result, Arguments, Derived> 
      , naming::id_type const& gid
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3
      , F && f);
}
namespace hpx
{
    
    namespace detail
    {
        template <
            typename Action
          , typename RemoteResult
          , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4
          , typename F>
        typename boost::enable_if_c<
            util::tuple_size<typename Action::arguments_type>::value == 5
          , lcos::future<
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            >
        >::type
        async_continue_r(
            naming::id_type const& gid
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4
          , F && f);
    }
    
    template <
        typename Action
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 5
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        >
    >::type
    async_continue(
        naming::id_type const& gid
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4
      , F && f);
    
    template <
        typename Component, typename Result, typename Arguments, typename Derived
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 5
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Derived, F>::type
            >::type
        >
    >::type
    async_continue(
        hpx::actions::action<Component, Result, Arguments, Derived> 
      , naming::id_type const& gid
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4
      , F && f);
}
