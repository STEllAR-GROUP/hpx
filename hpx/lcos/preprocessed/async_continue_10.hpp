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
           
          , F && f)
        {
            typedef
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            result_type;
            typedef
                typename hpx::actions::extract_action<
                    Action
                >::result_type
            continuation_result_type;
            lcos::promise<result_type, RemoteResult> p;
            apply<Action>(
                new hpx::actions::typed_continuation<continuation_result_type>(
                    p.get_gid(), std::forward<F>(f))
              , gid
               );
            return p.get_future();
        }
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
       
      , F && f)
    {
        typedef
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        result_type;
        return detail::async_continue_r<Action, result_type>(
            gid
           
          , std::forward<F>(f));
    }
    
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
       
      , F && f)
    {
        return async_continue<Derived>(
            gid
           
          , std::forward<F>(f));
    }
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
          , F && f)
        {
            typedef
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            result_type;
            typedef
                typename hpx::actions::extract_action<
                    Action
                >::result_type
            continuation_result_type;
            lcos::promise<result_type, RemoteResult> p;
            apply<Action>(
                new hpx::actions::typed_continuation<continuation_result_type>(
                    p.get_gid(), std::forward<F>(f))
              , gid
              , std::forward<Arg0>( arg0 ));
            return p.get_future();
        }
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
      , F && f)
    {
        typedef
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        result_type;
        return detail::async_continue_r<Action, result_type>(
            gid
          , std::forward<Arg0>( arg0 )
          , std::forward<F>(f));
    }
    
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
      , F && f)
    {
        return async_continue<Derived>(
            gid
          , std::forward<Arg0>( arg0 )
          , std::forward<F>(f));
    }
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
          , F && f)
        {
            typedef
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            result_type;
            typedef
                typename hpx::actions::extract_action<
                    Action
                >::result_type
            continuation_result_type;
            lcos::promise<result_type, RemoteResult> p;
            apply<Action>(
                new hpx::actions::typed_continuation<continuation_result_type>(
                    p.get_gid(), std::forward<F>(f))
              , gid
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
            return p.get_future();
        }
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
      , F && f)
    {
        typedef
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        result_type;
        return detail::async_continue_r<Action, result_type>(
            gid
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 )
          , std::forward<F>(f));
    }
    
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
      , F && f)
    {
        return async_continue<Derived>(
            gid
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 )
          , std::forward<F>(f));
    }
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
          , F && f)
        {
            typedef
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            result_type;
            typedef
                typename hpx::actions::extract_action<
                    Action
                >::result_type
            continuation_result_type;
            lcos::promise<result_type, RemoteResult> p;
            apply<Action>(
                new hpx::actions::typed_continuation<continuation_result_type>(
                    p.get_gid(), std::forward<F>(f))
              , gid
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
            return p.get_future();
        }
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
      , F && f)
    {
        typedef
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        result_type;
        return detail::async_continue_r<Action, result_type>(
            gid
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 )
          , std::forward<F>(f));
    }
    
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
      , F && f)
    {
        return async_continue<Derived>(
            gid
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 )
          , std::forward<F>(f));
    }
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
          , F && f)
        {
            typedef
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            result_type;
            typedef
                typename hpx::actions::extract_action<
                    Action
                >::result_type
            continuation_result_type;
            lcos::promise<result_type, RemoteResult> p;
            apply<Action>(
                new hpx::actions::typed_continuation<continuation_result_type>(
                    p.get_gid(), std::forward<F>(f))
              , gid
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
            return p.get_future();
        }
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
      , F && f)
    {
        typedef
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        result_type;
        return detail::async_continue_r<Action, result_type>(
            gid
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 )
          , std::forward<F>(f));
    }
    
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
      , F && f)
    {
        return async_continue<Derived>(
            gid
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 )
          , std::forward<F>(f));
    }
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
          , F && f)
        {
            typedef
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            result_type;
            typedef
                typename hpx::actions::extract_action<
                    Action
                >::result_type
            continuation_result_type;
            lcos::promise<result_type, RemoteResult> p;
            apply<Action>(
                new hpx::actions::typed_continuation<continuation_result_type>(
                    p.get_gid(), std::forward<F>(f))
              , gid
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
            return p.get_future();
        }
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
      , F && f)
    {
        typedef
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        result_type;
        return detail::async_continue_r<Action, result_type>(
            gid
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 )
          , std::forward<F>(f));
    }
    
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
      , F && f)
    {
        return async_continue<Derived>(
            gid
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 )
          , std::forward<F>(f));
    }
}
namespace hpx
{
    
    namespace detail
    {
        template <
            typename Action
          , typename RemoteResult
          , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5
          , typename F>
        typename boost::enable_if_c<
            util::tuple_size<typename Action::arguments_type>::value == 6
          , lcos::future<
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            >
        >::type
        async_continue_r(
            naming::id_type const& gid
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5
          , F && f)
        {
            typedef
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            result_type;
            typedef
                typename hpx::actions::extract_action<
                    Action
                >::result_type
            continuation_result_type;
            lcos::promise<result_type, RemoteResult> p;
            apply<Action>(
                new hpx::actions::typed_continuation<continuation_result_type>(
                    p.get_gid(), std::forward<F>(f))
              , gid
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
            return p.get_future();
        }
    }
    
    template <
        typename Action
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 6
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        >
    >::type
    async_continue(
        naming::id_type const& gid
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5
      , F && f)
    {
        typedef
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        result_type;
        return detail::async_continue_r<Action, result_type>(
            gid
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 )
          , std::forward<F>(f));
    }
    
    template <
        typename Component, typename Result, typename Arguments, typename Derived
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 6
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Derived, F>::type
            >::type
        >
    >::type
    async_continue(
        hpx::actions::action<Component, Result, Arguments, Derived> 
      , naming::id_type const& gid
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5
      , F && f)
    {
        return async_continue<Derived>(
            gid
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 )
          , std::forward<F>(f));
    }
}
namespace hpx
{
    
    namespace detail
    {
        template <
            typename Action
          , typename RemoteResult
          , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6
          , typename F>
        typename boost::enable_if_c<
            util::tuple_size<typename Action::arguments_type>::value == 7
          , lcos::future<
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            >
        >::type
        async_continue_r(
            naming::id_type const& gid
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6
          , F && f)
        {
            typedef
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            result_type;
            typedef
                typename hpx::actions::extract_action<
                    Action
                >::result_type
            continuation_result_type;
            lcos::promise<result_type, RemoteResult> p;
            apply<Action>(
                new hpx::actions::typed_continuation<continuation_result_type>(
                    p.get_gid(), std::forward<F>(f))
              , gid
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
            return p.get_future();
        }
    }
    
    template <
        typename Action
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 7
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        >
    >::type
    async_continue(
        naming::id_type const& gid
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6
      , F && f)
    {
        typedef
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        result_type;
        return detail::async_continue_r<Action, result_type>(
            gid
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 )
          , std::forward<F>(f));
    }
    
    template <
        typename Component, typename Result, typename Arguments, typename Derived
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 7
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Derived, F>::type
            >::type
        >
    >::type
    async_continue(
        hpx::actions::action<Component, Result, Arguments, Derived> 
      , naming::id_type const& gid
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6
      , F && f)
    {
        return async_continue<Derived>(
            gid
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 )
          , std::forward<F>(f));
    }
}
namespace hpx
{
    
    namespace detail
    {
        template <
            typename Action
          , typename RemoteResult
          , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7
          , typename F>
        typename boost::enable_if_c<
            util::tuple_size<typename Action::arguments_type>::value == 8
          , lcos::future<
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            >
        >::type
        async_continue_r(
            naming::id_type const& gid
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7
          , F && f)
        {
            typedef
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            result_type;
            typedef
                typename hpx::actions::extract_action<
                    Action
                >::result_type
            continuation_result_type;
            lcos::promise<result_type, RemoteResult> p;
            apply<Action>(
                new hpx::actions::typed_continuation<continuation_result_type>(
                    p.get_gid(), std::forward<F>(f))
              , gid
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
            return p.get_future();
        }
    }
    
    template <
        typename Action
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 8
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        >
    >::type
    async_continue(
        naming::id_type const& gid
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7
      , F && f)
    {
        typedef
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        result_type;
        return detail::async_continue_r<Action, result_type>(
            gid
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 )
          , std::forward<F>(f));
    }
    
    template <
        typename Component, typename Result, typename Arguments, typename Derived
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 8
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Derived, F>::type
            >::type
        >
    >::type
    async_continue(
        hpx::actions::action<Component, Result, Arguments, Derived> 
      , naming::id_type const& gid
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7
      , F && f)
    {
        return async_continue<Derived>(
            gid
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 )
          , std::forward<F>(f));
    }
}
namespace hpx
{
    
    namespace detail
    {
        template <
            typename Action
          , typename RemoteResult
          , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8
          , typename F>
        typename boost::enable_if_c<
            util::tuple_size<typename Action::arguments_type>::value == 9
          , lcos::future<
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            >
        >::type
        async_continue_r(
            naming::id_type const& gid
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8
          , F && f)
        {
            typedef
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            result_type;
            typedef
                typename hpx::actions::extract_action<
                    Action
                >::result_type
            continuation_result_type;
            lcos::promise<result_type, RemoteResult> p;
            apply<Action>(
                new hpx::actions::typed_continuation<continuation_result_type>(
                    p.get_gid(), std::forward<F>(f))
              , gid
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
            return p.get_future();
        }
    }
    
    template <
        typename Action
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 9
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        >
    >::type
    async_continue(
        naming::id_type const& gid
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8
      , F && f)
    {
        typedef
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        result_type;
        return detail::async_continue_r<Action, result_type>(
            gid
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 )
          , std::forward<F>(f));
    }
    
    template <
        typename Component, typename Result, typename Arguments, typename Derived
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 9
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Derived, F>::type
            >::type
        >
    >::type
    async_continue(
        hpx::actions::action<Component, Result, Arguments, Derived> 
      , naming::id_type const& gid
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8
      , F && f)
    {
        return async_continue<Derived>(
            gid
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 )
          , std::forward<F>(f));
    }
}
namespace hpx
{
    
    namespace detail
    {
        template <
            typename Action
          , typename RemoteResult
          , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9
          , typename F>
        typename boost::enable_if_c<
            util::tuple_size<typename Action::arguments_type>::value == 10
          , lcos::future<
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            >
        >::type
        async_continue_r(
            naming::id_type const& gid
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9
          , F && f)
        {
            typedef
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            result_type;
            typedef
                typename hpx::actions::extract_action<
                    Action
                >::result_type
            continuation_result_type;
            lcos::promise<result_type, RemoteResult> p;
            apply<Action>(
                new hpx::actions::typed_continuation<continuation_result_type>(
                    p.get_gid(), std::forward<F>(f))
              , gid
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
            return p.get_future();
        }
    }
    
    template <
        typename Action
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 10
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        >
    >::type
    async_continue(
        naming::id_type const& gid
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9
      , F && f)
    {
        typedef
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        result_type;
        return detail::async_continue_r<Action, result_type>(
            gid
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 )
          , std::forward<F>(f));
    }
    
    template <
        typename Component, typename Result, typename Arguments, typename Derived
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 10
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Derived, F>::type
            >::type
        >
    >::type
    async_continue(
        hpx::actions::action<Component, Result, Arguments, Derived> 
      , naming::id_type const& gid
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9
      , F && f)
    {
        return async_continue<Derived>(
            gid
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 )
          , std::forward<F>(f));
    }
}
