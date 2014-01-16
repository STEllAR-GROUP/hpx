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
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 1
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply<Action>(
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 1
            >::type
        >
        {
            typedef
                lcos::unique_future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::result_type
                    >::type
                >
                type;
            static BOOST_FORCEINLINE
            type call(
                BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::async<Action>(
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type>
        >
    >::type
    bind(T0 && t0)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ))
            );
    }
     
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 2
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply<Action>(
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 2
            >::type
        >
        {
            typedef
                lcos::unique_future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::result_type
                    >::type
                >
                type;
            static BOOST_FORCEINLINE
            type call(
                BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::async<Action>(
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0 , typename T1>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type>
        >
    >::type
    bind(T0 && t0 , T1 && t1)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ))
            );
    }
     
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0 , typename T1>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0 , T1 && t1)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 3
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply<Action>(
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 3
            >::type
        >
        {
            typedef
                lcos::unique_future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::result_type
                    >::type
                >
                type;
            static BOOST_FORCEINLINE
            type call(
                BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::async<Action>(
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0 , typename T1 , typename T2>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type>
        >
    >::type
    bind(T0 && t0 , T1 && t1 , T2 && t2)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ))
            );
    }
     
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0 , typename T1 , typename T2>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0 , T1 && t1 , T2 && t2)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 4
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply<Action>(
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 4
            >::type
        >
        {
            typedef
                lcos::unique_future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::result_type
                    >::type
                >
                type;
            static BOOST_FORCEINLINE
            type call(
                BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::async<Action>(
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0 , typename T1 , typename T2 , typename T3>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type>
        >
    >::type
    bind(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ))
            );
    }
     
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 5
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply<Action>(
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 5
            >::type
        >
        {
            typedef
                lcos::unique_future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::result_type
                    >::type
                >
                type;
            static BOOST_FORCEINLINE
            type call(
                BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::async<Action>(
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type>
        >
    >::type
    bind(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ))
            );
    }
     
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 6
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply<Action>(
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 6
            >::type
        >
        {
            typedef
                lcos::unique_future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::result_type
                    >::type
                >
                type;
            static BOOST_FORCEINLINE
            type call(
                BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::async<Action>(
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type>
        >
    >::type
    bind(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ))
            );
    }
     
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 7
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply<Action>(
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 7
            >::type
        >
        {
            typedef
                lcos::unique_future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::result_type
                    >::type
                >
                type;
            static BOOST_FORCEINLINE
            type call(
                BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::async<Action>(
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type>
        >
    >::type
    bind(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ))
            );
    }
     
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 8
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply<Action>(
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 8
            >::type
        >
        {
            typedef
                lcos::unique_future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::result_type
                    >::type
                >
                type;
            static BOOST_FORCEINLINE
            type call(
                BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::async<Action>(
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type>
        >
    >::type
    bind(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ))
            );
    }
     
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ))
            );
    }
}}
