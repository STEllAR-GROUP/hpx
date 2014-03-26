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
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 1
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
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
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 2
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
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
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 3
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
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
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 4
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
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
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 5
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
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
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 6
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
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
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 7
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
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
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 8
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
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
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 9
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 9
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 9
            >::type
        >
        {
            typedef
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type>
        >
    >::type
    bind(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ))
            );
    }
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ))
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
                util::tuple_size<BoundArgs>::value == 10
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 10
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 10
            >::type
        >
        {
            typedef
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type>
        >
    >::type
    bind(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ))
            );
    }
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ))
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
                util::tuple_size<BoundArgs>::value == 11
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 11
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 11
            >::type
        >
        {
            typedef
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type>
        >
    >::type
    bind(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ))
            );
    }
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ))
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
                util::tuple_size<BoundArgs>::value == 12
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 12
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 12
            >::type
        >
        {
            typedef
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type>
        >
    >::type
    bind(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ))
            );
    }
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ))
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
                util::tuple_size<BoundArgs>::value == 13
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 13
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 13
            >::type
        >
        {
            typedef
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type>
        >
    >::type
    bind(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ))
            );
    }
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ))
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
                util::tuple_size<BoundArgs>::value == 14
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 14
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 14
            >::type
        >
        {
            typedef
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type>
        >
    >::type
    bind(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ))
            );
    }
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ))
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
                util::tuple_size<BoundArgs>::value == 15
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 15
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 15
            >::type
        >
        {
            typedef
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type>
        >
    >::type
    bind(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ) , std::forward<T14>( t14 ))
            );
    }
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ) , std::forward<T14>( t14 ))
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
                util::tuple_size<BoundArgs>::value == 16
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 16
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 16
            >::type
        >
        {
            typedef
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type>
        >
    >::type
    bind(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14 , T15 && t15)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ) , std::forward<T14>( t14 ) , std::forward<T15>( t15 ))
            );
    }
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14 , T15 && t15)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ) , std::forward<T14>( t14 ) , std::forward<T15>( t15 ))
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
                util::tuple_size<BoundArgs>::value == 17
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 16>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 17
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 16>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 17
            >::type
        >
        {
            typedef
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 16>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type>
        >
    >::type
    bind(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14 , T15 && t15 , T16 && t16)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ) , std::forward<T14>( t14 ) , std::forward<T15>( t15 ) , std::forward<T16>( t16 ))
            );
    }
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14 , T15 && t15 , T16 && t16)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ) , std::forward<T14>( t14 ) , std::forward<T15>( t15 ) , std::forward<T16>( t16 ))
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
                util::tuple_size<BoundArgs>::value == 18
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 16>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 17>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 18
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 16>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 17>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 18
            >::type
        >
        {
            typedef
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 16>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 17>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type>
        >
    >::type
    bind(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14 , T15 && t15 , T16 && t16 , T17 && t17)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ) , std::forward<T14>( t14 ) , std::forward<T15>( t15 ) , std::forward<T16>( t16 ) , std::forward<T17>( t17 ))
            );
    }
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14 , T15 && t15 , T16 && t16 , T17 && t17)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ) , std::forward<T14>( t14 ) , std::forward<T15>( t15 ) , std::forward<T16>( t16 ) , std::forward<T17>( t17 ))
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
                util::tuple_size<BoundArgs>::value == 19
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 16>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 17>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 18>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 19
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 16>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 17>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 18>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 19
            >::type
        >
        {
            typedef
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 16>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 17>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 18>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type>
        >
    >::type
    bind(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14 , T15 && t15 , T16 && t16 , T17 && t17 , T18 && t18)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ) , std::forward<T14>( t14 ) , std::forward<T15>( t15 ) , std::forward<T16>( t16 ) , std::forward<T17>( t17 ) , std::forward<T18>( t18 ))
            );
    }
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14 , T15 && t15 , T16 && t16 , T17 && t17 , T18 && t18)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ) , std::forward<T14>( t14 ) , std::forward<T15>( t15 ) , std::forward<T16>( t16 ) , std::forward<T17>( t17 ) , std::forward<T18>( t18 ))
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
                util::tuple_size<BoundArgs>::value == 20
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 16>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 17>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 18>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 19>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 20
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 16>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 17>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 18>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 19>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 20
            >::type
        >
        {
            typedef
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 16>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 17>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 18>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 19>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type>
        >
    >::type
    bind(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14 , T15 && t15 , T16 && t16 , T17 && t17 , T18 && t18 , T19 && t19)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ) , std::forward<T14>( t14 ) , std::forward<T15>( t15 ) , std::forward<T16>( t16 ) , std::forward<T17>( t17 ) , std::forward<T18>( t18 ) , std::forward<T19>( t19 ))
            );
    }
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14 , T15 && t15 , T16 && t16 , T17 && t17 , T18 && t18 , T19 && t19)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ) , std::forward<T14>( t14 ) , std::forward<T15>( t15 ) , std::forward<T16>( t16 ) , std::forward<T17>( t17 ) , std::forward<T18>( t18 ) , std::forward<T19>( t19 ))
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
                util::tuple_size<BoundArgs>::value == 21
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 16>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 17>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 18>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 19>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 20>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 21
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 16>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 17>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 18>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 19>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 20>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 21
            >::type
        >
        {
            typedef
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 16>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 17>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 18>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 19>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 20>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type>
        >
    >::type
    bind(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14 , T15 && t15 , T16 && t16 , T17 && t17 , T18 && t18 , T19 && t19 , T20 && t20)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ) , std::forward<T14>( t14 ) , std::forward<T15>( t15 ) , std::forward<T16>( t16 ) , std::forward<T17>( t17 ) , std::forward<T18>( t18 ) , std::forward<T19>( t19 ) , std::forward<T20>( t20 ))
            );
    }
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14 , T15 && t15 , T16 && t16 , T17 && t17 , T18 && t18 , T19 && t19 , T20 && t20)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ) , std::forward<T14>( t14 ) , std::forward<T15>( t15 ) , std::forward<T16>( t16 ) , std::forward<T17>( t17 ) , std::forward<T18>( t18 ) , std::forward<T19>( t19 ) , std::forward<T20>( t20 ))
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
                util::tuple_size<BoundArgs>::value == 22
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 16>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 17>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 18>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 19>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 20>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 21>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 22
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 16>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 17>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 18>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 19>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 20>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 21>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 22
            >::type
        >
        {
            typedef
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 16>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 17>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 18>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 19>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 20>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 21>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20 , typename T21>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type , typename util::decay<T21>::type>
        >
    >::type
    bind(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14 , T15 && t15 , T16 && t16 , T17 && t17 , T18 && t18 , T19 && t19 , T20 && t20 , T21 && t21)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type , typename util::decay<T21>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ) , std::forward<T14>( t14 ) , std::forward<T15>( t15 ) , std::forward<T16>( t16 ) , std::forward<T17>( t17 ) , std::forward<T18>( t18 ) , std::forward<T19>( t19 ) , std::forward<T20>( t20 ) , std::forward<T21>( t21 ))
            );
    }
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20 , typename T21>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type , typename util::decay<T21>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14 , T15 && t15 , T16 && t16 , T17 && t17 , T18 && t18 , T19 && t19 , T20 && t20 , T21 && t21)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type , typename util::decay<T21>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ) , std::forward<T14>( t14 ) , std::forward<T15>( t15 ) , std::forward<T16>( t16 ) , std::forward<T17>( t17 ) , std::forward<T18>( t18 ) , std::forward<T19>( t19 ) , std::forward<T20>( t20 ) , std::forward<T21>( t21 ))
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
                util::tuple_size<BoundArgs>::value == 23
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 16>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 17>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 18>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 19>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 20>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 21>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 22>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_apply_cont_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 23
            >::type
        >
        {
            typedef bool type;
            static BOOST_FORCEINLINE
            type call(
                naming::id_type const& contgid
              , BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return
                    hpx::apply_c<Action>(
                        contgid, detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 16>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 17>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 18>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 19>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 20>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 21>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 22>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
        template <typename Action, typename BoundArgs, typename UnboundArgs>
        struct bind_action_async_impl<
            Action, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 23
            >::type
        >
        {
            typedef
                lcos::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
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
                        detail::bind_eval<Action>( util::get< 0>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 1>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 2>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 3>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 4>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 5>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 6>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 7>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 8>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 9>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 10>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 11>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 12>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 13>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 14>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 15>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 16>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 17>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 18>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 19>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 20>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 21>(bound_args) , std::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<Action>( util::get< 22>(bound_args) , std::forward<UnboundArgs>(unbound_args) )
                    );
            }
        };
    }
    template <typename Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20 , typename T21 , typename T22>
    typename boost::enable_if_c<
        traits::is_action<typename boost::remove_reference<Action>::type>::value
      , detail::bound_action<
            typename util::decay<Action>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type , typename util::decay<T21>::type , typename util::decay<T22>::type>
        >
    >::type
    bind(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14 , T15 && t15 , T16 && t16 , T17 && t17 , T18 && t18 , T19 && t19 , T20 && t20 , T21 && t21 , T22 && t22)
    {
        typedef
            detail::bound_action<
                typename util::decay<Action>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type , typename util::decay<T21>::type , typename util::decay<T22>::type>
            >
            result_type;
        return
            result_type(
                Action()
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ) , std::forward<T14>( t14 ) , std::forward<T15>( t15 ) , std::forward<T16>( t16 ) , std::forward<T17>( t17 ) , std::forward<T18>( t18 ) , std::forward<T19>( t19 ) , std::forward<T20>( t20 ) , std::forward<T21>( t21 ) , std::forward<T22>( t22 ))
            );
    }
    template <
        typename Component, typename Result, typename Arguments
      , typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20 , typename T21 , typename T22>
    detail::bound_action<
        Derived
      , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type , typename util::decay<T21>::type , typename util::decay<T22>::type>
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > action, T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14 , T15 && t15 , T16 && t16 , T17 && t17 , T18 && t18 , T19 && t19 , T20 && t20 , T21 && t21 , T22 && t22)
    {
        typedef
            detail::bound_action<
                Derived
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type , typename util::decay<T21>::type , typename util::decay<T22>::type>
            >
            result_type;
        return
            result_type(
                static_cast<Derived const&>(action)
              , util::forward_as_tuple(std::forward<T0>( t0 ) , std::forward<T1>( t1 ) , std::forward<T2>( t2 ) , std::forward<T3>( t3 ) , std::forward<T4>( t4 ) , std::forward<T5>( t5 ) , std::forward<T6>( t6 ) , std::forward<T7>( t7 ) , std::forward<T8>( t8 ) , std::forward<T9>( t9 ) , std::forward<T10>( t10 ) , std::forward<T11>( t11 ) , std::forward<T12>( t12 ) , std::forward<T13>( t13 ) , std::forward<T14>( t14 ) , std::forward<T15>( t15 ) , std::forward<T16>( t16 ) , std::forward<T17>( t17 ) , std::forward<T18>( t18 ) , std::forward<T19>( t19 ) , std::forward<T20>( t20 ) , std::forward<T21>( t21 ) , std::forward<T22>( t22 ))
            );
    }
}}
