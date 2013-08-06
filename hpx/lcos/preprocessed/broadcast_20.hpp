// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
           
        >
        hpx::future<void>
        broadcast_impl0(
            Action const & act
          , std::vector<hpx::id_type> const & ids
           
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return hpx::lcos::make_ready_future();
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                   
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                       
                      , boost::integral_constant<bool, true>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                           
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).then(&return_void);
        }
        template <
            typename Action
           
        >
        hpx::future<typename broadcast_result<Action>::type>
        broadcast_impl0(
            Action const & act
          , std::vector<hpx::id_type> const & ids
           
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            if(ids.empty()) return hpx::lcos::make_ready_future(result_type());
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                   
                )
                .then(
                    &wrap_into_vector<action_result>
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                       
                      , boost::integral_constant<bool, false>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                           
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>);
        }
        
        template <
            typename Action
           
          , typename IsVoid
        >
        struct broadcast_invoker0
        {
            static hpx::future<typename broadcast_result<Action>::type>
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
               
              , IsVoid
            )
            {
                return
                    broadcast_impl0(
                        act
                      , ids
                       
                      , IsVoid()
                    );
            }
        };
        template <
            typename Action
        >
        struct make_broadcast_action_impl<Action, 0>
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef broadcast_invoker0<
                        Action
                      
                        
                      , typename boost::is_same<void, action_result>::type
                    >
                    broadcast_invoker_type;
            typedef
                typename hpx::actions::make_action<decltype(&broadcast_invoker_type::call), &broadcast_invoker_type::call>::type
                type;
        };
    }
    
    template <
        typename Action
       
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast(
        std::vector<hpx::id_type> const & ids
       )
    {
        hpx::id_type dest = hpx::get_colocation_id(ids[0]);
        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , ids
                 
              , typename boost::is_same<void, action_result>::type()
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
       
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast(
            hpx::actions::action<
                Component, Result, Arguments, Derived
            > 
          , std::vector<hpx::id_type> const & ids
           )
    {
        return broadcast<Derived>(
                ids
               
            );
    }
}}
namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
          , typename A0
        >
        hpx::future<void>
        broadcast_impl1(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return hpx::lcos::make_ready_future();
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0
                      , boost::integral_constant<bool, true>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).then(&return_void);
        }
        template <
            typename Action
          , typename A0
        >
        hpx::future<typename broadcast_result<Action>::type>
        broadcast_impl1(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            if(ids.empty()) return hpx::lcos::make_ready_future(result_type());
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0
                )
                .then(
                    &wrap_into_vector<action_result>
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0
                      , boost::integral_constant<bool, false>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>);
        }
        
        template <
            typename Action
          , typename A0
          , typename IsVoid
        >
        struct broadcast_invoker1
        {
            static hpx::future<typename broadcast_result<Action>::type>
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0
              , IsVoid
            )
            {
                return
                    broadcast_impl1(
                        act
                      , ids
                      , a0
                      , IsVoid()
                    );
            }
        };
        template <
            typename Action
        >
        struct make_broadcast_action_impl<Action, 1>
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef broadcast_invoker1<
                        Action
                      ,
                        typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type
                      , typename boost::is_same<void, action_result>::type
                    >
                    broadcast_invoker_type;
            typedef
                typename hpx::actions::make_action<decltype(&broadcast_invoker_type::call), &broadcast_invoker_type::call>::type
                type;
        };
    }
    
    template <
        typename Action
      , typename A0
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0)
    {
        hpx::id_type dest = hpx::get_colocation_id(ids[0]);
        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , ids
                , a0
              , typename boost::is_same<void, action_result>::type()
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename A0
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast(
            hpx::actions::action<
                Component, Result, Arguments, Derived
            > 
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0)
    {
        return broadcast<Derived>(
                ids
              , a0
            );
    }
}}
namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
          , typename A0 , typename A1
        >
        hpx::future<void>
        broadcast_impl2(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return hpx::lcos::make_ready_future();
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1
                      , boost::integral_constant<bool, true>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).then(&return_void);
        }
        template <
            typename Action
          , typename A0 , typename A1
        >
        hpx::future<typename broadcast_result<Action>::type>
        broadcast_impl2(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            if(ids.empty()) return hpx::lcos::make_ready_future(result_type());
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1
                )
                .then(
                    &wrap_into_vector<action_result>
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1
                      , boost::integral_constant<bool, false>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>);
        }
        
        template <
            typename Action
          , typename A0 , typename A1
          , typename IsVoid
        >
        struct broadcast_invoker2
        {
            static hpx::future<typename broadcast_result<Action>::type>
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1
              , IsVoid
            )
            {
                return
                    broadcast_impl2(
                        act
                      , ids
                      , a0 , a1
                      , IsVoid()
                    );
            }
        };
        template <
            typename Action
        >
        struct make_broadcast_action_impl<Action, 2>
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef broadcast_invoker2<
                        Action
                      ,
                        typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type
                      , typename boost::is_same<void, action_result>::type
                    >
                    broadcast_invoker_type;
            typedef
                typename hpx::actions::make_action<decltype(&broadcast_invoker_type::call), &broadcast_invoker_type::call>::type
                type;
        };
    }
    
    template <
        typename Action
      , typename A0 , typename A1
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1)
    {
        hpx::id_type dest = hpx::get_colocation_id(ids[0]);
        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , ids
                , a0 , a1
              , typename boost::is_same<void, action_result>::type()
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename A0 , typename A1
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast(
            hpx::actions::action<
                Component, Result, Arguments, Derived
            > 
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1)
    {
        return broadcast<Derived>(
                ids
              , a0 , a1
            );
    }
}}
namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
          , typename A0 , typename A1 , typename A2
        >
        hpx::future<void>
        broadcast_impl3(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return hpx::lcos::make_ready_future();
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2
                      , boost::integral_constant<bool, true>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).then(&return_void);
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2
        >
        hpx::future<typename broadcast_result<Action>::type>
        broadcast_impl3(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            if(ids.empty()) return hpx::lcos::make_ready_future(result_type());
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2
                )
                .then(
                    &wrap_into_vector<action_result>
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2
                      , boost::integral_constant<bool, false>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>);
        }
        
        template <
            typename Action
          , typename A0 , typename A1 , typename A2
          , typename IsVoid
        >
        struct broadcast_invoker3
        {
            static hpx::future<typename broadcast_result<Action>::type>
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1 , A2 const & a2
              , IsVoid
            )
            {
                return
                    broadcast_impl3(
                        act
                      , ids
                      , a0 , a1 , a2
                      , IsVoid()
                    );
            }
        };
        template <
            typename Action
        >
        struct make_broadcast_action_impl<Action, 3>
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef broadcast_invoker3<
                        Action
                      ,
                        typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type
                      , typename boost::is_same<void, action_result>::type
                    >
                    broadcast_invoker_type;
            typedef
                typename hpx::actions::make_action<decltype(&broadcast_invoker_type::call), &broadcast_invoker_type::call>::type
                type;
        };
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2)
    {
        hpx::id_type dest = hpx::get_colocation_id(ids[0]);
        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , ids
                , a0 , a1 , a2
              , typename boost::is_same<void, action_result>::type()
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename A0 , typename A1 , typename A2
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast(
            hpx::actions::action<
                Component, Result, Arguments, Derived
            > 
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2)
    {
        return broadcast<Derived>(
                ids
              , a0 , a1 , a2
            );
    }
}}
namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3
        >
        hpx::future<void>
        broadcast_impl4(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return hpx::lcos::make_ready_future();
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3
                      , boost::integral_constant<bool, true>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).then(&return_void);
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3
        >
        hpx::future<typename broadcast_result<Action>::type>
        broadcast_impl4(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            if(ids.empty()) return hpx::lcos::make_ready_future(result_type());
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3
                )
                .then(
                    &wrap_into_vector<action_result>
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3
                      , boost::integral_constant<bool, false>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>);
        }
        
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3
          , typename IsVoid
        >
        struct broadcast_invoker4
        {
            static hpx::future<typename broadcast_result<Action>::type>
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3
              , IsVoid
            )
            {
                return
                    broadcast_impl4(
                        act
                      , ids
                      , a0 , a1 , a2 , a3
                      , IsVoid()
                    );
            }
        };
        template <
            typename Action
        >
        struct make_broadcast_action_impl<Action, 4>
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef broadcast_invoker4<
                        Action
                      ,
                        typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type
                      , typename boost::is_same<void, action_result>::type
                    >
                    broadcast_invoker_type;
            typedef
                typename hpx::actions::make_action<decltype(&broadcast_invoker_type::call), &broadcast_invoker_type::call>::type
                type;
        };
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3)
    {
        hpx::id_type dest = hpx::get_colocation_id(ids[0]);
        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , ids
                , a0 , a1 , a2 , a3
              , typename boost::is_same<void, action_result>::type()
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename A0 , typename A1 , typename A2 , typename A3
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast(
            hpx::actions::action<
                Component, Result, Arguments, Derived
            > 
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3)
    {
        return broadcast<Derived>(
                ids
              , a0 , a1 , a2 , a3
            );
    }
}}
namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
        >
        hpx::future<void>
        broadcast_impl5(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return hpx::lcos::make_ready_future();
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4
                      , boost::integral_constant<bool, true>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).then(&return_void);
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
        >
        hpx::future<typename broadcast_result<Action>::type>
        broadcast_impl5(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            if(ids.empty()) return hpx::lcos::make_ready_future(result_type());
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4
                )
                .then(
                    &wrap_into_vector<action_result>
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4
                      , boost::integral_constant<bool, false>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>);
        }
        
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
          , typename IsVoid
        >
        struct broadcast_invoker5
        {
            static hpx::future<typename broadcast_result<Action>::type>
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4
              , IsVoid
            )
            {
                return
                    broadcast_impl5(
                        act
                      , ids
                      , a0 , a1 , a2 , a3 , a4
                      , IsVoid()
                    );
            }
        };
        template <
            typename Action
        >
        struct make_broadcast_action_impl<Action, 5>
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef broadcast_invoker5<
                        Action
                      ,
                        typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type
                      , typename boost::is_same<void, action_result>::type
                    >
                    broadcast_invoker_type;
            typedef
                typename hpx::actions::make_action<decltype(&broadcast_invoker_type::call), &broadcast_invoker_type::call>::type
                type;
        };
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4)
    {
        hpx::id_type dest = hpx::get_colocation_id(ids[0]);
        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , ids
                , a0 , a1 , a2 , a3 , a4
              , typename boost::is_same<void, action_result>::type()
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast(
            hpx::actions::action<
                Component, Result, Arguments, Derived
            > 
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4)
    {
        return broadcast<Derived>(
                ids
              , a0 , a1 , a2 , a3 , a4
            );
    }
}}
namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
        >
        hpx::future<void>
        broadcast_impl6(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return hpx::lcos::make_ready_future();
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5
                      , boost::integral_constant<bool, true>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).then(&return_void);
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
        >
        hpx::future<typename broadcast_result<Action>::type>
        broadcast_impl6(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            if(ids.empty()) return hpx::lcos::make_ready_future(result_type());
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5
                )
                .then(
                    &wrap_into_vector<action_result>
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5
                      , boost::integral_constant<bool, false>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>);
        }
        
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
          , typename IsVoid
        >
        struct broadcast_invoker6
        {
            static hpx::future<typename broadcast_result<Action>::type>
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5
              , IsVoid
            )
            {
                return
                    broadcast_impl6(
                        act
                      , ids
                      , a0 , a1 , a2 , a3 , a4 , a5
                      , IsVoid()
                    );
            }
        };
        template <
            typename Action
        >
        struct make_broadcast_action_impl<Action, 6>
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef broadcast_invoker6<
                        Action
                      ,
                        typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type
                      , typename boost::is_same<void, action_result>::type
                    >
                    broadcast_invoker_type;
            typedef
                typename hpx::actions::make_action<decltype(&broadcast_invoker_type::call), &broadcast_invoker_type::call>::type
                type;
        };
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5)
    {
        hpx::id_type dest = hpx::get_colocation_id(ids[0]);
        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , ids
                , a0 , a1 , a2 , a3 , a4 , a5
              , typename boost::is_same<void, action_result>::type()
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast(
            hpx::actions::action<
                Component, Result, Arguments, Derived
            > 
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5)
    {
        return broadcast<Derived>(
                ids
              , a0 , a1 , a2 , a3 , a4 , a5
            );
    }
}}
namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
        >
        hpx::future<void>
        broadcast_impl7(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return hpx::lcos::make_ready_future();
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6
                      , boost::integral_constant<bool, true>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).then(&return_void);
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
        >
        hpx::future<typename broadcast_result<Action>::type>
        broadcast_impl7(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            if(ids.empty()) return hpx::lcos::make_ready_future(result_type());
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6
                )
                .then(
                    &wrap_into_vector<action_result>
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6
                      , boost::integral_constant<bool, false>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>);
        }
        
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
          , typename IsVoid
        >
        struct broadcast_invoker7
        {
            static hpx::future<typename broadcast_result<Action>::type>
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6
              , IsVoid
            )
            {
                return
                    broadcast_impl7(
                        act
                      , ids
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6
                      , IsVoid()
                    );
            }
        };
        template <
            typename Action
        >
        struct make_broadcast_action_impl<Action, 7>
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef broadcast_invoker7<
                        Action
                      ,
                        typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type
                      , typename boost::is_same<void, action_result>::type
                    >
                    broadcast_invoker_type;
            typedef
                typename hpx::actions::make_action<decltype(&broadcast_invoker_type::call), &broadcast_invoker_type::call>::type
                type;
        };
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6)
    {
        hpx::id_type dest = hpx::get_colocation_id(ids[0]);
        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , ids
                , a0 , a1 , a2 , a3 , a4 , a5 , a6
              , typename boost::is_same<void, action_result>::type()
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast(
            hpx::actions::action<
                Component, Result, Arguments, Derived
            > 
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6)
    {
        return broadcast<Derived>(
                ids
              , a0 , a1 , a2 , a3 , a4 , a5 , a6
            );
    }
}}
namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
        >
        hpx::future<void>
        broadcast_impl8(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return hpx::lcos::make_ready_future();
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7
                      , boost::integral_constant<bool, true>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).then(&return_void);
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
        >
        hpx::future<typename broadcast_result<Action>::type>
        broadcast_impl8(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            if(ids.empty()) return hpx::lcos::make_ready_future(result_type());
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7
                )
                .then(
                    &wrap_into_vector<action_result>
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7
                      , boost::integral_constant<bool, false>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>);
        }
        
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
          , typename IsVoid
        >
        struct broadcast_invoker8
        {
            static hpx::future<typename broadcast_result<Action>::type>
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7
              , IsVoid
            )
            {
                return
                    broadcast_impl8(
                        act
                      , ids
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7
                      , IsVoid()
                    );
            }
        };
        template <
            typename Action
        >
        struct make_broadcast_action_impl<Action, 8>
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef broadcast_invoker8<
                        Action
                      ,
                        typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 7 >::type
                      , typename boost::is_same<void, action_result>::type
                    >
                    broadcast_invoker_type;
            typedef
                typename hpx::actions::make_action<decltype(&broadcast_invoker_type::call), &broadcast_invoker_type::call>::type
                type;
        };
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7)
    {
        hpx::id_type dest = hpx::get_colocation_id(ids[0]);
        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , ids
                , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7
              , typename boost::is_same<void, action_result>::type()
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast(
            hpx::actions::action<
                Component, Result, Arguments, Derived
            > 
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7)
    {
        return broadcast<Derived>(
                ids
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7
            );
    }
}}
namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
        >
        hpx::future<void>
        broadcast_impl9(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return hpx::lcos::make_ready_future();
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8
                      , boost::integral_constant<bool, true>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).then(&return_void);
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
        >
        hpx::future<typename broadcast_result<Action>::type>
        broadcast_impl9(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            if(ids.empty()) return hpx::lcos::make_ready_future(result_type());
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8
                )
                .then(
                    &wrap_into_vector<action_result>
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8
                      , boost::integral_constant<bool, false>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>);
        }
        
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
          , typename IsVoid
        >
        struct broadcast_invoker9
        {
            static hpx::future<typename broadcast_result<Action>::type>
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8
              , IsVoid
            )
            {
                return
                    broadcast_impl9(
                        act
                      , ids
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8
                      , IsVoid()
                    );
            }
        };
        template <
            typename Action
        >
        struct make_broadcast_action_impl<Action, 9>
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef broadcast_invoker9<
                        Action
                      ,
                        typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 7 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 8 >::type
                      , typename boost::is_same<void, action_result>::type
                    >
                    broadcast_invoker_type;
            typedef
                typename hpx::actions::make_action<decltype(&broadcast_invoker_type::call), &broadcast_invoker_type::call>::type
                type;
        };
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8)
    {
        hpx::id_type dest = hpx::get_colocation_id(ids[0]);
        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , ids
                , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8
              , typename boost::is_same<void, action_result>::type()
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast(
            hpx::actions::action<
                Component, Result, Arguments, Derived
            > 
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8)
    {
        return broadcast<Derived>(
                ids
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8
            );
    }
}}
namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
        >
        hpx::future<void>
        broadcast_impl10(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return hpx::lcos::make_ready_future();
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9
                      , boost::integral_constant<bool, true>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).then(&return_void);
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
        >
        hpx::future<typename broadcast_result<Action>::type>
        broadcast_impl10(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            if(ids.empty()) return hpx::lcos::make_ready_future(result_type());
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9
                )
                .then(
                    &wrap_into_vector<action_result>
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9
                      , boost::integral_constant<bool, false>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>);
        }
        
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
          , typename IsVoid
        >
        struct broadcast_invoker10
        {
            static hpx::future<typename broadcast_result<Action>::type>
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9
              , IsVoid
            )
            {
                return
                    broadcast_impl10(
                        act
                      , ids
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9
                      , IsVoid()
                    );
            }
        };
        template <
            typename Action
        >
        struct make_broadcast_action_impl<Action, 10>
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef broadcast_invoker10<
                        Action
                      ,
                        typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 7 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 8 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 9 >::type
                      , typename boost::is_same<void, action_result>::type
                    >
                    broadcast_invoker_type;
            typedef
                typename hpx::actions::make_action<decltype(&broadcast_invoker_type::call), &broadcast_invoker_type::call>::type
                type;
        };
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9)
    {
        hpx::id_type dest = hpx::get_colocation_id(ids[0]);
        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , ids
                , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9
              , typename boost::is_same<void, action_result>::type()
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast(
            hpx::actions::action<
                Component, Result, Arguments, Derived
            > 
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9)
    {
        return broadcast<Derived>(
                ids
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9
            );
    }
}}
namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
        >
        hpx::future<void>
        broadcast_impl11(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return hpx::lcos::make_ready_future();
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10
                      , boost::integral_constant<bool, true>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).then(&return_void);
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
        >
        hpx::future<typename broadcast_result<Action>::type>
        broadcast_impl11(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            if(ids.empty()) return hpx::lcos::make_ready_future(result_type());
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10
                )
                .then(
                    &wrap_into_vector<action_result>
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10
                      , boost::integral_constant<bool, false>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>);
        }
        
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
          , typename IsVoid
        >
        struct broadcast_invoker11
        {
            static hpx::future<typename broadcast_result<Action>::type>
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10
              , IsVoid
            )
            {
                return
                    broadcast_impl11(
                        act
                      , ids
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10
                      , IsVoid()
                    );
            }
        };
        template <
            typename Action
        >
        struct make_broadcast_action_impl<Action, 11>
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef broadcast_invoker11<
                        Action
                      ,
                        typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 7 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 8 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 9 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 10 >::type
                      , typename boost::is_same<void, action_result>::type
                    >
                    broadcast_invoker_type;
            typedef
                typename hpx::actions::make_action<decltype(&broadcast_invoker_type::call), &broadcast_invoker_type::call>::type
                type;
        };
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10)
    {
        hpx::id_type dest = hpx::get_colocation_id(ids[0]);
        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , ids
                , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10
              , typename boost::is_same<void, action_result>::type()
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast(
            hpx::actions::action<
                Component, Result, Arguments, Derived
            > 
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10)
    {
        return broadcast<Derived>(
                ids
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10
            );
    }
}}
namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
        >
        hpx::future<void>
        broadcast_impl12(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return hpx::lcos::make_ready_future();
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11
                      , boost::integral_constant<bool, true>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).then(&return_void);
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
        >
        hpx::future<typename broadcast_result<Action>::type>
        broadcast_impl12(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            if(ids.empty()) return hpx::lcos::make_ready_future(result_type());
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11
                )
                .then(
                    &wrap_into_vector<action_result>
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11
                      , boost::integral_constant<bool, false>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>);
        }
        
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
          , typename IsVoid
        >
        struct broadcast_invoker12
        {
            static hpx::future<typename broadcast_result<Action>::type>
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11
              , IsVoid
            )
            {
                return
                    broadcast_impl12(
                        act
                      , ids
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11
                      , IsVoid()
                    );
            }
        };
        template <
            typename Action
        >
        struct make_broadcast_action_impl<Action, 12>
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef broadcast_invoker12<
                        Action
                      ,
                        typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 7 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 8 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 9 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 10 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 11 >::type
                      , typename boost::is_same<void, action_result>::type
                    >
                    broadcast_invoker_type;
            typedef
                typename hpx::actions::make_action<decltype(&broadcast_invoker_type::call), &broadcast_invoker_type::call>::type
                type;
        };
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11)
    {
        hpx::id_type dest = hpx::get_colocation_id(ids[0]);
        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , ids
                , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11
              , typename boost::is_same<void, action_result>::type()
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast(
            hpx::actions::action<
                Component, Result, Arguments, Derived
            > 
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11)
    {
        return broadcast<Derived>(
                ids
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11
            );
    }
}}
namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
        >
        hpx::future<void>
        broadcast_impl13(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return hpx::lcos::make_ready_future();
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12
                      , boost::integral_constant<bool, true>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).then(&return_void);
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
        >
        hpx::future<typename broadcast_result<Action>::type>
        broadcast_impl13(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            if(ids.empty()) return hpx::lcos::make_ready_future(result_type());
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12
                )
                .then(
                    &wrap_into_vector<action_result>
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12
                      , boost::integral_constant<bool, false>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>);
        }
        
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
          , typename IsVoid
        >
        struct broadcast_invoker13
        {
            static hpx::future<typename broadcast_result<Action>::type>
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12
              , IsVoid
            )
            {
                return
                    broadcast_impl13(
                        act
                      , ids
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12
                      , IsVoid()
                    );
            }
        };
        template <
            typename Action
        >
        struct make_broadcast_action_impl<Action, 13>
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef broadcast_invoker13<
                        Action
                      ,
                        typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 7 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 8 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 9 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 10 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 11 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 12 >::type
                      , typename boost::is_same<void, action_result>::type
                    >
                    broadcast_invoker_type;
            typedef
                typename hpx::actions::make_action<decltype(&broadcast_invoker_type::call), &broadcast_invoker_type::call>::type
                type;
        };
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12)
    {
        hpx::id_type dest = hpx::get_colocation_id(ids[0]);
        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , ids
                , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12
              , typename boost::is_same<void, action_result>::type()
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast(
            hpx::actions::action<
                Component, Result, Arguments, Derived
            > 
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12)
    {
        return broadcast<Derived>(
                ids
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12
            );
    }
}}
namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
        >
        hpx::future<void>
        broadcast_impl14(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return hpx::lcos::make_ready_future();
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13
                      , boost::integral_constant<bool, true>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).then(&return_void);
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
        >
        hpx::future<typename broadcast_result<Action>::type>
        broadcast_impl14(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            if(ids.empty()) return hpx::lcos::make_ready_future(result_type());
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13
                )
                .then(
                    &wrap_into_vector<action_result>
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13
                      , boost::integral_constant<bool, false>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>);
        }
        
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
          , typename IsVoid
        >
        struct broadcast_invoker14
        {
            static hpx::future<typename broadcast_result<Action>::type>
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13
              , IsVoid
            )
            {
                return
                    broadcast_impl14(
                        act
                      , ids
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13
                      , IsVoid()
                    );
            }
        };
        template <
            typename Action
        >
        struct make_broadcast_action_impl<Action, 14>
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef broadcast_invoker14<
                        Action
                      ,
                        typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 7 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 8 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 9 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 10 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 11 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 12 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 13 >::type
                      , typename boost::is_same<void, action_result>::type
                    >
                    broadcast_invoker_type;
            typedef
                typename hpx::actions::make_action<decltype(&broadcast_invoker_type::call), &broadcast_invoker_type::call>::type
                type;
        };
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13)
    {
        hpx::id_type dest = hpx::get_colocation_id(ids[0]);
        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , ids
                , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13
              , typename boost::is_same<void, action_result>::type()
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast(
            hpx::actions::action<
                Component, Result, Arguments, Derived
            > 
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13)
    {
        return broadcast<Derived>(
                ids
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13
            );
    }
}}
namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
        >
        hpx::future<void>
        broadcast_impl15(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return hpx::lcos::make_ready_future();
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14
                      , boost::integral_constant<bool, true>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).then(&return_void);
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
        >
        hpx::future<typename broadcast_result<Action>::type>
        broadcast_impl15(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            if(ids.empty()) return hpx::lcos::make_ready_future(result_type());
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14
                )
                .then(
                    &wrap_into_vector<action_result>
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14
                      , boost::integral_constant<bool, false>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>);
        }
        
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
          , typename IsVoid
        >
        struct broadcast_invoker15
        {
            static hpx::future<typename broadcast_result<Action>::type>
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14
              , IsVoid
            )
            {
                return
                    broadcast_impl15(
                        act
                      , ids
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14
                      , IsVoid()
                    );
            }
        };
        template <
            typename Action
        >
        struct make_broadcast_action_impl<Action, 15>
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef broadcast_invoker15<
                        Action
                      ,
                        typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 7 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 8 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 9 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 10 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 11 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 12 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 13 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 14 >::type
                      , typename boost::is_same<void, action_result>::type
                    >
                    broadcast_invoker_type;
            typedef
                typename hpx::actions::make_action<decltype(&broadcast_invoker_type::call), &broadcast_invoker_type::call>::type
                type;
        };
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14)
    {
        hpx::id_type dest = hpx::get_colocation_id(ids[0]);
        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , ids
                , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14
              , typename boost::is_same<void, action_result>::type()
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast(
            hpx::actions::action<
                Component, Result, Arguments, Derived
            > 
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14)
    {
        return broadcast<Derived>(
                ids
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14
            );
    }
}}
namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
        >
        hpx::future<void>
        broadcast_impl16(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return hpx::lcos::make_ready_future();
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15
                      , boost::integral_constant<bool, true>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).then(&return_void);
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
        >
        hpx::future<typename broadcast_result<Action>::type>
        broadcast_impl16(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            if(ids.empty()) return hpx::lcos::make_ready_future(result_type());
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15
                )
                .then(
                    &wrap_into_vector<action_result>
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15
                      , boost::integral_constant<bool, false>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>);
        }
        
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
          , typename IsVoid
        >
        struct broadcast_invoker16
        {
            static hpx::future<typename broadcast_result<Action>::type>
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15
              , IsVoid
            )
            {
                return
                    broadcast_impl16(
                        act
                      , ids
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15
                      , IsVoid()
                    );
            }
        };
        template <
            typename Action
        >
        struct make_broadcast_action_impl<Action, 16>
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef broadcast_invoker16<
                        Action
                      ,
                        typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 7 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 8 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 9 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 10 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 11 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 12 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 13 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 14 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 15 >::type
                      , typename boost::is_same<void, action_result>::type
                    >
                    broadcast_invoker_type;
            typedef
                typename hpx::actions::make_action<decltype(&broadcast_invoker_type::call), &broadcast_invoker_type::call>::type
                type;
        };
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15)
    {
        hpx::id_type dest = hpx::get_colocation_id(ids[0]);
        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , ids
                , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15
              , typename boost::is_same<void, action_result>::type()
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast(
            hpx::actions::action<
                Component, Result, Arguments, Derived
            > 
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15)
    {
        return broadcast<Derived>(
                ids
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15
            );
    }
}}
namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
        >
        hpx::future<void>
        broadcast_impl17(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return hpx::lcos::make_ready_future();
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16
                      , boost::integral_constant<bool, true>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).then(&return_void);
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
        >
        hpx::future<typename broadcast_result<Action>::type>
        broadcast_impl17(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            if(ids.empty()) return hpx::lcos::make_ready_future(result_type());
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16
                )
                .then(
                    &wrap_into_vector<action_result>
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16
                      , boost::integral_constant<bool, false>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>);
        }
        
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
          , typename IsVoid
        >
        struct broadcast_invoker17
        {
            static hpx::future<typename broadcast_result<Action>::type>
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16
              , IsVoid
            )
            {
                return
                    broadcast_impl17(
                        act
                      , ids
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16
                      , IsVoid()
                    );
            }
        };
        template <
            typename Action
        >
        struct make_broadcast_action_impl<Action, 17>
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef broadcast_invoker17<
                        Action
                      ,
                        typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 7 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 8 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 9 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 10 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 11 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 12 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 13 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 14 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 15 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 16 >::type
                      , typename boost::is_same<void, action_result>::type
                    >
                    broadcast_invoker_type;
            typedef
                typename hpx::actions::make_action<decltype(&broadcast_invoker_type::call), &broadcast_invoker_type::call>::type
                type;
        };
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16)
    {
        hpx::id_type dest = hpx::get_colocation_id(ids[0]);
        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , ids
                , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16
              , typename boost::is_same<void, action_result>::type()
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast(
            hpx::actions::action<
                Component, Result, Arguments, Derived
            > 
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16)
    {
        return broadcast<Derived>(
                ids
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16
            );
    }
}}
namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17
        >
        hpx::future<void>
        broadcast_impl18(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16 , A17 const & a17
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return hpx::lcos::make_ready_future();
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17
                      , boost::integral_constant<bool, true>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).then(&return_void);
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17
        >
        hpx::future<typename broadcast_result<Action>::type>
        broadcast_impl18(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16 , A17 const & a17
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            if(ids.empty()) return hpx::lcos::make_ready_future(result_type());
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17
                )
                .then(
                    &wrap_into_vector<action_result>
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17
                      , boost::integral_constant<bool, false>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>);
        }
        
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17
          , typename IsVoid
        >
        struct broadcast_invoker18
        {
            static hpx::future<typename broadcast_result<Action>::type>
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16 , A17 const & a17
              , IsVoid
            )
            {
                return
                    broadcast_impl18(
                        act
                      , ids
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17
                      , IsVoid()
                    );
            }
        };
        template <
            typename Action
        >
        struct make_broadcast_action_impl<Action, 18>
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef broadcast_invoker18<
                        Action
                      ,
                        typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 7 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 8 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 9 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 10 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 11 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 12 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 13 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 14 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 15 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 16 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 17 >::type
                      , typename boost::is_same<void, action_result>::type
                    >
                    broadcast_invoker_type;
            typedef
                typename hpx::actions::make_action<decltype(&broadcast_invoker_type::call), &broadcast_invoker_type::call>::type
                type;
        };
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16 , A17 const & a17)
    {
        hpx::id_type dest = hpx::get_colocation_id(ids[0]);
        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , ids
                , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17
              , typename boost::is_same<void, action_result>::type()
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast(
            hpx::actions::action<
                Component, Result, Arguments, Derived
            > 
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16 , A17 const & a17)
    {
        return broadcast<Derived>(
                ids
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17
            );
    }
}}
namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18
        >
        hpx::future<void>
        broadcast_impl19(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16 , A17 const & a17 , A18 const & a18
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return hpx::lcos::make_ready_future();
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18
                      , boost::integral_constant<bool, true>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).then(&return_void);
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18
        >
        hpx::future<typename broadcast_result<Action>::type>
        broadcast_impl19(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16 , A17 const & a17 , A18 const & a18
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            if(ids.empty()) return hpx::lcos::make_ready_future(result_type());
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18
                )
                .then(
                    &wrap_into_vector<action_result>
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18
                      , boost::integral_constant<bool, false>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>);
        }
        
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18
          , typename IsVoid
        >
        struct broadcast_invoker19
        {
            static hpx::future<typename broadcast_result<Action>::type>
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16 , A17 const & a17 , A18 const & a18
              , IsVoid
            )
            {
                return
                    broadcast_impl19(
                        act
                      , ids
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18
                      , IsVoid()
                    );
            }
        };
        template <
            typename Action
        >
        struct make_broadcast_action_impl<Action, 19>
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef broadcast_invoker19<
                        Action
                      ,
                        typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 7 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 8 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 9 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 10 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 11 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 12 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 13 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 14 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 15 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 16 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 17 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 18 >::type
                      , typename boost::is_same<void, action_result>::type
                    >
                    broadcast_invoker_type;
            typedef
                typename hpx::actions::make_action<decltype(&broadcast_invoker_type::call), &broadcast_invoker_type::call>::type
                type;
        };
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16 , A17 const & a17 , A18 const & a18)
    {
        hpx::id_type dest = hpx::get_colocation_id(ids[0]);
        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , ids
                , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18
              , typename boost::is_same<void, action_result>::type()
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast(
            hpx::actions::action<
                Component, Result, Arguments, Derived
            > 
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16 , A17 const & a17 , A18 const & a18)
    {
        return broadcast<Derived>(
                ids
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18
            );
    }
}}
namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19
        >
        hpx::future<void>
        broadcast_impl20(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16 , A17 const & a17 , A18 const & a18 , A19 const & a19
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return hpx::lcos::make_ready_future();
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18 , a19
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18 , a19
                      , boost::integral_constant<bool, true>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18 , a19
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).then(&return_void);
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19
        >
        hpx::future<typename broadcast_result<Action>::type>
        broadcast_impl20(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16 , A17 const & a17 , A18 const & a18 , A19 const & a19
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            if(ids.empty()) return hpx::lcos::make_ready_future(result_type());
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_futures.push_back(
                hpx::async(
                    act
                  , ids[0]
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18 , a19
                )
                .then(
                    &wrap_into_vector<action_result>
                )
            );
            if(ids.size() > 1)
            {
                std::size_t half = (ids.size() / 2) + 1;
                std::vector<hpx::id_type>
                    ids_first(ids.begin() + 1, ids.begin() + half);
                std::vector<hpx::id_type>
                    ids_second(ids.begin() + half, ids.end());
                typedef
                    typename detail::make_broadcast_action<
                        Action
                    >::type
                    broadcast_impl_action;
                hpx::id_type id = hpx::get_colocation_id(ids_first[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id
                      , act
                      , boost::move(ids_first)
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18 , a19
                      , boost::integral_constant<bool, false>::type()
                    )
                );
                if(!ids_second.empty())
                {
                    id = hpx::get_colocation_id(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18 , a19
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>);
        }
        
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19
          , typename IsVoid
        >
        struct broadcast_invoker20
        {
            static hpx::future<typename broadcast_result<Action>::type>
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16 , A17 const & a17 , A18 const & a18 , A19 const & a19
              , IsVoid
            )
            {
                return
                    broadcast_impl20(
                        act
                      , ids
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18 , a19
                      , IsVoid()
                    );
            }
        };
        template <
            typename Action
        >
        struct make_broadcast_action_impl<Action, 20>
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef broadcast_invoker20<
                        Action
                      ,
                        typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 7 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 8 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 9 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 10 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 11 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 12 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 13 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 14 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 15 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 16 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 17 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 18 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 19 >::type
                      , typename boost::is_same<void, action_result>::type
                    >
                    broadcast_invoker_type;
            typedef
                typename hpx::actions::make_action<decltype(&broadcast_invoker_type::call), &broadcast_invoker_type::call>::type
                type;
        };
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16 , A17 const & a17 , A18 const & a18 , A19 const & a19)
    {
        hpx::id_type dest = hpx::get_colocation_id(ids[0]);
        typedef
            typename detail::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename detail::broadcast_result<Action>::action_result
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest
              , Action()
              , ids
                , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18 , a19
              , typename boost::is_same<void, action_result>::type()
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19
    >
    hpx::future<
        typename detail::broadcast_result<Derived>::type
    >
    broadcast(
            hpx::actions::action<
                Component, Result, Arguments, Derived
            > 
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16 , A17 const & a17 , A18 const & a18 , A19 const & a19)
    {
        return broadcast<Derived>(
                ids
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18 , a19
            );
    }
}}
