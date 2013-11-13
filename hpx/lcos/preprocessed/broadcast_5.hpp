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
          , typename Futures
           
        >
        void
        broadcast_invoke(Action act, Futures& futures, hpx::id_type const& id
           
          , std::size_t)
        {
            futures.push_back(
                hpx::async(
                    act
                  , id
                   
                )
            );
        }
        template <
            typename Action
          , typename Futures
           
        >
        void
        broadcast_invoke(broadcast_with_index<Action>, Futures& futures, hpx::id_type const& id
           
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async(
                    Action()
                  , id
                   
                  , global_idx
                )
            );
        }
        template <
            typename Action
          , typename Futures
          , typename Cont
           
        >
        void
        broadcast_invoke(Action act, Futures& futures, BOOST_FWD_REF(Cont) cont
          , hpx::id_type const& id
           
          , std::size_t)
        {
            futures.push_back(
                hpx::async(
                    act
                  , id
                   
                ).then(boost::forward<Cont>(cont))
            );
        }
        template <
            typename Action
          , typename Futures
          , typename Cont
           
        >
        void
        broadcast_invoke(broadcast_with_index<Action>, Futures& futures
          , BOOST_FWD_REF(Cont) cont
          , hpx::id_type const& id
           
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async(
                    Action()
                  , id
                   
                  , global_idx
                ).then(boost::forward<Cont>(cont))
            );
        }
        template <
            typename Action
           
        >
        
        void
        broadcast_impl0(
            Action const & act
          , std::vector<hpx::id_type> const & ids
           
          , std::size_t global_idx
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return;
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_invoke(
                act
              , broadcast_futures
              , ids[0]
               
              , global_idx
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
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                           
                          , global_idx + 1
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                           
                          , global_idx + half
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            
            hpx::when_all(broadcast_futures).then(&return_void).get();
        }
        template <
            typename Action
           
        >
        
        typename broadcast_result<Action>::type
        broadcast_impl0(
            Action const & act
          , std::vector<hpx::id_type> const & ids
           
          , std::size_t global_idx
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            
            if(ids.empty()) return result_type();
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_invoke(
                act
              , broadcast_futures
              , &wrap_into_vector<action_result>
              , ids[0]
               
              , global_idx
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
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                           
                          , global_idx + 1
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                           
                          , global_idx + half
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>).move();
        }
        
        template <
            typename Action
           
          , typename IsVoid
        >
        struct broadcast_invoker0
        {
            
            static typename broadcast_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
               
              , std::size_t global_idx
              , IsVoid
            )
            {
                return
                    broadcast_impl0(
                        act
                      , ids
                       
                      , global_idx
                      , IsVoid()
                    );
            }
        };
        template <typename Action>
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
                typename HPX_MAKE_ACTION_TPL(broadcast_invoker_type::call)::type
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
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);
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
               
              , 0
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
    template <
        typename Action
       
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast_with_index(
        std::vector<hpx::id_type> const & ids
       )
    {
        return broadcast<detail::broadcast_with_index<Action> >(
                ids
               
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
    broadcast_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
       )
    {
        return broadcast<detail::broadcast_with_index<Derived> >(
                ids
               
            );
    }
}}
namespace hpx { namespace lcos {
    namespace detail
    {
        template <
            typename Action
          , typename Futures
          , typename A0
        >
        void
        broadcast_invoke(Action act, Futures& futures, hpx::id_type const& id
          , A0 const & a0
          , std::size_t)
        {
            futures.push_back(
                hpx::async(
                    act
                  , id
                  , a0
                )
            );
        }
        template <
            typename Action
          , typename Futures
          , typename A0
        >
        void
        broadcast_invoke(broadcast_with_index<Action>, Futures& futures, hpx::id_type const& id
          , A0 const & a0
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async(
                    Action()
                  , id
                  , a0
                  , global_idx
                )
            );
        }
        template <
            typename Action
          , typename Futures
          , typename Cont
          , typename A0
        >
        void
        broadcast_invoke(Action act, Futures& futures, BOOST_FWD_REF(Cont) cont
          , hpx::id_type const& id
          , A0 const & a0
          , std::size_t)
        {
            futures.push_back(
                hpx::async(
                    act
                  , id
                  , a0
                ).then(boost::forward<Cont>(cont))
            );
        }
        template <
            typename Action
          , typename Futures
          , typename Cont
          , typename A0
        >
        void
        broadcast_invoke(broadcast_with_index<Action>, Futures& futures
          , BOOST_FWD_REF(Cont) cont
          , hpx::id_type const& id
          , A0 const & a0
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async(
                    Action()
                  , id
                  , a0
                  , global_idx
                ).then(boost::forward<Cont>(cont))
            );
        }
        template <
            typename Action
          , typename A0
        >
        
        void
        broadcast_impl1(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0
          , std::size_t global_idx
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return;
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_invoke(
                act
              , broadcast_futures
              , ids[0]
              , a0
              , global_idx
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
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , a0
                          , global_idx + 1
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0
                          , global_idx + half
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            
            hpx::when_all(broadcast_futures).then(&return_void).get();
        }
        template <
            typename Action
          , typename A0
        >
        
        typename broadcast_result<Action>::type
        broadcast_impl1(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0
          , std::size_t global_idx
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            
            if(ids.empty()) return result_type();
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_invoke(
                act
              , broadcast_futures
              , &wrap_into_vector<action_result>
              , ids[0]
              , a0
              , global_idx
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
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , a0
                          , global_idx + 1
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0
                          , global_idx + half
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>).move();
        }
        
        template <
            typename Action
          , typename A0
          , typename IsVoid
        >
        struct broadcast_invoker1
        {
            
            static typename broadcast_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0
              , std::size_t global_idx
              , IsVoid
            )
            {
                return
                    broadcast_impl1(
                        act
                      , ids
                      , a0
                      , global_idx
                      , IsVoid()
                    );
            }
        };
        template <typename Action>
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
                typename HPX_MAKE_ACTION_TPL(broadcast_invoker_type::call)::type
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
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);
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
              , 0
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
    template <
        typename Action
      , typename A0
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast_with_index(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0)
    {
        return broadcast<detail::broadcast_with_index<Action> >(
                ids
              , a0
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
    broadcast_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , A0 const & a0)
    {
        return broadcast<detail::broadcast_with_index<Derived> >(
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
          , typename Futures
          , typename A0 , typename A1
        >
        void
        broadcast_invoke(Action act, Futures& futures, hpx::id_type const& id
          , A0 const & a0 , A1 const & a1
          , std::size_t)
        {
            futures.push_back(
                hpx::async(
                    act
                  , id
                  , a0 , a1
                )
            );
        }
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1
        >
        void
        broadcast_invoke(broadcast_with_index<Action>, Futures& futures, hpx::id_type const& id
          , A0 const & a0 , A1 const & a1
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async(
                    Action()
                  , id
                  , a0 , a1
                  , global_idx
                )
            );
        }
        template <
            typename Action
          , typename Futures
          , typename Cont
          , typename A0 , typename A1
        >
        void
        broadcast_invoke(Action act, Futures& futures, BOOST_FWD_REF(Cont) cont
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1
          , std::size_t)
        {
            futures.push_back(
                hpx::async(
                    act
                  , id
                  , a0 , a1
                ).then(boost::forward<Cont>(cont))
            );
        }
        template <
            typename Action
          , typename Futures
          , typename Cont
          , typename A0 , typename A1
        >
        void
        broadcast_invoke(broadcast_with_index<Action>, Futures& futures
          , BOOST_FWD_REF(Cont) cont
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async(
                    Action()
                  , id
                  , a0 , a1
                  , global_idx
                ).then(boost::forward<Cont>(cont))
            );
        }
        template <
            typename Action
          , typename A0 , typename A1
        >
        
        void
        broadcast_impl2(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1
          , std::size_t global_idx
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return;
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_invoke(
                act
              , broadcast_futures
              , ids[0]
              , a0 , a1
              , global_idx
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
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , a0 , a1
                          , global_idx + 1
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1
                          , global_idx + half
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            
            hpx::when_all(broadcast_futures).then(&return_void).get();
        }
        template <
            typename Action
          , typename A0 , typename A1
        >
        
        typename broadcast_result<Action>::type
        broadcast_impl2(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1
          , std::size_t global_idx
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            
            if(ids.empty()) return result_type();
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_invoke(
                act
              , broadcast_futures
              , &wrap_into_vector<action_result>
              , ids[0]
              , a0 , a1
              , global_idx
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
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , a0 , a1
                          , global_idx + 1
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1
                          , global_idx + half
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>).move();
        }
        
        template <
            typename Action
          , typename A0 , typename A1
          , typename IsVoid
        >
        struct broadcast_invoker2
        {
            
            static typename broadcast_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1
              , std::size_t global_idx
              , IsVoid
            )
            {
                return
                    broadcast_impl2(
                        act
                      , ids
                      , a0 , a1
                      , global_idx
                      , IsVoid()
                    );
            }
        };
        template <typename Action>
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
                typename HPX_MAKE_ACTION_TPL(broadcast_invoker_type::call)::type
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
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);
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
              , 0
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
    template <
        typename Action
      , typename A0 , typename A1
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast_with_index(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1)
    {
        return broadcast<detail::broadcast_with_index<Action> >(
                ids
              , a0 , a1
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
    broadcast_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1)
    {
        return broadcast<detail::broadcast_with_index<Derived> >(
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
          , typename Futures
          , typename A0 , typename A1 , typename A2
        >
        void
        broadcast_invoke(Action act, Futures& futures, hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2
          , std::size_t)
        {
            futures.push_back(
                hpx::async(
                    act
                  , id
                  , a0 , a1 , a2
                )
            );
        }
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2
        >
        void
        broadcast_invoke(broadcast_with_index<Action>, Futures& futures, hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async(
                    Action()
                  , id
                  , a0 , a1 , a2
                  , global_idx
                )
            );
        }
        template <
            typename Action
          , typename Futures
          , typename Cont
          , typename A0 , typename A1 , typename A2
        >
        void
        broadcast_invoke(Action act, Futures& futures, BOOST_FWD_REF(Cont) cont
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2
          , std::size_t)
        {
            futures.push_back(
                hpx::async(
                    act
                  , id
                  , a0 , a1 , a2
                ).then(boost::forward<Cont>(cont))
            );
        }
        template <
            typename Action
          , typename Futures
          , typename Cont
          , typename A0 , typename A1 , typename A2
        >
        void
        broadcast_invoke(broadcast_with_index<Action>, Futures& futures
          , BOOST_FWD_REF(Cont) cont
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async(
                    Action()
                  , id
                  , a0 , a1 , a2
                  , global_idx
                ).then(boost::forward<Cont>(cont))
            );
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2
        >
        
        void
        broadcast_impl3(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2
          , std::size_t global_idx
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return;
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_invoke(
                act
              , broadcast_futures
              , ids[0]
              , a0 , a1 , a2
              , global_idx
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
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , a0 , a1 , a2
                          , global_idx + 1
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2
                          , global_idx + half
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            
            hpx::when_all(broadcast_futures).then(&return_void).get();
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2
        >
        
        typename broadcast_result<Action>::type
        broadcast_impl3(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2
          , std::size_t global_idx
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            
            if(ids.empty()) return result_type();
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_invoke(
                act
              , broadcast_futures
              , &wrap_into_vector<action_result>
              , ids[0]
              , a0 , a1 , a2
              , global_idx
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
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , a0 , a1 , a2
                          , global_idx + 1
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2
                          , global_idx + half
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>).move();
        }
        
        template <
            typename Action
          , typename A0 , typename A1 , typename A2
          , typename IsVoid
        >
        struct broadcast_invoker3
        {
            
            static typename broadcast_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1 , A2 const & a2
              , std::size_t global_idx
              , IsVoid
            )
            {
                return
                    broadcast_impl3(
                        act
                      , ids
                      , a0 , a1 , a2
                      , global_idx
                      , IsVoid()
                    );
            }
        };
        template <typename Action>
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
                typename HPX_MAKE_ACTION_TPL(broadcast_invoker_type::call)::type
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
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);
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
              , 0
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
    template <
        typename Action
      , typename A0 , typename A1 , typename A2
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast_with_index(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2)
    {
        return broadcast<detail::broadcast_with_index<Action> >(
                ids
              , a0 , a1 , a2
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
    broadcast_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2)
    {
        return broadcast<detail::broadcast_with_index<Derived> >(
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
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3
        >
        void
        broadcast_invoke(Action act, Futures& futures, hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3
          , std::size_t)
        {
            futures.push_back(
                hpx::async(
                    act
                  , id
                  , a0 , a1 , a2 , a3
                )
            );
        }
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3
        >
        void
        broadcast_invoke(broadcast_with_index<Action>, Futures& futures, hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async(
                    Action()
                  , id
                  , a0 , a1 , a2 , a3
                  , global_idx
                )
            );
        }
        template <
            typename Action
          , typename Futures
          , typename Cont
          , typename A0 , typename A1 , typename A2 , typename A3
        >
        void
        broadcast_invoke(Action act, Futures& futures, BOOST_FWD_REF(Cont) cont
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3
          , std::size_t)
        {
            futures.push_back(
                hpx::async(
                    act
                  , id
                  , a0 , a1 , a2 , a3
                ).then(boost::forward<Cont>(cont))
            );
        }
        template <
            typename Action
          , typename Futures
          , typename Cont
          , typename A0 , typename A1 , typename A2 , typename A3
        >
        void
        broadcast_invoke(broadcast_with_index<Action>, Futures& futures
          , BOOST_FWD_REF(Cont) cont
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async(
                    Action()
                  , id
                  , a0 , a1 , a2 , a3
                  , global_idx
                ).then(boost::forward<Cont>(cont))
            );
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3
        >
        
        void
        broadcast_impl4(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3
          , std::size_t global_idx
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return;
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_invoke(
                act
              , broadcast_futures
              , ids[0]
              , a0 , a1 , a2 , a3
              , global_idx
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
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , a0 , a1 , a2 , a3
                          , global_idx + 1
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3
                          , global_idx + half
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            
            hpx::when_all(broadcast_futures).then(&return_void).get();
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3
        >
        
        typename broadcast_result<Action>::type
        broadcast_impl4(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3
          , std::size_t global_idx
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            
            if(ids.empty()) return result_type();
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_invoke(
                act
              , broadcast_futures
              , &wrap_into_vector<action_result>
              , ids[0]
              , a0 , a1 , a2 , a3
              , global_idx
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
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , a0 , a1 , a2 , a3
                          , global_idx + 1
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3
                          , global_idx + half
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>).move();
        }
        
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3
          , typename IsVoid
        >
        struct broadcast_invoker4
        {
            
            static typename broadcast_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3
              , std::size_t global_idx
              , IsVoid
            )
            {
                return
                    broadcast_impl4(
                        act
                      , ids
                      , a0 , a1 , a2 , a3
                      , global_idx
                      , IsVoid()
                    );
            }
        };
        template <typename Action>
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
                typename HPX_MAKE_ACTION_TPL(broadcast_invoker_type::call)::type
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
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);
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
              , 0
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
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast_with_index(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3)
    {
        return broadcast<detail::broadcast_with_index<Action> >(
                ids
              , a0 , a1 , a2 , a3
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
    broadcast_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3)
    {
        return broadcast<detail::broadcast_with_index<Derived> >(
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
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
        >
        void
        broadcast_invoke(Action act, Futures& futures, hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4
          , std::size_t)
        {
            futures.push_back(
                hpx::async(
                    act
                  , id
                  , a0 , a1 , a2 , a3 , a4
                )
            );
        }
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
        >
        void
        broadcast_invoke(broadcast_with_index<Action>, Futures& futures, hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async(
                    Action()
                  , id
                  , a0 , a1 , a2 , a3 , a4
                  , global_idx
                )
            );
        }
        template <
            typename Action
          , typename Futures
          , typename Cont
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
        >
        void
        broadcast_invoke(Action act, Futures& futures, BOOST_FWD_REF(Cont) cont
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4
          , std::size_t)
        {
            futures.push_back(
                hpx::async(
                    act
                  , id
                  , a0 , a1 , a2 , a3 , a4
                ).then(boost::forward<Cont>(cont))
            );
        }
        template <
            typename Action
          , typename Futures
          , typename Cont
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
        >
        void
        broadcast_invoke(broadcast_with_index<Action>, Futures& futures
          , BOOST_FWD_REF(Cont) cont
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async(
                    Action()
                  , id
                  , a0 , a1 , a2 , a3 , a4
                  , global_idx
                ).then(boost::forward<Cont>(cont))
            );
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
        >
        
        void
        broadcast_impl5(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4
          , std::size_t global_idx
          , boost::mpl::true_
        )
        {
            if(ids.empty()) return;
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_invoke(
                act
              , broadcast_futures
              , ids[0]
              , a0 , a1 , a2 , a3 , a4
              , global_idx
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
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , a0 , a1 , a2 , a3 , a4
                          , global_idx + 1
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4
                          , global_idx + half
                          , boost::integral_constant<bool, true>::type()
                        )
                    );
                }
            }
            
            hpx::when_all(broadcast_futures).then(&return_void).get();
        }
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
        >
        
        typename broadcast_result<Action>::type
        broadcast_impl5(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4
          , std::size_t global_idx
          , boost::mpl::false_
        )
        {
            typedef
                typename broadcast_result<Action>::action_result
                action_result;
            typedef
                typename broadcast_result<Action>::type
                result_type;
            
            if(ids.empty()) return result_type();
            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(3);
            broadcast_invoke(
                act
              , broadcast_futures
              , &wrap_into_vector<action_result>
              , ids[0]
              , a0 , a1 , a2 , a3 , a4
              , global_idx
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
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , a0 , a1 , a2 , a3 , a4
                          , global_idx + 1
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    broadcast_futures.push_back(
                        hpx::async<broadcast_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , a0 , a1 , a2 , a3 , a4
                          , global_idx + half
                          , boost::integral_constant<bool, false>::type()
                        )
                    );
                }
            }
            return hpx::when_all(broadcast_futures).
                then(&return_result_type<action_result>).move();
        }
        
        template <
            typename Action
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
          , typename IsVoid
        >
        struct broadcast_invoker5
        {
            
            static typename broadcast_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4
              , std::size_t global_idx
              , IsVoid
            )
            {
                return
                    broadcast_impl5(
                        act
                      , ids
                      , a0 , a1 , a2 , a3 , a4
                      , global_idx
                      , IsVoid()
                    );
            }
        };
        template <typename Action>
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
                typename HPX_MAKE_ACTION_TPL(broadcast_invoker_type::call)::type
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
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);
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
              , 0
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
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
    >
    hpx::future<
        typename detail::broadcast_result<Action>::type
    >
    broadcast_with_index(
        std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4)
    {
        return broadcast<detail::broadcast_with_index<Action> >(
                ids
              , a0 , a1 , a2 , a3 , a4
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
    broadcast_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4)
    {
        return broadcast<detail::broadcast_with_index<Derived> >(
                ids
              , a0 , a1 , a2 , a3 , a4
            );
    }
}}
