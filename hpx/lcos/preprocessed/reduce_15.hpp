// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace lcos
{
    namespace detail
    {
        template <
            typename Action
          , typename Futures
          
        >
        void
        reduce_invoke(Action 
          , Futures& futures
          , hpx::id_type const& id
          
          , std::size_t)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                   
                )
            );
        }
        template <
            typename Action
          , typename Futures
          
        >
        void
        reduce_invoke(reduce_with_index<Action>
          , Futures& futures
          , hpx::id_type const& id
          
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  
                  , global_idx
                )
            );
        }
        
        template <
            typename Action
          , typename ReduceOp
          
        >
        typename reduce_result<Action>::type
        reduce_impl0(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , BOOST_FWD_REF(ReduceOp) reduce_op
          
          , std::size_t global_idx
        )
        {
            typedef
                typename reduce_result<Action>::type
                result_type;
            if(ids.empty()) return result_type();
            std::vector<hpx::future<result_type> > reduce_futures;
            reduce_futures.reserve(3);
            reduce_invoke(
                act
              , reduce_futures
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
                    typename detail::make_reduce_action<Action>::
                        template reduce_invoker<ReduceOp>::type
                    reduce_impl_action;
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , reduce_op
                          
                          , global_idx + 1
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , reduce_op
                          
                          , global_idx + half
                        )
                    );
                }
            }
            return hpx::when_all(reduce_futures).
                then(perform_reduction<result_type, ReduceOp>(reduce_op)).
                move();
        }
        
        template <
            typename Action
          , typename ReduceOp
          
        >
        struct reduce_invoker0
        {
            
            static typename reduce_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , ReduceOp const& reduce_op
              
              , std::size_t global_idx
            )
            {
                return
                    reduce_impl0(
                        act
                      , ids
                      , reduce_op
                      
                      , global_idx
                    );
            }
        };
        template <typename Action>
        struct make_reduce_action_impl<Action, 0>
        {
            typedef
                typename reduce_result<Action>::type
                action_result;
            template <typename ReduceOp>
            struct reduce_invoker
            {
                typedef
                    typename util::decay<ReduceOp>::type
                    reduce_op_type;
                typedef reduce_invoker0<
                        Action
                      , reduce_op_type
                      
                    >
                    reduce_invoker_type;
                typedef
                    typename HPX_MAKE_ACTION_TPL(reduce_invoker_type::call)::type
                    type;
            };
        };
    }
    
    template <
        typename Action
      , typename ReduceOp
      
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      )
    {
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);
        typedef
            typename detail::make_reduce_action<Action>::
                template reduce_invoker<ReduceOp>::type
            reduce_impl_action;
        typedef
            typename detail::reduce_result<Action>::type
            action_result;
        return
            hpx::async<reduce_impl_action>(
                dest
              , Action()
              , ids
              , boost::forward<ReduceOp>(reduce_op)
              
              , 0
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      )
    {
        return reduce<Derived>(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              
            );
    }
    template <
        typename Action
      , typename ReduceOp
      
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce_with_index(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      )
    {
        return reduce<detail::reduce_with_index<Action> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , BOOST_FWD_REF(ReduceOp) reduce_op
      
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      )
    {
        return reduce<detail::reduce_with_index<Derived> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              
            );
    }
}}
namespace hpx { namespace lcos
{
    namespace detail
    {
        template <
            typename Action
          , typename Futures
          , typename A0
        >
        void
        reduce_invoke(Action 
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0
          , std::size_t)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
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
        reduce_invoke(reduce_with_index<Action>
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0
                  , global_idx
                )
            );
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0
        >
        typename reduce_result<Action>::type
        reduce_impl1(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , BOOST_FWD_REF(ReduceOp) reduce_op
          , A0 const & a0
          , std::size_t global_idx
        )
        {
            typedef
                typename reduce_result<Action>::type
                result_type;
            if(ids.empty()) return result_type();
            std::vector<hpx::future<result_type> > reduce_futures;
            reduce_futures.reserve(3);
            reduce_invoke(
                act
              , reduce_futures
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
                    typename detail::make_reduce_action<Action>::
                        template reduce_invoker<ReduceOp>::type
                    reduce_impl_action;
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , reduce_op
                          , a0
                          , global_idx + 1
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , reduce_op
                          , a0
                          , global_idx + half
                        )
                    );
                }
            }
            return hpx::when_all(reduce_futures).
                then(perform_reduction<result_type, ReduceOp>(reduce_op)).
                move();
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0
        >
        struct reduce_invoker1
        {
            
            static typename reduce_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , ReduceOp const& reduce_op
              , A0 const & a0
              , std::size_t global_idx
            )
            {
                return
                    reduce_impl1(
                        act
                      , ids
                      , reduce_op
                      , a0
                      , global_idx
                    );
            }
        };
        template <typename Action>
        struct make_reduce_action_impl<Action, 1>
        {
            typedef
                typename reduce_result<Action>::type
                action_result;
            template <typename ReduceOp>
            struct reduce_invoker
            {
                typedef
                    typename util::decay<ReduceOp>::type
                    reduce_op_type;
                typedef reduce_invoker1<
                        Action
                      , reduce_op_type
                      , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type
                    >
                    reduce_invoker_type;
                typedef
                    typename HPX_MAKE_ACTION_TPL(reduce_invoker_type::call)::type
                    type;
            };
        };
    }
    
    template <
        typename Action
      , typename ReduceOp
      , typename A0
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0)
    {
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);
        typedef
            typename detail::make_reduce_action<Action>::
                template reduce_invoker<ReduceOp>::type
            reduce_impl_action;
        typedef
            typename detail::reduce_result<Action>::type
            action_result;
        return
            hpx::async<reduce_impl_action>(
                dest
              , Action()
              , ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0
              , 0
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , typename A0
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0)
    {
        return reduce<Derived>(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0
            );
    }
    template <
        typename Action
      , typename ReduceOp
      , typename A0
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce_with_index(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0)
    {
        return reduce<detail::reduce_with_index<Action> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , typename A0
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , A0 const & a0)
    {
        return reduce<detail::reduce_with_index<Derived> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0
            );
    }
}}
namespace hpx { namespace lcos
{
    namespace detail
    {
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1
        >
        void
        reduce_invoke(Action 
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1
          , std::size_t)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
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
        reduce_invoke(reduce_with_index<Action>
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1
                  , global_idx
                )
            );
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1
        >
        typename reduce_result<Action>::type
        reduce_impl2(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , BOOST_FWD_REF(ReduceOp) reduce_op
          , A0 const & a0 , A1 const & a1
          , std::size_t global_idx
        )
        {
            typedef
                typename reduce_result<Action>::type
                result_type;
            if(ids.empty()) return result_type();
            std::vector<hpx::future<result_type> > reduce_futures;
            reduce_futures.reserve(3);
            reduce_invoke(
                act
              , reduce_futures
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
                    typename detail::make_reduce_action<Action>::
                        template reduce_invoker<ReduceOp>::type
                    reduce_impl_action;
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , reduce_op
                          , a0 , a1
                          , global_idx + 1
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , reduce_op
                          , a0 , a1
                          , global_idx + half
                        )
                    );
                }
            }
            return hpx::when_all(reduce_futures).
                then(perform_reduction<result_type, ReduceOp>(reduce_op)).
                move();
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1
        >
        struct reduce_invoker2
        {
            
            static typename reduce_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , ReduceOp const& reduce_op
              , A0 const & a0 , A1 const & a1
              , std::size_t global_idx
            )
            {
                return
                    reduce_impl2(
                        act
                      , ids
                      , reduce_op
                      , a0 , a1
                      , global_idx
                    );
            }
        };
        template <typename Action>
        struct make_reduce_action_impl<Action, 2>
        {
            typedef
                typename reduce_result<Action>::type
                action_result;
            template <typename ReduceOp>
            struct reduce_invoker
            {
                typedef
                    typename util::decay<ReduceOp>::type
                    reduce_op_type;
                typedef reduce_invoker2<
                        Action
                      , reduce_op_type
                      , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type
                    >
                    reduce_invoker_type;
                typedef
                    typename HPX_MAKE_ACTION_TPL(reduce_invoker_type::call)::type
                    type;
            };
        };
    }
    
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1)
    {
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);
        typedef
            typename detail::make_reduce_action<Action>::
                template reduce_invoker<ReduceOp>::type
            reduce_impl_action;
        typedef
            typename detail::reduce_result<Action>::type
            action_result;
        return
            hpx::async<reduce_impl_action>(
                dest
              , Action()
              , ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1
              , 0
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , typename A0 , typename A1
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1)
    {
        return reduce<Derived>(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1
            );
    }
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce_with_index(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1)
    {
        return reduce<detail::reduce_with_index<Action> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , typename A0 , typename A1
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1)
    {
        return reduce<detail::reduce_with_index<Derived> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1
            );
    }
}}
namespace hpx { namespace lcos
{
    namespace detail
    {
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2
        >
        void
        reduce_invoke(Action 
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2
          , std::size_t)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
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
        reduce_invoke(reduce_with_index<Action>
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2
                  , global_idx
                )
            );
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2
        >
        typename reduce_result<Action>::type
        reduce_impl3(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , BOOST_FWD_REF(ReduceOp) reduce_op
          , A0 const & a0 , A1 const & a1 , A2 const & a2
          , std::size_t global_idx
        )
        {
            typedef
                typename reduce_result<Action>::type
                result_type;
            if(ids.empty()) return result_type();
            std::vector<hpx::future<result_type> > reduce_futures;
            reduce_futures.reserve(3);
            reduce_invoke(
                act
              , reduce_futures
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
                    typename detail::make_reduce_action<Action>::
                        template reduce_invoker<ReduceOp>::type
                    reduce_impl_action;
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , reduce_op
                          , a0 , a1 , a2
                          , global_idx + 1
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , reduce_op
                          , a0 , a1 , a2
                          , global_idx + half
                        )
                    );
                }
            }
            return hpx::when_all(reduce_futures).
                then(perform_reduction<result_type, ReduceOp>(reduce_op)).
                move();
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2
        >
        struct reduce_invoker3
        {
            
            static typename reduce_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , ReduceOp const& reduce_op
              , A0 const & a0 , A1 const & a1 , A2 const & a2
              , std::size_t global_idx
            )
            {
                return
                    reduce_impl3(
                        act
                      , ids
                      , reduce_op
                      , a0 , a1 , a2
                      , global_idx
                    );
            }
        };
        template <typename Action>
        struct make_reduce_action_impl<Action, 3>
        {
            typedef
                typename reduce_result<Action>::type
                action_result;
            template <typename ReduceOp>
            struct reduce_invoker
            {
                typedef
                    typename util::decay<ReduceOp>::type
                    reduce_op_type;
                typedef reduce_invoker3<
                        Action
                      , reduce_op_type
                      , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type
                    >
                    reduce_invoker_type;
                typedef
                    typename HPX_MAKE_ACTION_TPL(reduce_invoker_type::call)::type
                    type;
            };
        };
    }
    
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2)
    {
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);
        typedef
            typename detail::make_reduce_action<Action>::
                template reduce_invoker<ReduceOp>::type
            reduce_impl_action;
        typedef
            typename detail::reduce_result<Action>::type
            action_result;
        return
            hpx::async<reduce_impl_action>(
                dest
              , Action()
              , ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2
              , 0
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2)
    {
        return reduce<Derived>(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2
            );
    }
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce_with_index(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2)
    {
        return reduce<detail::reduce_with_index<Action> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , typename A0 , typename A1 , typename A2
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2)
    {
        return reduce<detail::reduce_with_index<Derived> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2
            );
    }
}}
namespace hpx { namespace lcos
{
    namespace detail
    {
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3
        >
        void
        reduce_invoke(Action 
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3
          , std::size_t)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
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
        reduce_invoke(reduce_with_index<Action>
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2 , a3
                  , global_idx
                )
            );
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3
        >
        typename reduce_result<Action>::type
        reduce_impl4(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , BOOST_FWD_REF(ReduceOp) reduce_op
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3
          , std::size_t global_idx
        )
        {
            typedef
                typename reduce_result<Action>::type
                result_type;
            if(ids.empty()) return result_type();
            std::vector<hpx::future<result_type> > reduce_futures;
            reduce_futures.reserve(3);
            reduce_invoke(
                act
              , reduce_futures
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
                    typename detail::make_reduce_action<Action>::
                        template reduce_invoker<ReduceOp>::type
                    reduce_impl_action;
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , reduce_op
                          , a0 , a1 , a2 , a3
                          , global_idx + 1
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , reduce_op
                          , a0 , a1 , a2 , a3
                          , global_idx + half
                        )
                    );
                }
            }
            return hpx::when_all(reduce_futures).
                then(perform_reduction<result_type, ReduceOp>(reduce_op)).
                move();
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3
        >
        struct reduce_invoker4
        {
            
            static typename reduce_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , ReduceOp const& reduce_op
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3
              , std::size_t global_idx
            )
            {
                return
                    reduce_impl4(
                        act
                      , ids
                      , reduce_op
                      , a0 , a1 , a2 , a3
                      , global_idx
                    );
            }
        };
        template <typename Action>
        struct make_reduce_action_impl<Action, 4>
        {
            typedef
                typename reduce_result<Action>::type
                action_result;
            template <typename ReduceOp>
            struct reduce_invoker
            {
                typedef
                    typename util::decay<ReduceOp>::type
                    reduce_op_type;
                typedef reduce_invoker4<
                        Action
                      , reduce_op_type
                      , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type
                    >
                    reduce_invoker_type;
                typedef
                    typename HPX_MAKE_ACTION_TPL(reduce_invoker_type::call)::type
                    type;
            };
        };
    }
    
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3)
    {
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);
        typedef
            typename detail::make_reduce_action<Action>::
                template reduce_invoker<ReduceOp>::type
            reduce_impl_action;
        typedef
            typename detail::reduce_result<Action>::type
            action_result;
        return
            hpx::async<reduce_impl_action>(
                dest
              , Action()
              , ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3
              , 0
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3)
    {
        return reduce<Derived>(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3
            );
    }
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce_with_index(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3)
    {
        return reduce<detail::reduce_with_index<Action> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , typename A0 , typename A1 , typename A2 , typename A3
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3)
    {
        return reduce<detail::reduce_with_index<Derived> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3
            );
    }
}}
namespace hpx { namespace lcos
{
    namespace detail
    {
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
        >
        void
        reduce_invoke(Action 
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4
          , std::size_t)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
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
        reduce_invoke(reduce_with_index<Action>
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2 , a3 , a4
                  , global_idx
                )
            );
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
        >
        typename reduce_result<Action>::type
        reduce_impl5(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , BOOST_FWD_REF(ReduceOp) reduce_op
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4
          , std::size_t global_idx
        )
        {
            typedef
                typename reduce_result<Action>::type
                result_type;
            if(ids.empty()) return result_type();
            std::vector<hpx::future<result_type> > reduce_futures;
            reduce_futures.reserve(3);
            reduce_invoke(
                act
              , reduce_futures
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
                    typename detail::make_reduce_action<Action>::
                        template reduce_invoker<ReduceOp>::type
                    reduce_impl_action;
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , reduce_op
                          , a0 , a1 , a2 , a3 , a4
                          , global_idx + 1
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , reduce_op
                          , a0 , a1 , a2 , a3 , a4
                          , global_idx + half
                        )
                    );
                }
            }
            return hpx::when_all(reduce_futures).
                then(perform_reduction<result_type, ReduceOp>(reduce_op)).
                move();
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
        >
        struct reduce_invoker5
        {
            
            static typename reduce_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , ReduceOp const& reduce_op
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4
              , std::size_t global_idx
            )
            {
                return
                    reduce_impl5(
                        act
                      , ids
                      , reduce_op
                      , a0 , a1 , a2 , a3 , a4
                      , global_idx
                    );
            }
        };
        template <typename Action>
        struct make_reduce_action_impl<Action, 5>
        {
            typedef
                typename reduce_result<Action>::type
                action_result;
            template <typename ReduceOp>
            struct reduce_invoker
            {
                typedef
                    typename util::decay<ReduceOp>::type
                    reduce_op_type;
                typedef reduce_invoker5<
                        Action
                      , reduce_op_type
                      , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type
                    >
                    reduce_invoker_type;
                typedef
                    typename HPX_MAKE_ACTION_TPL(reduce_invoker_type::call)::type
                    type;
            };
        };
    }
    
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4)
    {
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);
        typedef
            typename detail::make_reduce_action<Action>::
                template reduce_invoker<ReduceOp>::type
            reduce_impl_action;
        typedef
            typename detail::reduce_result<Action>::type
            action_result;
        return
            hpx::async<reduce_impl_action>(
                dest
              , Action()
              , ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4
              , 0
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4)
    {
        return reduce<Derived>(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4
            );
    }
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce_with_index(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4)
    {
        return reduce<detail::reduce_with_index<Action> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4)
    {
        return reduce<detail::reduce_with_index<Derived> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4
            );
    }
}}
namespace hpx { namespace lcos
{
    namespace detail
    {
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
        >
        void
        reduce_invoke(Action 
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5
          , std::size_t)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2 , a3 , a4 , a5
                )
            );
        }
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
        >
        void
        reduce_invoke(reduce_with_index<Action>
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2 , a3 , a4 , a5
                  , global_idx
                )
            );
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
        >
        typename reduce_result<Action>::type
        reduce_impl6(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , BOOST_FWD_REF(ReduceOp) reduce_op
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5
          , std::size_t global_idx
        )
        {
            typedef
                typename reduce_result<Action>::type
                result_type;
            if(ids.empty()) return result_type();
            std::vector<hpx::future<result_type> > reduce_futures;
            reduce_futures.reserve(3);
            reduce_invoke(
                act
              , reduce_futures
              , ids[0]
              , a0 , a1 , a2 , a3 , a4 , a5
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
                    typename detail::make_reduce_action<Action>::
                        template reduce_invoker<ReduceOp>::type
                    reduce_impl_action;
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , reduce_op
                          , a0 , a1 , a2 , a3 , a4 , a5
                          , global_idx + 1
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , reduce_op
                          , a0 , a1 , a2 , a3 , a4 , a5
                          , global_idx + half
                        )
                    );
                }
            }
            return hpx::when_all(reduce_futures).
                then(perform_reduction<result_type, ReduceOp>(reduce_op)).
                move();
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
        >
        struct reduce_invoker6
        {
            
            static typename reduce_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , ReduceOp const& reduce_op
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5
              , std::size_t global_idx
            )
            {
                return
                    reduce_impl6(
                        act
                      , ids
                      , reduce_op
                      , a0 , a1 , a2 , a3 , a4 , a5
                      , global_idx
                    );
            }
        };
        template <typename Action>
        struct make_reduce_action_impl<Action, 6>
        {
            typedef
                typename reduce_result<Action>::type
                action_result;
            template <typename ReduceOp>
            struct reduce_invoker
            {
                typedef
                    typename util::decay<ReduceOp>::type
                    reduce_op_type;
                typedef reduce_invoker6<
                        Action
                      , reduce_op_type
                      , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type
                    >
                    reduce_invoker_type;
                typedef
                    typename HPX_MAKE_ACTION_TPL(reduce_invoker_type::call)::type
                    type;
            };
        };
    }
    
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5)
    {
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);
        typedef
            typename detail::make_reduce_action<Action>::
                template reduce_invoker<ReduceOp>::type
            reduce_impl_action;
        typedef
            typename detail::reduce_result<Action>::type
            action_result;
        return
            hpx::async<reduce_impl_action>(
                dest
              , Action()
              , ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5
              , 0
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5)
    {
        return reduce<Derived>(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5
            );
    }
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce_with_index(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5)
    {
        return reduce<detail::reduce_with_index<Action> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5)
    {
        return reduce<detail::reduce_with_index<Derived> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5
            );
    }
}}
namespace hpx { namespace lcos
{
    namespace detail
    {
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
        >
        void
        reduce_invoke(Action 
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6
          , std::size_t)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6
                )
            );
        }
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
        >
        void
        reduce_invoke(reduce_with_index<Action>
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6
                  , global_idx
                )
            );
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
        >
        typename reduce_result<Action>::type
        reduce_impl7(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , BOOST_FWD_REF(ReduceOp) reduce_op
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6
          , std::size_t global_idx
        )
        {
            typedef
                typename reduce_result<Action>::type
                result_type;
            if(ids.empty()) return result_type();
            std::vector<hpx::future<result_type> > reduce_futures;
            reduce_futures.reserve(3);
            reduce_invoke(
                act
              , reduce_futures
              , ids[0]
              , a0 , a1 , a2 , a3 , a4 , a5 , a6
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
                    typename detail::make_reduce_action<Action>::
                        template reduce_invoker<ReduceOp>::type
                    reduce_impl_action;
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , reduce_op
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6
                          , global_idx + 1
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , reduce_op
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6
                          , global_idx + half
                        )
                    );
                }
            }
            return hpx::when_all(reduce_futures).
                then(perform_reduction<result_type, ReduceOp>(reduce_op)).
                move();
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
        >
        struct reduce_invoker7
        {
            
            static typename reduce_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , ReduceOp const& reduce_op
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6
              , std::size_t global_idx
            )
            {
                return
                    reduce_impl7(
                        act
                      , ids
                      , reduce_op
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6
                      , global_idx
                    );
            }
        };
        template <typename Action>
        struct make_reduce_action_impl<Action, 7>
        {
            typedef
                typename reduce_result<Action>::type
                action_result;
            template <typename ReduceOp>
            struct reduce_invoker
            {
                typedef
                    typename util::decay<ReduceOp>::type
                    reduce_op_type;
                typedef reduce_invoker7<
                        Action
                      , reduce_op_type
                      , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type
                    >
                    reduce_invoker_type;
                typedef
                    typename HPX_MAKE_ACTION_TPL(reduce_invoker_type::call)::type
                    type;
            };
        };
    }
    
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6)
    {
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);
        typedef
            typename detail::make_reduce_action<Action>::
                template reduce_invoker<ReduceOp>::type
            reduce_impl_action;
        typedef
            typename detail::reduce_result<Action>::type
            action_result;
        return
            hpx::async<reduce_impl_action>(
                dest
              , Action()
              , ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6
              , 0
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6)
    {
        return reduce<Derived>(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6
            );
    }
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce_with_index(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6)
    {
        return reduce<detail::reduce_with_index<Action> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6)
    {
        return reduce<detail::reduce_with_index<Derived> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6
            );
    }
}}
namespace hpx { namespace lcos
{
    namespace detail
    {
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
        >
        void
        reduce_invoke(Action 
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7
          , std::size_t)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7
                )
            );
        }
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
        >
        void
        reduce_invoke(reduce_with_index<Action>
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7
                  , global_idx
                )
            );
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
        >
        typename reduce_result<Action>::type
        reduce_impl8(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , BOOST_FWD_REF(ReduceOp) reduce_op
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7
          , std::size_t global_idx
        )
        {
            typedef
                typename reduce_result<Action>::type
                result_type;
            if(ids.empty()) return result_type();
            std::vector<hpx::future<result_type> > reduce_futures;
            reduce_futures.reserve(3);
            reduce_invoke(
                act
              , reduce_futures
              , ids[0]
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7
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
                    typename detail::make_reduce_action<Action>::
                        template reduce_invoker<ReduceOp>::type
                    reduce_impl_action;
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , reduce_op
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7
                          , global_idx + 1
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , reduce_op
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7
                          , global_idx + half
                        )
                    );
                }
            }
            return hpx::when_all(reduce_futures).
                then(perform_reduction<result_type, ReduceOp>(reduce_op)).
                move();
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
        >
        struct reduce_invoker8
        {
            
            static typename reduce_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , ReduceOp const& reduce_op
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7
              , std::size_t global_idx
            )
            {
                return
                    reduce_impl8(
                        act
                      , ids
                      , reduce_op
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7
                      , global_idx
                    );
            }
        };
        template <typename Action>
        struct make_reduce_action_impl<Action, 8>
        {
            typedef
                typename reduce_result<Action>::type
                action_result;
            template <typename ReduceOp>
            struct reduce_invoker
            {
                typedef
                    typename util::decay<ReduceOp>::type
                    reduce_op_type;
                typedef reduce_invoker8<
                        Action
                      , reduce_op_type
                      , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 7 >::type
                    >
                    reduce_invoker_type;
                typedef
                    typename HPX_MAKE_ACTION_TPL(reduce_invoker_type::call)::type
                    type;
            };
        };
    }
    
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7)
    {
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);
        typedef
            typename detail::make_reduce_action<Action>::
                template reduce_invoker<ReduceOp>::type
            reduce_impl_action;
        typedef
            typename detail::reduce_result<Action>::type
            action_result;
        return
            hpx::async<reduce_impl_action>(
                dest
              , Action()
              , ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7
              , 0
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7)
    {
        return reduce<Derived>(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7
            );
    }
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce_with_index(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7)
    {
        return reduce<detail::reduce_with_index<Action> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7)
    {
        return reduce<detail::reduce_with_index<Derived> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7
            );
    }
}}
namespace hpx { namespace lcos
{
    namespace detail
    {
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
        >
        void
        reduce_invoke(Action 
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8
          , std::size_t)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8
                )
            );
        }
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
        >
        void
        reduce_invoke(reduce_with_index<Action>
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8
                  , global_idx
                )
            );
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
        >
        typename reduce_result<Action>::type
        reduce_impl9(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , BOOST_FWD_REF(ReduceOp) reduce_op
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8
          , std::size_t global_idx
        )
        {
            typedef
                typename reduce_result<Action>::type
                result_type;
            if(ids.empty()) return result_type();
            std::vector<hpx::future<result_type> > reduce_futures;
            reduce_futures.reserve(3);
            reduce_invoke(
                act
              , reduce_futures
              , ids[0]
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8
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
                    typename detail::make_reduce_action<Action>::
                        template reduce_invoker<ReduceOp>::type
                    reduce_impl_action;
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , reduce_op
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8
                          , global_idx + 1
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , reduce_op
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8
                          , global_idx + half
                        )
                    );
                }
            }
            return hpx::when_all(reduce_futures).
                then(perform_reduction<result_type, ReduceOp>(reduce_op)).
                move();
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
        >
        struct reduce_invoker9
        {
            
            static typename reduce_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , ReduceOp const& reduce_op
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8
              , std::size_t global_idx
            )
            {
                return
                    reduce_impl9(
                        act
                      , ids
                      , reduce_op
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8
                      , global_idx
                    );
            }
        };
        template <typename Action>
        struct make_reduce_action_impl<Action, 9>
        {
            typedef
                typename reduce_result<Action>::type
                action_result;
            template <typename ReduceOp>
            struct reduce_invoker
            {
                typedef
                    typename util::decay<ReduceOp>::type
                    reduce_op_type;
                typedef reduce_invoker9<
                        Action
                      , reduce_op_type
                      , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 7 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 8 >::type
                    >
                    reduce_invoker_type;
                typedef
                    typename HPX_MAKE_ACTION_TPL(reduce_invoker_type::call)::type
                    type;
            };
        };
    }
    
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8)
    {
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);
        typedef
            typename detail::make_reduce_action<Action>::
                template reduce_invoker<ReduceOp>::type
            reduce_impl_action;
        typedef
            typename detail::reduce_result<Action>::type
            action_result;
        return
            hpx::async<reduce_impl_action>(
                dest
              , Action()
              , ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8
              , 0
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8)
    {
        return reduce<Derived>(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8
            );
    }
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce_with_index(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8)
    {
        return reduce<detail::reduce_with_index<Action> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8)
    {
        return reduce<detail::reduce_with_index<Derived> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8
            );
    }
}}
namespace hpx { namespace lcos
{
    namespace detail
    {
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
        >
        void
        reduce_invoke(Action 
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9
          , std::size_t)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9
                )
            );
        }
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
        >
        void
        reduce_invoke(reduce_with_index<Action>
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9
                  , global_idx
                )
            );
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
        >
        typename reduce_result<Action>::type
        reduce_impl10(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , BOOST_FWD_REF(ReduceOp) reduce_op
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9
          , std::size_t global_idx
        )
        {
            typedef
                typename reduce_result<Action>::type
                result_type;
            if(ids.empty()) return result_type();
            std::vector<hpx::future<result_type> > reduce_futures;
            reduce_futures.reserve(3);
            reduce_invoke(
                act
              , reduce_futures
              , ids[0]
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9
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
                    typename detail::make_reduce_action<Action>::
                        template reduce_invoker<ReduceOp>::type
                    reduce_impl_action;
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , reduce_op
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9
                          , global_idx + 1
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , reduce_op
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9
                          , global_idx + half
                        )
                    );
                }
            }
            return hpx::when_all(reduce_futures).
                then(perform_reduction<result_type, ReduceOp>(reduce_op)).
                move();
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
        >
        struct reduce_invoker10
        {
            
            static typename reduce_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , ReduceOp const& reduce_op
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9
              , std::size_t global_idx
            )
            {
                return
                    reduce_impl10(
                        act
                      , ids
                      , reduce_op
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9
                      , global_idx
                    );
            }
        };
        template <typename Action>
        struct make_reduce_action_impl<Action, 10>
        {
            typedef
                typename reduce_result<Action>::type
                action_result;
            template <typename ReduceOp>
            struct reduce_invoker
            {
                typedef
                    typename util::decay<ReduceOp>::type
                    reduce_op_type;
                typedef reduce_invoker10<
                        Action
                      , reduce_op_type
                      , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 7 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 8 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 9 >::type
                    >
                    reduce_invoker_type;
                typedef
                    typename HPX_MAKE_ACTION_TPL(reduce_invoker_type::call)::type
                    type;
            };
        };
    }
    
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9)
    {
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);
        typedef
            typename detail::make_reduce_action<Action>::
                template reduce_invoker<ReduceOp>::type
            reduce_impl_action;
        typedef
            typename detail::reduce_result<Action>::type
            action_result;
        return
            hpx::async<reduce_impl_action>(
                dest
              , Action()
              , ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9
              , 0
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9)
    {
        return reduce<Derived>(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9
            );
    }
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce_with_index(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9)
    {
        return reduce<detail::reduce_with_index<Action> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9)
    {
        return reduce<detail::reduce_with_index<Derived> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9
            );
    }
}}
namespace hpx { namespace lcos
{
    namespace detail
    {
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
        >
        void
        reduce_invoke(Action 
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10
          , std::size_t)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10
                )
            );
        }
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
        >
        void
        reduce_invoke(reduce_with_index<Action>
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10
                  , global_idx
                )
            );
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
        >
        typename reduce_result<Action>::type
        reduce_impl11(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , BOOST_FWD_REF(ReduceOp) reduce_op
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10
          , std::size_t global_idx
        )
        {
            typedef
                typename reduce_result<Action>::type
                result_type;
            if(ids.empty()) return result_type();
            std::vector<hpx::future<result_type> > reduce_futures;
            reduce_futures.reserve(3);
            reduce_invoke(
                act
              , reduce_futures
              , ids[0]
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10
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
                    typename detail::make_reduce_action<Action>::
                        template reduce_invoker<ReduceOp>::type
                    reduce_impl_action;
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , reduce_op
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10
                          , global_idx + 1
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , reduce_op
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10
                          , global_idx + half
                        )
                    );
                }
            }
            return hpx::when_all(reduce_futures).
                then(perform_reduction<result_type, ReduceOp>(reduce_op)).
                move();
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
        >
        struct reduce_invoker11
        {
            
            static typename reduce_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , ReduceOp const& reduce_op
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10
              , std::size_t global_idx
            )
            {
                return
                    reduce_impl11(
                        act
                      , ids
                      , reduce_op
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10
                      , global_idx
                    );
            }
        };
        template <typename Action>
        struct make_reduce_action_impl<Action, 11>
        {
            typedef
                typename reduce_result<Action>::type
                action_result;
            template <typename ReduceOp>
            struct reduce_invoker
            {
                typedef
                    typename util::decay<ReduceOp>::type
                    reduce_op_type;
                typedef reduce_invoker11<
                        Action
                      , reduce_op_type
                      , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 7 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 8 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 9 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 10 >::type
                    >
                    reduce_invoker_type;
                typedef
                    typename HPX_MAKE_ACTION_TPL(reduce_invoker_type::call)::type
                    type;
            };
        };
    }
    
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10)
    {
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);
        typedef
            typename detail::make_reduce_action<Action>::
                template reduce_invoker<ReduceOp>::type
            reduce_impl_action;
        typedef
            typename detail::reduce_result<Action>::type
            action_result;
        return
            hpx::async<reduce_impl_action>(
                dest
              , Action()
              , ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10
              , 0
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10)
    {
        return reduce<Derived>(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10
            );
    }
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce_with_index(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10)
    {
        return reduce<detail::reduce_with_index<Action> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10)
    {
        return reduce<detail::reduce_with_index<Derived> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10
            );
    }
}}
namespace hpx { namespace lcos
{
    namespace detail
    {
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
        >
        void
        reduce_invoke(Action 
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11
          , std::size_t)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11
                )
            );
        }
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
        >
        void
        reduce_invoke(reduce_with_index<Action>
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11
                  , global_idx
                )
            );
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
        >
        typename reduce_result<Action>::type
        reduce_impl12(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , BOOST_FWD_REF(ReduceOp) reduce_op
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11
          , std::size_t global_idx
        )
        {
            typedef
                typename reduce_result<Action>::type
                result_type;
            if(ids.empty()) return result_type();
            std::vector<hpx::future<result_type> > reduce_futures;
            reduce_futures.reserve(3);
            reduce_invoke(
                act
              , reduce_futures
              , ids[0]
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11
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
                    typename detail::make_reduce_action<Action>::
                        template reduce_invoker<ReduceOp>::type
                    reduce_impl_action;
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , reduce_op
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11
                          , global_idx + 1
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , reduce_op
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11
                          , global_idx + half
                        )
                    );
                }
            }
            return hpx::when_all(reduce_futures).
                then(perform_reduction<result_type, ReduceOp>(reduce_op)).
                move();
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
        >
        struct reduce_invoker12
        {
            
            static typename reduce_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , ReduceOp const& reduce_op
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11
              , std::size_t global_idx
            )
            {
                return
                    reduce_impl12(
                        act
                      , ids
                      , reduce_op
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11
                      , global_idx
                    );
            }
        };
        template <typename Action>
        struct make_reduce_action_impl<Action, 12>
        {
            typedef
                typename reduce_result<Action>::type
                action_result;
            template <typename ReduceOp>
            struct reduce_invoker
            {
                typedef
                    typename util::decay<ReduceOp>::type
                    reduce_op_type;
                typedef reduce_invoker12<
                        Action
                      , reduce_op_type
                      , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 7 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 8 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 9 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 10 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 11 >::type
                    >
                    reduce_invoker_type;
                typedef
                    typename HPX_MAKE_ACTION_TPL(reduce_invoker_type::call)::type
                    type;
            };
        };
    }
    
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11)
    {
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);
        typedef
            typename detail::make_reduce_action<Action>::
                template reduce_invoker<ReduceOp>::type
            reduce_impl_action;
        typedef
            typename detail::reduce_result<Action>::type
            action_result;
        return
            hpx::async<reduce_impl_action>(
                dest
              , Action()
              , ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11
              , 0
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11)
    {
        return reduce<Derived>(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11
            );
    }
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce_with_index(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11)
    {
        return reduce<detail::reduce_with_index<Action> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11)
    {
        return reduce<detail::reduce_with_index<Derived> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11
            );
    }
}}
namespace hpx { namespace lcos
{
    namespace detail
    {
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
        >
        void
        reduce_invoke(Action 
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12
          , std::size_t)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12
                )
            );
        }
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
        >
        void
        reduce_invoke(reduce_with_index<Action>
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12
                  , global_idx
                )
            );
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
        >
        typename reduce_result<Action>::type
        reduce_impl13(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , BOOST_FWD_REF(ReduceOp) reduce_op
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12
          , std::size_t global_idx
        )
        {
            typedef
                typename reduce_result<Action>::type
                result_type;
            if(ids.empty()) return result_type();
            std::vector<hpx::future<result_type> > reduce_futures;
            reduce_futures.reserve(3);
            reduce_invoke(
                act
              , reduce_futures
              , ids[0]
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12
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
                    typename detail::make_reduce_action<Action>::
                        template reduce_invoker<ReduceOp>::type
                    reduce_impl_action;
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , reduce_op
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12
                          , global_idx + 1
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , reduce_op
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12
                          , global_idx + half
                        )
                    );
                }
            }
            return hpx::when_all(reduce_futures).
                then(perform_reduction<result_type, ReduceOp>(reduce_op)).
                move();
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
        >
        struct reduce_invoker13
        {
            
            static typename reduce_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , ReduceOp const& reduce_op
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12
              , std::size_t global_idx
            )
            {
                return
                    reduce_impl13(
                        act
                      , ids
                      , reduce_op
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12
                      , global_idx
                    );
            }
        };
        template <typename Action>
        struct make_reduce_action_impl<Action, 13>
        {
            typedef
                typename reduce_result<Action>::type
                action_result;
            template <typename ReduceOp>
            struct reduce_invoker
            {
                typedef
                    typename util::decay<ReduceOp>::type
                    reduce_op_type;
                typedef reduce_invoker13<
                        Action
                      , reduce_op_type
                      , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 7 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 8 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 9 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 10 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 11 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 12 >::type
                    >
                    reduce_invoker_type;
                typedef
                    typename HPX_MAKE_ACTION_TPL(reduce_invoker_type::call)::type
                    type;
            };
        };
    }
    
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12)
    {
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);
        typedef
            typename detail::make_reduce_action<Action>::
                template reduce_invoker<ReduceOp>::type
            reduce_impl_action;
        typedef
            typename detail::reduce_result<Action>::type
            action_result;
        return
            hpx::async<reduce_impl_action>(
                dest
              , Action()
              , ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12
              , 0
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12)
    {
        return reduce<Derived>(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12
            );
    }
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce_with_index(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12)
    {
        return reduce<detail::reduce_with_index<Action> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12)
    {
        return reduce<detail::reduce_with_index<Derived> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12
            );
    }
}}
namespace hpx { namespace lcos
{
    namespace detail
    {
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
        >
        void
        reduce_invoke(Action 
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13
          , std::size_t)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13
                )
            );
        }
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
        >
        void
        reduce_invoke(reduce_with_index<Action>
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13
                  , global_idx
                )
            );
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
        >
        typename reduce_result<Action>::type
        reduce_impl14(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , BOOST_FWD_REF(ReduceOp) reduce_op
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13
          , std::size_t global_idx
        )
        {
            typedef
                typename reduce_result<Action>::type
                result_type;
            if(ids.empty()) return result_type();
            std::vector<hpx::future<result_type> > reduce_futures;
            reduce_futures.reserve(3);
            reduce_invoke(
                act
              , reduce_futures
              , ids[0]
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13
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
                    typename detail::make_reduce_action<Action>::
                        template reduce_invoker<ReduceOp>::type
                    reduce_impl_action;
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , reduce_op
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13
                          , global_idx + 1
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , reduce_op
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13
                          , global_idx + half
                        )
                    );
                }
            }
            return hpx::when_all(reduce_futures).
                then(perform_reduction<result_type, ReduceOp>(reduce_op)).
                move();
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
        >
        struct reduce_invoker14
        {
            
            static typename reduce_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , ReduceOp const& reduce_op
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13
              , std::size_t global_idx
            )
            {
                return
                    reduce_impl14(
                        act
                      , ids
                      , reduce_op
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13
                      , global_idx
                    );
            }
        };
        template <typename Action>
        struct make_reduce_action_impl<Action, 14>
        {
            typedef
                typename reduce_result<Action>::type
                action_result;
            template <typename ReduceOp>
            struct reduce_invoker
            {
                typedef
                    typename util::decay<ReduceOp>::type
                    reduce_op_type;
                typedef reduce_invoker14<
                        Action
                      , reduce_op_type
                      , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 7 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 8 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 9 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 10 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 11 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 12 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 13 >::type
                    >
                    reduce_invoker_type;
                typedef
                    typename HPX_MAKE_ACTION_TPL(reduce_invoker_type::call)::type
                    type;
            };
        };
    }
    
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13)
    {
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);
        typedef
            typename detail::make_reduce_action<Action>::
                template reduce_invoker<ReduceOp>::type
            reduce_impl_action;
        typedef
            typename detail::reduce_result<Action>::type
            action_result;
        return
            hpx::async<reduce_impl_action>(
                dest
              , Action()
              , ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13
              , 0
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13)
    {
        return reduce<Derived>(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13
            );
    }
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce_with_index(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13)
    {
        return reduce<detail::reduce_with_index<Action> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13)
    {
        return reduce<detail::reduce_with_index<Derived> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13
            );
    }
}}
namespace hpx { namespace lcos
{
    namespace detail
    {
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
        >
        void
        reduce_invoke(Action 
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14
          , std::size_t)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14
                )
            );
        }
        template <
            typename Action
          , typename Futures
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
        >
        void
        reduce_invoke(reduce_with_index<Action>
          , Futures& futures
          , hpx::id_type const& id
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14
          , std::size_t global_idx)
        {
            futures.push_back(
                hpx::async<Action>(
                    id
                  , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14
                  , global_idx
                )
            );
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
        >
        typename reduce_result<Action>::type
        reduce_impl15(
            Action const & act
          , std::vector<hpx::id_type> const & ids
          , BOOST_FWD_REF(ReduceOp) reduce_op
          , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14
          , std::size_t global_idx
        )
        {
            typedef
                typename reduce_result<Action>::type
                result_type;
            if(ids.empty()) return result_type();
            std::vector<hpx::future<result_type> > reduce_futures;
            reduce_futures.reserve(3);
            reduce_invoke(
                act
              , reduce_futures
              , ids[0]
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14
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
                    typename detail::make_reduce_action<Action>::
                        template reduce_invoker<ReduceOp>::type
                    reduce_impl_action;
                if(!ids_first.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_first[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_first)
                          , reduce_op
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14
                          , global_idx + 1
                        )
                    );
                }
                if(!ids_second.empty())
                {
                    hpx::id_type id = hpx::get_colocation_id_sync(ids_second[0]);
                    reduce_futures.push_back(
                        hpx::async<reduce_impl_action>(
                            id
                          , act
                          , boost::move(ids_second)
                          , reduce_op
                          , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14
                          , global_idx + half
                        )
                    );
                }
            }
            return hpx::when_all(reduce_futures).
                then(perform_reduction<result_type, ReduceOp>(reduce_op)).
                move();
        }
        
        template <
            typename Action
          , typename ReduceOp
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
        >
        struct reduce_invoker15
        {
            
            static typename reduce_result<Action>::type
            call(
                Action const & act
              , std::vector<hpx::id_type> const & ids
              , ReduceOp const& reduce_op
              , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14
              , std::size_t global_idx
            )
            {
                return
                    reduce_impl15(
                        act
                      , ids
                      , reduce_op
                      , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14
                      , global_idx
                    );
            }
        };
        template <typename Action>
        struct make_reduce_action_impl<Action, 15>
        {
            typedef
                typename reduce_result<Action>::type
                action_result;
            template <typename ReduceOp>
            struct reduce_invoker
            {
                typedef
                    typename util::decay<ReduceOp>::type
                    reduce_op_type;
                typedef reduce_invoker15<
                        Action
                      , reduce_op_type
                      , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 0 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 1 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 2 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 3 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 4 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 5 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 6 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 7 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 8 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 9 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 10 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 11 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 12 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 13 >::type , typename boost::fusion::result_of::value_at_c< typename Action::arguments_type, 14 >::type
                    >
                    reduce_invoker_type;
                typedef
                    typename HPX_MAKE_ACTION_TPL(reduce_invoker_type::call)::type
                    type;
            };
        };
    }
    
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14)
    {
        hpx::id_type dest = hpx::get_colocation_id_sync(ids[0]);
        typedef
            typename detail::make_reduce_action<Action>::
                template reduce_invoker<ReduceOp>::type
            reduce_impl_action;
        typedef
            typename detail::reduce_result<Action>::type
            action_result;
        return
            hpx::async<reduce_impl_action>(
                dest
              , Action()
              , ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14
              , 0
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14)
    {
        return reduce<Derived>(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14
            );
    }
    template <
        typename Action
      , typename ReduceOp
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
    >
    hpx::future<
        typename detail::reduce_result<Action>::type
    >
    reduce_with_index(
        std::vector<hpx::id_type> const & ids
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14)
    {
        return reduce<detail::reduce_with_index<Action> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14
            );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename ReduceOp
      , BOOST_FWD_REF(ReduceOp) reduce_op
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
    >
    hpx::future<
        typename detail::reduce_result<Derived>::type
    >
    reduce_with_index(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , std::vector<hpx::id_type> const & ids
      , A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14)
    {
        return reduce<detail::reduce_with_index<Derived> >(
                ids
              , boost::forward<ReduceOp>(reduce_op)
              , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14
            );
    }
}}
