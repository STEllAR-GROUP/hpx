// Copyright (c) 2007-2012 Hartmut Kaiser
// Copyright (c)      2012 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


        template <typename A0>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0
          , boost::mpl::false_
        )
        {
            typedef components::server::create_component_action3<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , boost::forward<A0>( a0 )
                );
        }
        template <typename A0>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0
          , boost::mpl::true_
        )
        {
            typedef
                components::server::create_component_direct_action3<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , boost::forward<A0>( a0 )
                );
        }
        template <typename A0>
        dataflow(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0
        )
            : base_type(
                create_component(target
                  , boost::forward<A0>( a0 )
                  , typename Action::direct_execution()
                )
            )
        {
        }
        template <typename A0 , typename A1>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1
          , boost::mpl::false_
        )
        {
            typedef components::server::create_component_action4<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1>::type>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )
                );
        }
        template <typename A0 , typename A1>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1
          , boost::mpl::true_
        )
        {
            typedef
                components::server::create_component_direct_action4<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1>::type>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )
                );
        }
        template <typename A0 , typename A1>
        dataflow(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1
        )
            : base_type(
                create_component(target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )
                  , typename Action::direct_execution()
                )
            )
        {
        }
        template <typename A0 , typename A1 , typename A2>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2
          , boost::mpl::false_
        )
        {
            typedef components::server::create_component_action5<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2>::type>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )
                );
        }
        template <typename A0 , typename A1 , typename A2>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2
          , boost::mpl::true_
        )
        {
            typedef
                components::server::create_component_direct_action5<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2>::type>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )
                );
        }
        template <typename A0 , typename A1 , typename A2>
        dataflow(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2
        )
            : base_type(
                create_component(target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )
                  , typename Action::direct_execution()
                )
            )
        {
        }
        template <typename A0 , typename A1 , typename A2 , typename A3>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3
          , boost::mpl::false_
        )
        {
            typedef components::server::create_component_action6<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3>::type>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3
          , boost::mpl::true_
        )
        {
            typedef
                components::server::create_component_direct_action6<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3>::type>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3>
        dataflow(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3
        )
            : base_type(
                create_component(target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )
                  , typename Action::direct_execution()
                )
            )
        {
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4
          , boost::mpl::false_
        )
        {
            typedef components::server::create_component_action7<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4>::type>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4
          , boost::mpl::true_
        )
        {
            typedef
                components::server::create_component_direct_action7<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4>::type>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
        dataflow(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4
        )
            : base_type(
                create_component(target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )
                  , typename Action::direct_execution()
                )
            )
        {
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5
          , boost::mpl::false_
        )
        {
            typedef components::server::create_component_action8<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5>::type>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5
          , boost::mpl::true_
        )
        {
            typedef
                components::server::create_component_direct_action8<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5>::type>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
        dataflow(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5
        )
            : base_type(
                create_component(target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )
                  , typename Action::direct_execution()
                )
            )
        {
        }
