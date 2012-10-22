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
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6
          , boost::mpl::false_
        )
        {
            typedef components::server::create_component_action9<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6>::type>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6
          , boost::mpl::true_
        )
        {
            typedef
                components::server::create_component_direct_action9<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6>::type>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
        dataflow(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6
        )
            : base_type(
                create_component(target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )
                  , typename Action::direct_execution()
                )
            )
        {
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7
          , boost::mpl::false_
        )
        {
            typedef components::server::create_component_action10<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A7>::type>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7
          , boost::mpl::true_
        )
        {
            typedef
                components::server::create_component_direct_action10<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A7>::type>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
        dataflow(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7
        )
            : base_type(
                create_component(target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 )
                  , typename Action::direct_execution()
                )
            )
        {
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8
          , boost::mpl::false_
        )
        {
            typedef components::server::create_component_action11<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A7>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A8>::type>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8
          , boost::mpl::true_
        )
        {
            typedef
                components::server::create_component_direct_action11<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A7>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A8>::type>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
        dataflow(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8
        )
            : base_type(
                create_component(target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 )
                  , typename Action::direct_execution()
                )
            )
        {
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9
          , boost::mpl::false_
        )
        {
            typedef components::server::create_component_action12<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A7>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A8>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A9>::type>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9
          , boost::mpl::true_
        )
        {
            typedef
                components::server::create_component_direct_action12<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A7>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A8>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A9>::type>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
        dataflow(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9
        )
            : base_type(
                create_component(target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 )
                  , typename Action::direct_execution()
                )
            )
        {
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10
          , boost::mpl::false_
        )
        {
            typedef components::server::create_component_action13<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A7>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A8>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A9>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A10>::type>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10
          , boost::mpl::true_
        )
        {
            typedef
                components::server::create_component_direct_action13<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A7>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A8>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A9>::type>::type const & , typename boost::remove_const< typename hpx::util::detail::remove_reference< A10>::type>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
        dataflow(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10
        )
            : base_type(
                create_component(target
                  , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 )
                  , typename Action::direct_execution()
                )
            )
        {
        }
