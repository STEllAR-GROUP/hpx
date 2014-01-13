// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


        template <typename A0>
        static inline lcos::unique_future<naming::id_type>
        create_component(naming::id_type const & target
          , A0 && a0
          , boost::mpl::false_
        )
        {
            typedef components::server::create_component_action3<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<A0>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<A0>( a0 )
                );
        }
        template <typename A0>
        static inline lcos::unique_future<naming::id_type>
        create_component(naming::id_type const & target
          , A0 && a0
          , boost::mpl::true_
        )
        {
            typedef
                components::server::create_component_direct_action3<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<A0>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<A0>( a0 )
                );
        }
        template <typename A0>
        dataflow(
            naming::id_type const & target
          , A0 && a0
        )
            : base_type(
                create_component(target
                  , std::forward<A0>( a0 )
                  , typename Action::direct_execution()
                )
            )
        {
        }
        template <typename A0 , typename A1>
        static inline lcos::unique_future<naming::id_type>
        create_component(naming::id_type const & target
          , A0 && a0 , A1 && a1
          , boost::mpl::false_
        )
        {
            typedef components::server::create_component_action4<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<A0>::type const & , typename util::decay<A1>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 )
                );
        }
        template <typename A0 , typename A1>
        static inline lcos::unique_future<naming::id_type>
        create_component(naming::id_type const & target
          , A0 && a0 , A1 && a1
          , boost::mpl::true_
        )
        {
            typedef
                components::server::create_component_direct_action4<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<A0>::type const & , typename util::decay<A1>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 )
                );
        }
        template <typename A0 , typename A1>
        dataflow(
            naming::id_type const & target
          , A0 && a0 , A1 && a1
        )
            : base_type(
                create_component(target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 )
                  , typename Action::direct_execution()
                )
            )
        {
        }
        template <typename A0 , typename A1 , typename A2>
        static inline lcos::unique_future<naming::id_type>
        create_component(naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2
          , boost::mpl::false_
        )
        {
            typedef components::server::create_component_action5<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<A0>::type const & , typename util::decay<A1>::type const & , typename util::decay<A2>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 )
                );
        }
        template <typename A0 , typename A1 , typename A2>
        static inline lcos::unique_future<naming::id_type>
        create_component(naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2
          , boost::mpl::true_
        )
        {
            typedef
                components::server::create_component_direct_action5<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<A0>::type const & , typename util::decay<A1>::type const & , typename util::decay<A2>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 )
                );
        }
        template <typename A0 , typename A1 , typename A2>
        dataflow(
            naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2
        )
            : base_type(
                create_component(target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 )
                  , typename Action::direct_execution()
                )
            )
        {
        }
        template <typename A0 , typename A1 , typename A2 , typename A3>
        static inline lcos::unique_future<naming::id_type>
        create_component(naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3
          , boost::mpl::false_
        )
        {
            typedef components::server::create_component_action6<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<A0>::type const & , typename util::decay<A1>::type const & , typename util::decay<A2>::type const & , typename util::decay<A3>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3>
        static inline lcos::unique_future<naming::id_type>
        create_component(naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3
          , boost::mpl::true_
        )
        {
            typedef
                components::server::create_component_direct_action6<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<A0>::type const & , typename util::decay<A1>::type const & , typename util::decay<A2>::type const & , typename util::decay<A3>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3>
        dataflow(
            naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3
        )
            : base_type(
                create_component(target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 )
                  , typename Action::direct_execution()
                )
            )
        {
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
        static inline lcos::unique_future<naming::id_type>
        create_component(naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4
          , boost::mpl::false_
        )
        {
            typedef components::server::create_component_action7<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<A0>::type const & , typename util::decay<A1>::type const & , typename util::decay<A2>::type const & , typename util::decay<A3>::type const & , typename util::decay<A4>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
        static inline lcos::unique_future<naming::id_type>
        create_component(naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4
          , boost::mpl::true_
        )
        {
            typedef
                components::server::create_component_direct_action7<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<A0>::type const & , typename util::decay<A1>::type const & , typename util::decay<A2>::type const & , typename util::decay<A3>::type const & , typename util::decay<A4>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
        dataflow(
            naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4
        )
            : base_type(
                create_component(target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 )
                  , typename Action::direct_execution()
                )
            )
        {
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
        static inline lcos::unique_future<naming::id_type>
        create_component(naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5
          , boost::mpl::false_
        )
        {
            typedef components::server::create_component_action8<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<A0>::type const & , typename util::decay<A1>::type const & , typename util::decay<A2>::type const & , typename util::decay<A3>::type const & , typename util::decay<A4>::type const & , typename util::decay<A5>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
        static inline lcos::unique_future<naming::id_type>
        create_component(naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5
          , boost::mpl::true_
        )
        {
            typedef
                components::server::create_component_direct_action8<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<A0>::type const & , typename util::decay<A1>::type const & , typename util::decay<A2>::type const & , typename util::decay<A3>::type const & , typename util::decay<A4>::type const & , typename util::decay<A5>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
        dataflow(
            naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5
        )
            : base_type(
                create_component(target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 )
                  , typename Action::direct_execution()
                )
            )
        {
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
        static inline lcos::unique_future<naming::id_type>
        create_component(naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6
          , boost::mpl::false_
        )
        {
            typedef components::server::create_component_action9<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<A0>::type const & , typename util::decay<A1>::type const & , typename util::decay<A2>::type const & , typename util::decay<A3>::type const & , typename util::decay<A4>::type const & , typename util::decay<A5>::type const & , typename util::decay<A6>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
        static inline lcos::unique_future<naming::id_type>
        create_component(naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6
          , boost::mpl::true_
        )
        {
            typedef
                components::server::create_component_direct_action9<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<A0>::type const & , typename util::decay<A1>::type const & , typename util::decay<A2>::type const & , typename util::decay<A3>::type const & , typename util::decay<A4>::type const & , typename util::decay<A5>::type const & , typename util::decay<A6>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
        dataflow(
            naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6
        )
            : base_type(
                create_component(target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 )
                  , typename Action::direct_execution()
                )
            )
        {
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
        static inline lcos::unique_future<naming::id_type>
        create_component(naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7
          , boost::mpl::false_
        )
        {
            typedef components::server::create_component_action10<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<A0>::type const & , typename util::decay<A1>::type const & , typename util::decay<A2>::type const & , typename util::decay<A3>::type const & , typename util::decay<A4>::type const & , typename util::decay<A5>::type const & , typename util::decay<A6>::type const & , typename util::decay<A7>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
        static inline lcos::unique_future<naming::id_type>
        create_component(naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7
          , boost::mpl::true_
        )
        {
            typedef
                components::server::create_component_direct_action10<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<A0>::type const & , typename util::decay<A1>::type const & , typename util::decay<A2>::type const & , typename util::decay<A3>::type const & , typename util::decay<A4>::type const & , typename util::decay<A5>::type const & , typename util::decay<A6>::type const & , typename util::decay<A7>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
        dataflow(
            naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7
        )
            : base_type(
                create_component(target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 )
                  , typename Action::direct_execution()
                )
            )
        {
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
        static inline lcos::unique_future<naming::id_type>
        create_component(naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8
          , boost::mpl::false_
        )
        {
            typedef components::server::create_component_action11<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<A0>::type const & , typename util::decay<A1>::type const & , typename util::decay<A2>::type const & , typename util::decay<A3>::type const & , typename util::decay<A4>::type const & , typename util::decay<A5>::type const & , typename util::decay<A6>::type const & , typename util::decay<A7>::type const & , typename util::decay<A8>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
        static inline lcos::unique_future<naming::id_type>
        create_component(naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8
          , boost::mpl::true_
        )
        {
            typedef
                components::server::create_component_direct_action11<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<A0>::type const & , typename util::decay<A1>::type const & , typename util::decay<A2>::type const & , typename util::decay<A3>::type const & , typename util::decay<A4>::type const & , typename util::decay<A5>::type const & , typename util::decay<A6>::type const & , typename util::decay<A7>::type const & , typename util::decay<A8>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
        dataflow(
            naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8
        )
            : base_type(
                create_component(target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 )
                  , typename Action::direct_execution()
                )
            )
        {
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
        static inline lcos::unique_future<naming::id_type>
        create_component(naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9
          , boost::mpl::false_
        )
        {
            typedef components::server::create_component_action12<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<A0>::type const & , typename util::decay<A1>::type const & , typename util::decay<A2>::type const & , typename util::decay<A3>::type const & , typename util::decay<A4>::type const & , typename util::decay<A5>::type const & , typename util::decay<A6>::type const & , typename util::decay<A7>::type const & , typename util::decay<A8>::type const & , typename util::decay<A9>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
        static inline lcos::unique_future<naming::id_type>
        create_component(naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9
          , boost::mpl::true_
        )
        {
            typedef
                components::server::create_component_direct_action12<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<A0>::type const & , typename util::decay<A1>::type const & , typename util::decay<A2>::type const & , typename util::decay<A3>::type const & , typename util::decay<A4>::type const & , typename util::decay<A5>::type const & , typename util::decay<A6>::type const & , typename util::decay<A7>::type const & , typename util::decay<A8>::type const & , typename util::decay<A9>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
        dataflow(
            naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9
        )
            : base_type(
                create_component(target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 )
                  , typename Action::direct_execution()
                )
            )
        {
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
        static inline lcos::unique_future<naming::id_type>
        create_component(naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10
          , boost::mpl::false_
        )
        {
            typedef components::server::create_component_action13<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<A0>::type const & , typename util::decay<A1>::type const & , typename util::decay<A2>::type const & , typename util::decay<A3>::type const & , typename util::decay<A4>::type const & , typename util::decay<A5>::type const & , typename util::decay<A6>::type const & , typename util::decay<A7>::type const & , typename util::decay<A8>::type const & , typename util::decay<A9>::type const & , typename util::decay<A10>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
        static inline lcos::unique_future<naming::id_type>
        create_component(naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10
          , boost::mpl::true_
        )
        {
            typedef
                components::server::create_component_direct_action13<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<A0>::type const & , typename util::decay<A1>::type const & , typename util::decay<A2>::type const & , typename util::decay<A3>::type const & , typename util::decay<A4>::type const & , typename util::decay<A5>::type const & , typename util::decay<A6>::type const & , typename util::decay<A7>::type const & , typename util::decay<A8>::type const & , typename util::decay<A9>::type const & , typename util::decay<A10>::type const &
                > create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 )
                );
        }
        template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
        dataflow(
            naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10
        )
            : base_type(
                create_component(target
                  , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 )
                  , typename Action::direct_execution()
                )
            )
        {
        }
