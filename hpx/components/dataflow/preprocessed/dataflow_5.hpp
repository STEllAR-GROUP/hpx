// Copyright (c) 2007-2012 Hartmut Kaiser
// Copyright (c)      2012 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


        template <typename A0>
        static inline lcos::future<naming::id_type, naming::gid_type>
        create_component(naming::id_type const & target
          , BOOST_FWD_REF(A0) a0
          , boost::mpl::false_
        )
        {
            typedef
                typename hpx::components::server::create_one_component_action3<
                    components::managed_component<server::dataflow>
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const &
                >::type
                create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , stub_type::get_component_type()
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
                typename hpx::components::server::create_one_component_direct_action3<
                    components::managed_component<server::dataflow>
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0>::type>::type const &
                >::type
                create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , stub_type::get_component_type()
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
