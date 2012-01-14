//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DATAFLOW_STUBS_DATAFLOW_HPP
#define HPX_LCOS_DATAFLOW_STUBS_DATAFLOW_HPP

#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/components/dataflow/server/dataflow.hpp>

namespace hpx { namespace lcos {
    namespace stubs
    {
        struct dataflow
            : components::stubs::stub_base<
                server::dataflow
            >
        {
            typedef server::dataflow server_type;

#if 0
            template <typename Action>
            static promise<void>
            init_async(
                naming::id_type const & gid
              , naming::id_type const & target
            )
            {
                typedef
                    typename server::init_action<Action>
                    action_type;

                return eager_future<action_type>(gid, target);
            }

            template <typename Action>
            static void init(
                naming::id_type const & gid
              , naming::id_type const & target
            )
            {
                init_async<Action>(gid, target).get();
            }

#define HPX_LCOS_DATAFLOW_M0(Z, N, D)                                           \
            template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename A)>     \
            static promise<void>                                                \
            init_async(                                                         \
                naming::id_type const & gid                                     \
              , naming::id_type const & target                                  \
              , BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)                    \
            )                                                                   \
            {                                                                   \
                typedef                                                         \
                    typename server::init_action<                               \
                        Action                                                  \
                      , BOOST_PP_ENUM_PARAMS(N, A)                              \
                    >                                                           \
                    action_type;                                                \
                                                                                \
                return                                                          \
                    eager_future<action_type>(                                  \
                        gid                                                     \
                      , target                                                  \
                      , BOOST_PP_ENUM_PARAMS(N, a)                              \
                    );                                                          \
            }                                                                   \
                                                                                \
            template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename A)>     \
            static void init(                                                   \
                naming::id_type const & gid                                     \
              , naming::id_type const & target                                  \
              , BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)                    \
            )                                                                   \
            {                                                                   \
                init_async<Action>(                                             \
                    gid                                                         \
                  , target                                                      \
                  , BOOST_PP_ENUM_PARAMS(N, a)                                  \
                ).get();                                                        \
            }                                                                   \
    /**/
        BOOST_PP_REPEAT_FROM_TO(
            1
          , HPX_ACTION_ARGUMENT_LIMIT
          , HPX_LCOS_DATAFLOW_M0
          , _
        )
#undef HPX_LCOS_DATAFLOW_M0
#endif

            static promise<void>
            connect_async(
                naming::id_type const & gid
              , naming::id_type const & target
            )
            {
                typedef server_type::connect_action action_type;
                return eager_future<action_type>(gid, target);
            }

            static void connect(
                naming::id_type const & gid
              , naming::id_type const & target
            )
            {
                connect_async(gid, target).get();
            }
        };
    }
}}
#endif
