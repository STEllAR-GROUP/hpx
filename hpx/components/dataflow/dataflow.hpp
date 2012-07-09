//  Copyright (c) 2011-2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#ifndef HPX_LCOS_DATAFLOW_HPP
#define HPX_LCOS_DATAFLOW_HPP

#include <hpx/config.hpp>
#include <hpx/util/move.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/components/dataflow/dataflow_base.hpp>
#include <hpx/components/dataflow/dataflow_fwd.hpp>
#include <hpx/include/async.hpp>

namespace hpx { namespace lcos
{
    namespace detail
    {
        template <typename Action>
        struct action_wrapper
        {
            typedef Action type;

            template <typename Archive>
            void serialize(Archive &, unsigned)
            {}
        };
    }

    template <
        typename Action
      , typename Result
      , typename DirectExecute
    >
    struct dataflow
        : dataflow_base<Result, typename Action::result_type>
    {
        typedef typename Action::result_type remote_result_type;
        typedef Result result_type;
        typedef
            dataflow_base<Result, typename Action::result_type>
            base_type;

        typedef stubs::dataflow stub_type;

        dataflow() {}

        ~dataflow()
        {
            LLCO_(info)
                << "~dataflow::dataflow() ";
        }

        // MSVC chokes on having the lambda in the member initializer list below
        static inline lcos::future<naming::id_type, naming::gid_type>
        create_component(naming::id_type const & target)
        {
            typedef
                typename hpx::components::server::create_one_component_action2<
                    components::managed_component<server::dataflow>
                  , detail::action_wrapper<Action>
                  , naming::id_type
                >::type
                create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , stub_type::get_component_type()
                  , detail::action_wrapper<Action>()
                  , target
                );
        }

        explicit dataflow(naming::id_type const & target)
            : base_type(create_component(target))
        {
        }

#define HPX_A(z, n, _)                                                        \
        BOOST_PP_COMMA_IF(n)                                                  \
            typename boost::remove_const<                                     \
                typename hpx::util::detail::remove_reference<                 \
                    BOOST_PP_CAT(A, n)>::type>::type const &                  \
    /**/

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (                                                                         \
        3                                                                     \
      , (                                                                     \
            1                                                                 \
          , BOOST_PP_SUB(HPX_ACTION_ARGUMENT_LIMIT, 4)                        \
          , "hpx/components/dataflow/dataflow.hpp"                            \
        )                                                                     \
    )                                                                         \
    /**/

#include BOOST_PP_ITERATE()

#undef HPX_A
    private:

        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
}}

#endif

#else

#define N BOOST_PP_ITERATION()

        template <BOOST_PP_ENUM_PARAMS(N, typename A)>
        static inline lcos::future<naming::id_type, naming::gid_type>
        create_component(naming::id_type const & target
          , HPX_ENUM_FWD_ARGS(N, A, a)
          , boost::mpl::false_
        )
        {
            typedef
                typename BOOST_PP_CAT(
                    hpx::components::server::create_one_component_action
                  , BOOST_PP_ADD(N, 2)
                )<
                    components::managed_component<server::dataflow>
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , BOOST_PP_REPEAT(N, HPX_A, _)
                >::type
                create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , stub_type::get_component_type()
                  , detail::action_wrapper<Action>()
                  , target
                  , HPX_ENUM_FORWARD_ARGS(N, A, a)
                );
        }

        template <BOOST_PP_ENUM_PARAMS(N, typename A)>
        static inline lcos::future<naming::id_type>
        create_component(naming::id_type const & target
          , HPX_ENUM_FWD_ARGS(N, A, a)
          , boost::mpl::true_
        )
        {
            typedef
                typename BOOST_PP_CAT(
                    hpx::components::server::create_one_component_direct_action
                  , BOOST_PP_ADD(N, 2)
                )<
                    components::managed_component<server::dataflow>
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , BOOST_PP_REPEAT(N, HPX_A, _)
                >::type
                create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , stub_type::get_component_type()
                  , detail::action_wrapper<Action>()
                  , target
                  , HPX_ENUM_FORWARD_ARGS(N, A, a)
                );
        }

        template <BOOST_PP_ENUM_PARAMS(N, typename A)>
        dataflow(
            naming::id_type const & target
          , HPX_ENUM_FWD_ARGS(N, A, a)
        )
            : base_type(
                create_component(target
                  , HPX_ENUM_FORWARD_ARGS(N, A, a)
                  , typename Action::direct_execution()
                )
            )
        {
        }

#endif
