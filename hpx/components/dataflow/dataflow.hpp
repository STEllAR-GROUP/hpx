//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef EXAMPLES_BRIGHT_FUTURE_DATAFLOW_HPP
#define EXAMPLES_BRIGHT_FUTURE_DATAFLOW_HPP

#include <hpx/components/dataflow/dataflow_base.hpp>
#include <hpx/components/dataflow/dataflow_fwd.hpp>

namespace hpx { namespace lcos 
{
    namespace detail
    {
        template <typename Action>
        struct action_wrapper
        {
            typedef Action type;

            template <typename Archive>
            void serialize(Archive & ar, unsigned)
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
        static inline lcos::promise<naming::id_type, naming::gid_type>
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
                /*async_callback<create_component_action>(
                    [target](naming::id_type const & gid)
                    {
                        LLCO_(info)
                            << "dataflow: created component "
                            << gid
                            << " with target "
                            << target
                            ;
                    }
                  ,*/ naming::get_locality_from_id(target)
                  , stub_type::get_component_type()
                  , detail::action_wrapper<Action>()
                  , target
                );
        }

        explicit dataflow(naming::id_type const & target)
            : base_type(
                create_component(target))
        {
        }

#define HPX_LCOS_DATAFLOW_M0(Z, N, D)                                           \
        template <BOOST_PP_ENUM_PARAMS(N, typename A)>                          \
        static inline lcos::promise<naming::id_type, naming::gid_type>          \
        create_component(naming::id_type const & target                         \
          , BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)                        \
        )                                                                       \
        {                                                                       \
            typedef                                                             \
                typename BOOST_PP_CAT(                                          \
                    hpx::components::server::create_one_component_action        \
                  , BOOST_PP_ADD(N, 2)                                          \
                )<                                                              \
                    components::managed_component<server::dataflow>             \
                  , detail::action_wrapper<Action>                              \
                  , naming::id_type                                             \
                  , BOOST_PP_ENUM_PARAMS(N, A)                                  \
                >::type                                                         \
                create_component_action;                                        \
            return                                                              \
                async<create_component_action>(                                 \
                    naming::get_locality_from_id(target)                        \
                  , stub_type::get_component_type()                             \
                  , detail::action_wrapper<Action>()                            \
                  , target                                                      \
                  , BOOST_PP_ENUM_PARAMS(N, a)                                  \
                );                                                              \
        }                                                                       \
        template <BOOST_PP_ENUM_PARAMS(N, typename A)>                          \
        dataflow(                                                               \
            naming::id_type const & target                                      \
          , BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)                        \
        )                                                                       \
            : base_type(                                                        \
                create_component(target                                         \
                  , BOOST_PP_ENUM_PARAMS(N, a)                                  \
                )                                                               \
                , BOOST_PP_ENUM_PARAMS(N, a)                                    \
            )                                                                   \
        {                                                                       \
        }                                                                       \
    /*
                async_callback<create_component_action>(                        \
                    [target, BOOST_PP_ENUM_PARAMS(N, a)]                        \
                    (naming::id_type const & gid)                               \
                    {                                                           \
                        LLCO_(info)                                             \
                            << "dataflow: created component "                   \
                            << gid                                              \
                            << " with target "                                  \
                            << target                                           \
                            ;                                                   \
                    }                                                           \
    */
    /**/
        BOOST_PP_REPEAT_FROM_TO(
            1
          , BOOST_PP_SUB(HPX_ACTION_ARGUMENT_LIMIT, 3)
          , HPX_LCOS_DATAFLOW_M0
          , _
        )
#undef HPX_LCOS_DATAFLOW_M0
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
