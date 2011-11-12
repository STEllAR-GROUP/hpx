
//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef EXAMPLES_BRIGHT_FUTURE_DATAFLOW_HPP
#define EXAMPLES_BRIGHT_FUTURE_DATAFLOW_HPP

#include <examples/bright_future/dataflow/dataflow_base.hpp>
#include <examples/bright_future/dataflow/dataflow_fwd.hpp>

namespace hpx { namespace lcos {

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

        typedef hpx::components::server::runtime_support::create_component_action create_component_action;


        // MSVC chokes on having the lambda in the member initializer list below
        static inline lcos::promise<naming::id_type, naming::gid_type>
        create_component(naming::id_type target)
        {
            return async_callback<create_component_action>(
                    [target](naming::id_type const & gid) mutable
                    {
                        typedef hpx::lcos::server::init_action<Action> action_type;
                        applier::apply<action_type>(gid, target);
                        target = naming::invalid_id;
                    }
                  , naming::get_locality_from_id(target)
                  , stub_type::get_component_type()
                  , 1
                );
        }

        explicit dataflow(naming::id_type const & target)
            : base_type(create_component(target))
        {
            //this->get_gid();
        }

#define HPX_LCOS_DATAFLOW_M0(Z, N, D)                                           \
        template <BOOST_PP_ENUM_PARAMS(N, typename A)>                          \
        static inline lcos::promise<naming::id_type, naming::gid_type>          \
        create_component(naming::id_type target                                 \
          , BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)                        \
        )                                                                       \
        {                                                                       \
            return async_callback<create_component_action>(                     \
                    [=](naming::id_type const & gid) mutable                    \
                    {                                                           \
                        typedef hpx::lcos::server::init_action<                 \
                            Action, BOOST_PP_ENUM_PARAMS(N, A)> action_type;    \
                        applier::apply<action_type>(gid, target                 \
                          , BOOST_PP_ENUM_PARAMS(N, a));                        \
                        target = naming::invalid_id;                            \
                    }                                                           \
                  , naming::get_locality_from_id(target)                        \
                  , stub_type::get_component_type()                             \
                  , 1                                                           \
                );                                                              \
        }                                                                       \
        template <BOOST_PP_ENUM_PARAMS(N, typename A)>                          \
        dataflow(                                                               \
            naming::id_type const & target                                      \
          , BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)                        \
        )                                                                       \
            : base_type(create_component(target                                 \
                , BOOST_PP_ENUM_PARAMS(N, a))                                   \
              )                                                                 \
        {                                                                       \
        }
            /*
            this->get_gid();                                                    \
             */
    /**/
        BOOST_PP_REPEAT_FROM_TO(
            1
          , HPX_ACTION_ARGUMENT_LIMIT
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
