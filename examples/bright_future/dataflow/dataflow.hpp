
//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef EXAMPLES_BRIGHT_FUTURE_DATAFLOW_HPP
#define EXAMPLES_BRIGHT_FUTURE_DATAFLOW_HPP

#include <examples/bright_future/dataflow/dataflow_base.hpp>
#include <examples/bright_future/dataflow/dataflow_fwd.hpp>

namespace hpx { namespace lcos {

    namespace detail {
        template <typename Action>
        struct apply_init_helper
        {
            naming::id_type target;
            apply_init_helper(naming::id_type const & target) : target(target) {}

            void operator()(naming::id_type const & gid) const
            {
                BOOST_ASSERT(gid);
                typedef hpx::lcos::server::init_action<Action> action_type;
                applier::apply<action_type>(gid, target);
            }
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
            
        typedef hpx::components::server::runtime_support::create_component_action create_component_action;

        explicit dataflow(naming::id_type const & target)
            : base_type(
                async_callback<create_component_action>(
                    [
                        target
                      , this
                    ](naming::gid_type const & gid)
                    {
                        typedef hpx::lcos::server::init_action<Action> action_type;
                        this->gid_ = naming::id_type(gid, naming::id_type::managed);
                        applier::apply<action_type>(this->gid_, target);
                    }
                  , naming::id_type(
                        naming::get_gid_from_prefix(
                            naming::get_prefix_from_id(
                                target
                            )
                        )
                      , naming::id_type::unmanaged
                    )
                  , stub_type::get_component_type()
                  , 1
                )
                /*
                stub_type::create_sync(
                    naming::get_gid_from_prefix(
                        naming::get_prefix_from_id(
                            target
                        )
                    )
                )
                */
            )
        {
            /*
            BOOST_ASSERT(this->get_gid());
            typedef hpx::lcos::server::init_action<Action> action_type;
            applier::apply<action_type>(this->get_gid(), target);
            */
        }

#define HPX_LCOS_DATAFLOW_M0(Z, N, D)                                           \
        template <BOOST_PP_ENUM_PARAMS(N, typename A)>                          \
        dataflow(                                                               \
            naming::id_type const & target                                      \
          , BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)                        \
        )                                                                       \
            : base_type(                                                        \
                async_callback<create_component_action>(                        \
                    [                                                           \
                        target                                                  \
                      , this                                                    \
                      , BOOST_PP_ENUM_PARAMS(N, a)                              \
                    ](naming::gid_type const & gid)                             \
                    {                                                           \
                        typedef hpx::lcos::server::init_action<Action, BOOST_PP_ENUM_PARAMS(N, A)> action_type; \
                        this->gid_ = naming::id_type(gid, naming::id_type::managed); \
                        applier::apply<action_type>(this->gid_, target, BOOST_PP_ENUM_PARAMS(N, a)); \
                    }                                                           \
                  , naming::id_type(                                            \
                        naming::get_gid_from_prefix(                            \
                            naming::get_prefix_from_id(                         \
                                target                                          \
                            )                                                   \
                        )                                                       \
                      , naming::id_type::unmanaged                              \
                    )                                                           \
                  , stub_type::get_component_type()                             \
                  , 1                                                           \
                )                                                               \
            )                                                                   \
        {}                                                                      \
        /*
                stub_type::create_sync(                                         \
                    naming::get_gid_from_prefix(                                \
                        naming::get_prefix_from_id(                             \
                            target                                              \
                        )                                                       \
                    )                                                           \
                )                                                               \
            )                                                                   \
        {                                                                       \
            BOOST_ASSERT(this->get_gid());                                      \
            typedef                                                             \
                hpx::lcos::server::init_action<                                 \
                    Action                                                      \
                  , BOOST_PP_ENUM_PARAMS(N, A)                                  \
                >                                                               \
                action_type;                                                    \
            applier::apply<action_type>(                                        \
                this->get_gid()                                                 \
              , target                                                          \
              , BOOST_PP_ENUM_PARAMS(N, a)                                      \
            );                                                                  \
        }                                                                       \
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
            ar & this->gid_;
            BOOST_ASSERT(this->get_gid());
        }
    };
}}

#endif
