//  Copyright (c) 2011-2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DATAFLOW_HPP
#define HPX_LCOS_DATAFLOW_HPP

#include <hpx/config.hpp>
#include <hpx/util/decay.hpp>
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
        : dataflow_base<Result, typename Action::remote_result_type>
    {
        typedef typename Action::remote_result_type remote_result_type;
        typedef Result result_type;
        typedef
            dataflow_base<Result, typename Action::remote_result_type>
            base_type;

        typedef stubs::dataflow stub_type;

        dataflow() {}

        ~dataflow()
        {
            LLCO_(info)
                << "~dataflow::dataflow() ";
        }

        explicit dataflow(naming::id_type const & target)
            : base_type(create_component(typename Action::direct_execution()
                , target))
        {
        }

        template <typename ...Ts>
        dataflow(
            naming::id_type const & target
          , Ts&&... vs
        )
            : base_type(create_component(typename Action::direct_execution()
                , target, std::forward<Ts>(vs)...))
        {
        }

        template <typename ...Ts>
        static inline lcos::future<naming::id_type>
        create_component(boost::mpl::false_
          , naming::id_type const & target, Ts&&... vs
        )
        {
            typedef components::server::create_component_action<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<Ts>::type...
                > create_component_action;

            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<Ts>(vs)...
                );
        }

        template <typename ...Ts>
        static inline lcos::future<naming::id_type>
        create_component(boost::mpl::true_
          , naming::id_type const & target, Ts&&... vs
        )
        {
            typedef components::server::create_component_direct_action<
                    server::dataflow
                  , detail::action_wrapper<Action> const &
                  , naming::id_type const &
                  , typename util::decay<Ts>::type...
                > create_component_action;

            return
                async<create_component_action>(
                    naming::get_locality_from_id(target)
                  , detail::action_wrapper<Action>()
                  , target
                  , std::forward<Ts>(vs)...
                );
        }

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
