
//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef EXAMPLES_BRIGHT_FUTURE_DATAFLOW_TRIGGER_HPP
#define EXAMPLES_BRIGHT_FUTURE_DATAFLOW_TRIGGER_HPP

#include <examples/bright_future/dataflow/dataflow_trigger_fwd.hpp>
#include <examples/bright_future/dataflow/dataflow_base.hpp>
#include <examples/bright_future/dataflow/stubs/dataflow_trigger.hpp>

namespace hpx { namespace lcos {
    struct dataflow_trigger
        : dataflow_base<void>
    {
        typedef traits::promise_remote_result<void>::type remote_result_type;
        typedef void result_type;

        typedef dataflow_base<void> base_type;
        
        typedef
            hpx::components::server::runtime_support::create_component_action
            create_component_action;
        
        typedef stubs::dataflow_trigger stub_type;
        
        dataflow_trigger() {}

        explicit dataflow_trigger(naming::id_type const & id)
            : base_type(
                async<create_component_action>(
                    naming::get_locality_from_id(id)
                  , stub_type::get_component_type()
                  , 1
                )
            )
        {}

        template <typename Result, typename RemoteResult>
        void add(dataflow_base<Result, RemoteResult> const & df)
        {
            typedef
                typename hpx::lcos::server::add_action<Result, RemoteResult>::type
                action_type;
            applier::apply<action_type>(get_gid(), df);
        }
    };
}}

namespace hpx { namespace traits {
        template <typename F>
        struct handle_gid<hpx::lcos::dataflow_trigger, F>
        {
            static bool call(
                hpx::lcos::dataflow_trigger const &df
              , F const& f
            )
            {
                f(df.get_gid());
                return true;
            }
        };
}}

#endif
