
//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef EXAMPLES_BRIGHT_FUTURE_DATAFLOW_TRIGGER_HPP
#define EXAMPLES_BRIGHT_FUTURE_DATAFLOW_TRIGGER_HPP

#include <examples/bright_future/dataflow/dataflow_trigger_fwd.hpp>
#include <examples/bright_future/dataflow/dataflow_base.hpp>
#include <examples/bright_future/dataflow/stubs/dataflow_trigger.hpp>
#include <hpx/traits/handle_gid.hpp>

namespace hpx { namespace lcos {
    struct dataflow_trigger
        : dataflow_base<void>
    {
        typedef traits::promise_remote_result<void>::type remote_result_type;
        typedef void result_type;

        typedef dataflow_base<void> base_type;

        typedef stubs::dataflow_trigger stub_type;

        dataflow_trigger() {}

        // MSVC chokes on having the lambda in the member initializer list below
        static inline lcos::promise<naming::id_type, naming::gid_type>
        create_component(
            naming::id_type const & id
          , std::vector<dataflow_base<void> > const & trigger
        )
        {
            typedef
                hpx::components::server::create_one_component_action1<
                    components::managed_component<server::dataflow_trigger>
                  , std::vector<dataflow_base<void> >
                >::type
                create_component_action;
            return
                async<create_component_action>(
                /*async_callback<create_component_action>(
                    [id, trigger](naming::id_type const & gid)
                    {
                        LLCO_(info)
                            << "dataflow_trigger: created component "
                            << gid
                            ;
                    }
                  ,*/ naming::get_locality_from_id(id)
                  , stub_type::get_component_type()
                  , trigger
                );
        }

        dataflow_trigger(
            naming::id_type const & id
          , std::vector<dataflow_base<void> > const & trigger
        )
            : base_type(
                create_component(id, trigger)
              , trigger
            )
        {}

        ~dataflow_trigger()
        {
            LLCO_(info)
                << "~dataflow_trigger::dataflow_trigger() ";
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
