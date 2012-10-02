//  Copyright (c) 2011-2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DATAFLOW_TRIGGER_HPP
#define HPX_LCOS_DATAFLOW_TRIGGER_HPP

#include <hpx/components/dataflow/dataflow_trigger_fwd.hpp>
#include <hpx/components/dataflow/dataflow_base.hpp>
#include <hpx/components/dataflow/stubs/dataflow_trigger.hpp>

namespace hpx { namespace lcos 
{
    struct dataflow_trigger
        : dataflow_base<void>
    {
        typedef traits::promise_remote_result<void>::type remote_result_type;
        typedef void result_type;

        typedef dataflow_base<void> base_type;

        typedef stubs::dataflow_trigger stub_type;

        dataflow_trigger() {}

        // MSVC chokes on having the lambda in the member initializer list below
        static inline lcos::future<naming::id_type, naming::gid_type>
        create_component(
            naming::id_type const & id
          , std::vector<dataflow_base<void> > const & trigger
        )
        {
            typedef
                hpx::components::server::runtime_support::
                    create_component_action1<
                        components::managed_component<server::dataflow_trigger>
                      , std::vector<dataflow_base<void> > const &
                    >::type
                create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(id)
                  , trigger
                );
        }
        
        /*
        static inline lcos::future<naming::id_type>
        create_component(
            naming::id_type const & id
          , std::vector<dataflow_base<void> > && trigger
        )
        {
            typedef
                hpx::components::server::create_component_action1<
                    components::managed_component<server::dataflow_trigger>
                  , std::vector<dataflow_base<void> >
                >::type
                create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(id)
                  , boost::move(trigger)
                );
        }
        */

        dataflow_trigger(
            naming::id_type const & id
          , std::vector<dataflow_base<void> > const & trigger
        )
            : base_type(create_component(id, trigger))
        {}

        /*
        dataflow_trigger(
            naming::id_type const & id
          , BOOST_RV_REF(std::vector<dataflow_base<void> >) trigger
        )
            : base_type(create_component(id, boost::move(trigger)))
        {}
        */

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

#endif
