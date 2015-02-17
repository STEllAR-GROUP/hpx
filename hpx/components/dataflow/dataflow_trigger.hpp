//  Copyright (c) 2011-2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DATAFLOW_TRIGGER_HPP
#define HPX_LCOS_DATAFLOW_TRIGGER_HPP

#include <hpx/components/dataflow/dataflow_trigger_fwd.hpp>
#include <hpx/components/dataflow/dataflow_base.hpp>
#include <hpx/components/dataflow/stubs/dataflow_trigger.hpp>

#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/base_object.hpp>

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
        static inline lcos::future<naming::id_type>
        create_component(
            naming::id_type const & id
          , std::vector<dataflow_base<void> > const & trigger
        )
        {
            typedef
                hpx::components::server::
                    create_component_action1<
                        server::dataflow_trigger
                      , std::vector<dataflow_base<void> > const &
                    >
                create_component_action;
            return
                async<create_component_action>(
                    naming::get_locality_from_id(id)
                  , trigger
                );
        }

        dataflow_trigger(
            naming::id_type const & id
          , std::vector<dataflow_base<void> > const & trigger
        )
            : base_type(create_component(id, trigger))
        {}

        ~dataflow_trigger()
        {
            LLCO_(info)
                << "~dataflow_trigger::dataflow_trigger() ";
        }

    private:

        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            using namespace hpx::serialization;

            base_object_type<dataflow_trigger, base_type> base
              = base_object<base_type>(*this);
            ar & base;
        }
    };
}}

#endif
