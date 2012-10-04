//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARTITION_AUG_04_2011_1204PM)
#define HPX_PARTITION_AUG_04_2011_1204PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>

#include "../dimension.hpp"

namespace interpolate1d { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT partition
      : public hpx::components::simple_component_base<partition>
    {
    private:
        typedef hpx::components::simple_component_base<partition> base_type;

    public:
        partition();

        // components must contain a typedef for wrapping_type defining the
        // component type used to encapsulate instances of this component
        typedef partition wrapping_type;

        ///////////////////////////////////////////////////////////////////////
        // parcel action code: the action to be performed on the destination
        // object (the accumulator)
        enum actions
        {
            partition_init = 0,
            partition_interpolate = 1
        };

        // exposed functionality
        void init(std::string datafilename, dimension const&,
            std::size_t num_nodes);
        double interpolate(double value);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::action3<
          partition, partition_init, std::string, dimension const&, std::size_t,
            &partition::init
        > init_action;

        typedef hpx::actions::result_action1<
            partition, double, partition_interpolate, double,
            &partition::interpolate
        > interpolate_action;

    private:
        dimension dim_;
        boost::scoped_array<double> values_;
        double min_value_, max_value_, delta_;
    };
}}

HPX_REGISTER_ACTION_DECLARATION(
    interpolate1d::server::partition::init_action,
    partition_init_action);
HPX_REGISTER_ACTION_DECLARATION(
    interpolate1d::server::partition::interpolate_action,
    partition_interpolate_action);

#endif



