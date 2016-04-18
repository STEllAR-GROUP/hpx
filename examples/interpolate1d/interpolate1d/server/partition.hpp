//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARTITION_AUG_04_2011_1204PM)
#define HPX_PARTITION_AUG_04_2011_1204PM

#include <hpx/hpx.hpp>
#include <hpx/include/components.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include <string>

#include "../dimension.hpp"

namespace interpolate1d { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT partition
      : public hpx::components::component_base<partition>
    {
    private:
        typedef hpx::components::component_base<partition> base_type;

    public:
        partition();

        // components must contain a typedef for wrapping_type defining the
        // component type used to encapsulate instances of this component
        typedef partition wrapping_type;

        // exposed functionality
        void init(std::string datafilename, dimension const&,
            std::size_t num_nodes);
        double interpolate(double value);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        HPX_DEFINE_COMPONENT_ACTION(partition, init);
        HPX_DEFINE_COMPONENT_ACTION(partition, interpolate);

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



