//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARTITION3D_AUG_06_2011_1015PM)
#define HPX_PARTITION3D_AUG_06_2011_1015PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>

#include "../dimension.hpp"

namespace interpolate3d { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT partition3d
      : public hpx::components::simple_component_base<partition3d>
    {
    private:
        typedef hpx::components::simple_component_base<partition3d> base_type;

        void init_dimension(std::string const& datafilename, int d,
            dimension const& dim, char const* name);
        std::size_t get_index(int d, double value);

    public:
        partition3d();

        // components must contain a typedef for wrapping_type defining the
        // component type used to encapsulate instances of this component
        typedef partition3d wrapping_type;

        ///////////////////////////////////////////////////////////////////////
        // parcel action code: the action to be performed on the destination
        // object (the accumulator)
        enum actions
        {
            partition3d_init = 0,
            partition3d_interpolate = 1
        };

        // exposed functionality
        void init(std::string const&, dimension const&, dimension const&,
            dimension const&);
        double interpolate(double valuex, double valuey, double valuez);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::action4<
            partition3d, partition3d_init, std::string const&,
            dimension const&, dimension const&, dimension const&,
            &partition3d::init
        > init_action;

        typedef hpx::actions::result_action3<
            partition3d, double, partition3d_interpolate, double, double,
            double, &partition3d::interpolate
        > interpolate_action;

    private:
        dimension dim_[dimension::dim];
        std::size_t ghost_left_[dimension::dim];
        double min_value_[dimension::dim], max_value_[dimension::dim];
        double delta_[dimension::dim];
        boost::scoped_array<double> values_;
    };
}}

HPX_REGISTER_ACTION_DECLARATION(interpolate3d::server::partition3d::init_action,
    interpolate3d_partition3d_init_action);
HPX_REGISTER_ACTION_DECLARATION(interpolate3d::server::partition3d::interpolate_action,
    interpolate3d_partition3d_interpolate_action);


#endif



