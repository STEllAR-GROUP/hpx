//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_INTERPOLATE3D_AUG_04_2011_0340PM)
#define HPX_INTERPOLATE3D_AUG_04_2011_0340PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

#include <vector>

#include "stubs/partition3d.hpp"
#include "configuration.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace interpolate3d
{
    // This class encapsulates N partitions and dispatches requests based on
    // the given values to get interpolated results for.
    class HPX_COMPONENT_EXPORT interpolate3d
    {
    public:
        // Initialize an interpolation object which is not connected to any
        // interpolation partitions. Call connect() to attach a running
        // instance or create() to create a new one.
        interpolate3d();

        // Destruct an interpolation object instance. If this instance was
        // initialized using create(), this will unregister the symbolic name
        //  of the associated partition objects.
        ~interpolate3d();

        // Create a new interpolation instance and initialize it synchronously.
        // Passing -1 as the second argument creates exactly one partition
        // instance on each available locality. Register this interpolation
        // object with the given symbolic name
        void create(std::string const& datafilename,
            std::string const& symbolic_name_base = "/interpolate3d/gauss/",
            std::size_t num_instances = std::size_t(-1));

        // Connect to an existing interpolation object with the given symbolic
        // name.
        void connect(std::string symbolic_name_base = "/interpolate3d/gauss/");

        // Return the interpolated  function value for the given argument. This
        // function dispatches to the proper partition for the actual
        // interpolation.
        hpx::lcos::promise<double>
        interpolate_async(double value_x, double value_y, double value_z)
        {
            return stubs::partition3d::interpolate_async(
                get_gid(value_x, value_y, value_z), value_x, value_y, value_z);
        }

        double interpolate(double value_x, double value_y, double value_z)
        {
            return stubs::partition3d::interpolate(
                get_gid(value_x, value_y, value_z), value_x, value_y, value_z);
        }

    private:
        // map the given value to the gid of the partition responsible for the
        // interpolation
        hpx::naming::id_type get_gid(double value_x, double value_y, double value_z);

        // initialize the partitions and store the mappings
        typedef hpx::components::distributing_factory distributing_factory;
        typedef distributing_factory::async_create_result_type
            async_create_result_type;

        void fill_partitions(std::string const& datafilename,
            std::string symbolic_name_base, async_create_result_type future);
        std::size_t get_index(int d, double value);

    private:
        std::vector<hpx::naming::id_type> partitions_;
        double minval_[dimension::dim];
        double delta_[dimension::dim];
        std::size_t num_values_[dimension::dim];
        std::size_t num_partitions_per_dim_;
        bool was_created_;

        configuration cfg_;
    };
}

#endif


