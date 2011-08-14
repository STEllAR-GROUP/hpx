//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SHENEOS_AUG_08_2011_1223PM)
#define HPX_SHENEOS_AUG_08_2011_1223PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future_value.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

#include <vector>

#include "stubs/partition3d.hpp"
#include "configuration.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace sheneos 
{
    // This class encapsulates N partitions and dispatches requests based on 
    // the given values to get interpolated results for.
    class HPX_COMPONENT_EXPORT sheneos 
    {
    public:
        // Initialize an interpolation object which is not connected to any
        // interpolation partitions. Call connect() to attach a running 
        // instance or create() to create a new one.
        sheneos();

        // Destruct an interpolation object instance. If this instance was 
        // initialized using create(), this will unregister the symbolic name
        //  of the associated partition objects.
        ~sheneos();

        // Create a new interpolation instance and initialize it synchronously.
        // Passing -1 as the second argument creates exactly one partition 
        // instance on each available locality. Register this interpolation
        // object with the given symbolic name
        void create(std::string const& datafilename, 
            std::string const& symbolic_name_base = 
                "/sheneos/220r_180t_50y_extT_analmu_20100322_SVNr28",
            std::size_t num_instances = std::size_t(-1));

        // Connect to an existing interpolation object with the given symbolic 
        // name.
        void connect(std::string symbolic_name_base = 
            "/sheneos/220r_180t_50y_extT_analmu_20100322_SVNr28");

        // Return the interpolated function values for the given argument. This
        // function dispatches to the proper partition for the actual 
        // interpolation.
        hpx::lcos::future_value<std::vector<double> >
        interpolate_async(double ye, double temp, double rho, 
            boost::uint32_t eosvalues = server::partition3d::small_api_values)
        {
            return stubs::partition3d::interpolate_async(
                get_gid(ye, temp, rho), ye, temp, rho, eosvalues);
        }

        std::vector<double> 
        interpolate(double ye, double temp, double rho, 
            boost::uint32_t eosvalues = server::partition3d::small_api_values)
        {
            return stubs::partition3d::interpolate(
                get_gid(ye, temp, rho), ye, temp, rho, eosvalues);
        }

    private:
        // map the given value to the gid of the partition responsible for the
        // interpolation
        hpx::naming::id_type get_gid(double ye, double temp, double rho);

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


