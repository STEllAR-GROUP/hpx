//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SHENEOS_AUG_08_2011_1223PM)
#define HPX_SHENEOS_AUG_08_2011_1223PM

#include <hpx/hpx.hpp>
#include <hpx/include/components.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "stubs/partition3d.hpp"
#include "configuration.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace sheneos
{
    /// This class encapsulates a set of \a sheneos::server::partition3d which
    /// spans a 3D space (ye, temp and rho) from the ShenEOS tables and
    /// dispatches interpolation requests to the correct partition.
    class HPX_COMPONENT_EXPORT interpolator
    {
    public:
        /// Initialize an interpolator which is not connected to any
        /// interpolation partitions. Call connect() to attach a running
        /// instance or create() to create a new one.
        interpolator();

        /// Destroy the interpolator. If this instance was initialized using
        /// create(), this will unregister the symbolic name of the associated
        /// partition objects.
        ~interpolator();

        /// Create a new interpolator instance and initialize it synchronously.
        ///
        /// \param symbolic_name_base [in] The name for the new interpolator
        ///                           object.
        /// \param num_instances      [in] The number of
        ///                           \a sheneos::server::partition3d to create.
        ///                           If -1, then one
        ///                           \a sheneos::server::partition3d is created
        ///                           on each locality that supports the
        ///                           component.
        void create(std::string const& datafilename,
            std::string const& symbolic_name_base =
                "/sheneos/interpolator",
            std::size_t num_instances = std::size_t(-1));

        /// Connect to an existing interpolation object with the given symbolic
        /// name.
        void connect(std::string symbolic_name_base =
            "/sheneos/interpolator");

        /// Asynchronously interpolate the function values for the given ye,
        /// temp and rho from the ShenEOS tables. This function dispatches to
        /// the proper partition for the actual interpolation.
        hpx::lcos::future<std::vector<double> >
        interpolate_async(double ye, double temp, double rho,
            std::uint32_t eosvalues = server::partition3d::small_api_values)  const
        {
            return stubs::partition3d::interpolate_async(
                get_id(ye, temp, rho), ye, temp, rho, eosvalues);
        }

        /// Synchronously interpolate the function values for the given ye,
        /// temp and rho from the ShenEOS tables. This function dispatches to
        /// the proper partition for the actual interpolation.
        std::vector<double> interpolate(double ye, double temp, double rho,
            std::uint32_t eosvalues = server::partition3d::small_api_values)  const
        {
            return stubs::partition3d::interpolate(
                get_id(ye, temp, rho), ye, temp, rho, eosvalues);
        }

        /// Asynchronously interpolate the function values for the given ye,
        /// temp and rho from the ShenEOS tables. This function dispatches to
        /// the proper partition for the actual interpolation.
        hpx::lcos::future<double>
        interpolate_one_async(double ye, double temp, double rho,
            std::uint32_t eosvalue)  const
        {
            return stubs::partition3d::interpolate_one_async(
                get_id(ye, temp, rho), ye, temp, rho, eosvalue);
        }

        /// Synchronously interpolate the function values for the given ye,
        /// temp and rho from the ShenEOS tables. This function dispatches to
        /// the proper partition for the actual interpolation.
        double interpolate_one(double ye, double temp, double rho,
            std::uint32_t eosvalue)  const
        {
            return stubs::partition3d::interpolate_one(
                get_id(ye, temp, rho), ye, temp, rho, eosvalue);
        }

        /// Asynchronously interpolate one function value for all the given ye,
        /// temp and rho from the ShenEOS tables. This function dispatches to
        /// the proper partitions for the actual interpolation using bulk
        /// operations.
        hpx::lcos::future<std::vector<double> >
        interpolate_one_bulk_async(std::vector<sheneos_coord> const& coords,
            std::uint32_t eosvalue) const;

        /// Synchronously interpolate the function value for all the given ye,
        /// temp and rho from the ShenEOS tables. This function dispatches to
        /// the proper partition for the actual interpolation using bulk
        /// operations.
        std::vector<double>
        interpolate_one_bulk(std::vector<sheneos_coord> const& coords,
            std::uint32_t eosvalue) const
        {
            return interpolate_one_bulk_async(coords, eosvalue).get();
        }

        /// Asynchronously interpolate the function values for all the given ye,
        /// temp and rho from the ShenEOS tables. This function dispatches to
        /// the proper partition for the actual interpolation using bulk
        /// operations.
        hpx::lcos::future<std::vector<std::vector<double> > >
        interpolate_bulk_async(std::vector<sheneos_coord> const& coords,
            std::uint32_t eosvalues = server::partition3d::small_api_values) const;

        /// Synchronously interpolate the function values for all the given ye,
        /// temp and rho from the ShenEOS tables. This function dispatches to
        /// the proper partition for the actual interpolation using bulk
        /// operations.
        std::vector<std::vector<double> >
        interpolate_bulk(std::vector<sheneos_coord> const& coords,
            std::uint32_t eosvalues = server::partition3d::small_api_values) const
        {
            return interpolate_bulk_async(coords, eosvalues).get();
        }

        /// Find the minimum and maximum values of the given dimension.
        void get_dimension(dimension::type what, double& min, double& max)  const;

    private:
        /// Find the GID of the partition that contains the specified value.
        hpx::naming::id_type const& get_id(double ye, double temp, double rho) const;

        hpx::naming::id_type const& get_id(sheneos_coord const& c) const
        {
            return get_id(c.ye_, c.temp_, c.rho_);
        }

        typedef hpx::components::distributing_factory distributing_factory;
        typedef distributing_factory::async_create_result_type
            async_create_result_type;

        /// Initialize the partitions and store the mappings.
        void fill_partitions(std::string const& datafilename,
            std::string symbolic_name_base, async_create_result_type future);

        std::size_t get_partition_index(std::size_t d, double value) const;

    private:
        std::vector<hpx::naming::id_type> partitions_;
        double minval_[dimension::dim];
        double maxval_[dimension::dim];
        double delta_[dimension::dim];
        std::size_t num_values_[dimension::dim];
        std::size_t num_partitions_per_dim_;
        bool was_created_;

        configuration cfg_;
    };
}

#endif

