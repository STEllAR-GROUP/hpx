//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SHENEOS_AUG_08_2011_1223PM)
#define HPX_SHENEOS_AUG_08_2011_1223PM

#include <hpx/hpx.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "server/configuration.hpp"
#include "partition3d.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace sheneos
{
    /// This class encapsulates a set of \a sheneos::server::partition3d which
    /// spans a 3D space (ye, temp and rho) from the ShenEOS tables and
    /// dispatches interpolation requests to the correct partition.
    class HPX_COMPONENT_EXPORT interpolator
      : public hpx::components::client_base<
            interpolator,
            hpx::components::server::distributed_metadata_base<config_data>
        >
    {
    private:
        typedef hpx::components::server::distributed_metadata_base<
                config_data
            > config_data_type;
        typedef hpx::components::client_base<interpolator, config_data_type>
            base_type;

    public:
        /// Initialize an interpolator which is not connected to any
        /// interpolation partitions. Call connect() to attach a running
        /// instance or create() to create a new one.
        interpolator();

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
        interpolator(std::string const& datafilename,
            std::string const& symbolic_name_base = "/sheneos/interpolator",
            std::size_t num_instances = std::size_t(-1));

        interpolator(hpx::future<hpx::id_type> && id)
          : base_type(std::move(id))
        {}

        /// Destroy the interpolator. If this instance was initialized using
        /// create(), this will unregister the symbolic name of the associated
        /// partition objects.
        ~interpolator();

        /// Connect to an existing interpolation object with the given symbolic
        /// name.
        void connect(std::string symbolic_name_base =
            "/sheneos/interpolator");

        /// Asynchronously interpolate the function values for the given ye,
        /// temp and rho from the ShenEOS tables. This function dispatches to
        /// the proper partition for the actual interpolation.
        hpx::future<std::vector<double> >
        interpolate_async(double ye, double temp, double rho,
            std::uint32_t eosvalues = server::partition3d::small_api_values)  const
        {
            return get_partition(ye, temp, rho).
                interpolate_async(ye, temp, rho, eosvalues);
        }

        /// Synchronously interpolate the function values for the given ye,
        /// temp and rho from the ShenEOS tables. This function dispatches to
        /// the proper partition for the actual interpolation.
        std::vector<double> interpolate(double ye, double temp, double rho,
            std::uint32_t eosvalues = server::partition3d::small_api_values)  const
        {
            return get_partition(ye, temp, rho).
                interpolate(ye, temp, rho, eosvalues);
        }

        /// Asynchronously interpolate the function values for the given ye,
        /// temp and rho from the ShenEOS tables. This function dispatches to
        /// the proper partition for the actual interpolation.
        hpx::future<double>
        interpolate_one_async(double ye, double temp, double rho,
            std::uint32_t eosvalue)  const
        {
            return get_partition(ye, temp, rho).
                interpolate_one_async(ye, temp, rho, eosvalue);
        }

        /// Synchronously interpolate the function values for the given ye,
        /// temp and rho from the ShenEOS tables. This function dispatches to
        /// the proper partition for the actual interpolation.
        double interpolate_one(double ye, double temp, double rho,
            std::uint32_t eosvalue)  const
        {
            return get_partition(ye, temp, rho).
                interpolate_one(ye, temp, rho, eosvalue);
        }

        /// Asynchronously interpolate one function value for all the given ye,
        /// temp and rho from the ShenEOS tables. This function dispatches to
        /// the proper partitions for the actual interpolation using bulk
        /// operations.
        hpx::future<std::vector<double> >
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
        partition3d const& get_partition(double ye, double temp, double rho) const;

        partition3d const& get_partition(sheneos_coord const& c) const
        {
            return get_partition(c.ye_, c.temp_, c.rho_);
        }

        /// Initialize the partitions and store the mappings.
        void fill_partitions(std::string const& datafilename,
            std::string symbolic_name_base, std::size_t num_instances);

        std::size_t get_partition_index(std::size_t d, double value) const;

    private:
        std::vector<partition3d> partitions_;

        double minval_[dimension::dim];
        double maxval_[dimension::dim];
        double delta_[dimension::dim];
        std::size_t num_values_[dimension::dim];
        std::size_t num_partitions_per_dim_;
        bool was_created_;
    };
}

#endif

