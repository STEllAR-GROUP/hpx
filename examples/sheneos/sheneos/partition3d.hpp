//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SHENEOS_PARTITION3D_AUG_08_2011_1223PM)
#define HPX_SHENEOS_PARTITION3D_AUG_08_2011_1223PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/include/client.hpp>

#include <string>
#include <vector>

#include "stubs/partition3d.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace sheneos
{
    class partition3d
      : public hpx::components::client_base<
            partition3d, sheneos::stubs::partition3d>
    {
    private:
        typedef hpx::components::client_base<
            partition3d, sheneos::stubs::partition3d> base_type;

    public:
        /// Create a new partition instance locally and initialize it
        /// synchronously.
        partition3d(std::string const& datafilename, dimension const& dimx,
                dimension const& dimy, dimension const& dimz)
          : base_type(sheneos::stubs::partition3d::create(hpx::find_here()))
        {
            init(datafilename, dimx, dimy, dimz);
        }

        /// Create a new partition instance on a specific locality and
        /// initialize it synchronously.
        ///
        /// \param gid [in] The locality where the partition should be created.
        partition3d(hpx::naming::id_type const& gid, std::string const& datafilename,
                dimension const& dimx, dimension const& dimy, dimension const& dimz)
          : base_type(sheneos::stubs::partition3d::create(gid))
        {
            init(datafilename, dimx, dimy, dimz);
        }

        /// Connect to an existing partition instance.
        partition3d(hpx::naming::id_type const& gid)
          : base_type(gid)
        {}

        /// Initialize this partition asynchronously.
        hpx::lcos::future<void>
        init_async(std::string const& datafilename,
            dimension const& dimx, dimension const& dimy, dimension const& dimz)
        {
            return stubs::partition3d::init_async(this->get_id(), datafilename,
                dimx, dimy, dimz);
        }

        /// Initialize this partition synchronously.
        void init(std::string const& datafilename,
            dimension const& dimx, dimension const& dimy, dimension const& dimz)
        {
            stubs::partition3d::init(this->get_id(), datafilename, dimx, dimy, dimz);
        }

        /// Asynchronously perform an interpolation on this partition.
        ///
        /// \param ye        [in] Electron fraction.
        /// \param temp      [in] Temperature.
        /// \param rho       [in] Rest mass density of the plasma.
        /// \param eosvalues [in] The EOS values to interpolate. Must be
        ///                  in the range of this partition.
        hpx::lcos::future<std::vector<double> >
        interpolate_async(double ye, double temp, double rho,
            boost::uint32_t eosvalues)
        {
            return stubs::partition3d::interpolate_async(this->get_id(),
                ye, temp, rho, eosvalues);
        }

        /// Synchronously perform an interpolation on this partition.
        ///
        /// \param ye        [in] Electron fraction.
        /// \param temp      [in] Temperature.
        /// \param rho       [in] Rest mass density of the plasma.
        /// \param eosvalues [in] The EOS values to interpolate. Must be
        ///                  in the range of this partition.
        std::vector<double> interpolate(double ye, double temp, double rho,
            boost::uint32_t eosvalues)
        {
            return stubs::partition3d::interpolate(this->get_id(),
                ye, temp, rho, eosvalues);
        }

        /// Asynchronously perform an interpolation of one given field on this
        /// partition.
        ///
        /// \param ye        [in] Electron fraction.
        /// \param temp      [in] Temperature.
        /// \param rho       [in] Rest mass density of the plasma.
        /// \param eosvalue  [in] The EOS value to interpolate. Must be
        ///                  in the range of the given partition.
        hpx::lcos::future<double>
        interpolate_one_async(double ye, double temp, double rho,
            boost::uint32_t eosvalue)
        {
            return stubs::partition3d::interpolate_one_async(this->get_id(),
                ye, temp, rho, eosvalue);
        }

        /// Synchronously perform an interpolation of one given field on this
        /// partition.
        ///
        /// \param ye        [in] Electron fraction.
        /// \param temp      [in] Temperature.
        /// \param rho       [in] Rest mass density of the plasma.
        /// \param eosvalue  [in] The EOS value to interpolate. Must be
        ///                  in the range of the given partition.
        double interpolate_one(double ye, double temp, double rho,
            boost::uint32_t eosvalue)
        {
            return stubs::partition3d::interpolate_one(this->get_id(),
                ye, temp, rho, eosvalue);
        }

        /// Asynchronously perform an interpolation of one given field on this
        /// partition.
        ///
        /// \param cords     [in] triples of electron fractions, temperatures,
        ///                  and rest mass densities of the plasma.
        /// \param eosvalue  [in] The EOS value to interpolate. Must be
        ///                  in the range of the given partition.
        hpx::lcos::future<std::vector<double> >
        interpolate_one_bulk_async(std::vector<sheneos_coord> const& coords,
            boost::uint32_t eosvalue)
        {
            return stubs::partition3d::interpolate_one_bulk_async(this->get_id(),
                coords, eosvalue);
        }

        /// Synchronously perform an interpolation of one given field on this
        /// partition.
        ///
        /// \param cords     [in] triples of electron fractions, temperatures,
        ///                  and rest mass densities of the plasma.
        /// \param eosvalue  [in] The EOS value to interpolate. Must be
        ///                  in the range of the given partition.
        std::vector<double> interpolate_one_bulk(
            std::vector<sheneos_coord> const& coords, boost::uint32_t eosvalue)
        {
            return stubs::partition3d::interpolate_one_bulk(this->get_id(),
                coords, eosvalue);
        }


        /// Asynchronously perform an interpolation of one given field on this
        /// partition.
        ///
        /// \param cords     [in] triples of electron fractions, temperatures,
        ///                  and rest mass densities of the plasma.
        /// \param eosvalue  [in] The EOS value to interpolate. Must be
        ///                  in the range of the given partition.
        hpx::lcos::future<std::vector<std::vector<double> > >
        interpolate_bulk_async(std::vector<sheneos_coord> const& coords,
            boost::uint32_t eosvalue)
        {
            return stubs::partition3d::interpolate_bulk_async(this->get_id(),
                coords, eosvalue);
        }

        /// Synchronously perform an interpolation of one given field on this
        /// partition.
        ///
        /// \param cords     [in] triples of electron fractions, temperatures,
        ///                  and rest mass densities of the plasma.
        /// \param eosvalue  [in] The EOS value to interpolate. Must be
        ///                  in the range of the given partition.
        std::vector<std::vector<double> >
        interpolate_bulk(std::vector<sheneos_coord> const& coords,
            boost::uint32_t eosvalue)
        {
            return stubs::partition3d::interpolate_bulk(this->get_id(),
                coords, eosvalue);
        }
    };
}

#endif

