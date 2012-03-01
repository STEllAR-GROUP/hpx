//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SHENEOS_PARTITION3D_AUG_08_2011_1219PM)
#define HPX_SHENEOS_PARTITION3D_AUG_08_2011_1219PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

#include "../server/partition3d.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace sheneos { namespace stubs
{
    struct partition3d
      : hpx::components::stubs::stub_base<sheneos::server::partition3d>
    {
        ///////////////////////////////////////////////////////////////////////
        static hpx::lcos::promise<void>
        init_async(hpx::naming::id_type const& gid, std::string const& datafilename,
            dimension const& dimx, dimension const& dimy, dimension const& dimz)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise.
            typedef sheneos::server::partition3d::init_action action_type;
            return hpx::lcos::async<action_type>(
                gid, datafilename, dimx, dimy, dimz);
        }

        static void init(hpx::naming::id_type const& gid,
            std::string const& datafilename, dimension const& dimx,
            dimension const& dimy, dimension const& dimz)
        {
            init_async(gid, datafilename, dimx, dimy, dimz).get();
        }

        ///////////////////////////////////////////////////////////////////////
        static hpx::lcos::promise<std::vector<double> >
        interpolate_async(hpx::naming::id_type const& gid,
            double ye, double temp, double rho, boost::uint32_t eosvalues)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise.
            typedef sheneos::server::partition3d::interpolate_action action_type;
            return hpx::lcos::async<action_type>(gid, ye, temp, rho, eosvalues);
        }

        static std::vector<double> interpolate(hpx::naming::id_type const& gid,
            double ye, double temp, double rho, boost::uint32_t eosvalues)
        {
            return interpolate_async(gid, ye, temp, rho, eosvalues).get();
        }

        ///////////////////////////////////////////////////////////////////////
        static hpx::lcos::promise<double>
        interpolate_one_async(hpx::naming::id_type const& gid,
            double ye, double temp, double rho, boost::uint32_t eosvalues)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise.
            typedef sheneos::server::partition3d::interpolate_one_action action_type;
            return hpx::lcos::async<action_type>(gid, ye, temp, rho, eosvalues);
        }

        static double interpolate_one(hpx::naming::id_type const& gid,
            double ye, double temp, double rho, boost::uint32_t eosvalues)
        {
            return interpolate_one_async(gid, ye, temp, rho, eosvalues).get();
        }

        ///////////////////////////////////////////////////////////////////////
        static hpx::lcos::promise<std::vector<double> >
        interpolate_one_bulk_async(hpx::naming::id_type const& gid,
            std::vector<double> const& ye, std::vector<double> const& temp,
            std::vector<double> const& rho, boost::uint32_t eosvalues)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise.
            typedef sheneos::server::partition3d::interpolate_one_bulk_action
                action_type;
            return hpx::lcos::async<action_type>(gid, ye, temp, rho, eosvalues);
        }

        static std::vector<double> interpolate_one_bulk(
            hpx::naming::id_type const& gid, std::vector<double> const& ye,
            std::vector<double> const& temp, std::vector<double> const& rho,
            boost::uint32_t eosvalues)
        {
            return interpolate_one_bulk_async(gid, ye, temp, rho, eosvalues).get();
        }
    };
}}

#endif

