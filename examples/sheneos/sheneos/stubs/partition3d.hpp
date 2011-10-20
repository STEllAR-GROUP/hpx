//  Copyright (c) 2007-2011 Hartmut Kaiser
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
            // Create an eager_future, execute the required action,
            // we simply return the initialized promise, the caller needs
            // to call get() on the return value to obtain the result
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
            // Create an eager_future, execute the required action,
            // we simply return the initialized promise, the caller needs
            // to call get() on the return value to obtain the result
            typedef sheneos::server::partition3d::interpolate_action action_type;
            return hpx::lcos::async<action_type>(gid, ye, temp, rho, eosvalues);
        }

        static std::vector<double> interpolate(hpx::naming::id_type const& gid,
            double ye, double temp, double rho, boost::uint32_t eosvalues)
        {
            return interpolate_async(gid, ye, temp, rho, eosvalues).get();
        }
    };
}}

#endif


