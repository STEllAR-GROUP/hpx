//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARTITION3D_AUG_06_2011_1017PM)
#define HPX_PARTITION3D_AUG_06_2011_1017PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

#include "../server/partition3d.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace interpolate3d { namespace stubs
{
    struct partition3d
      : hpx::components::stubs::stub_base<interpolate3d::server::partition3d>
    {
        ///////////////////////////////////////////////////////////////////////
        static hpx::lcos::future<void>
        init_async(hpx::naming::id_type const& gid, std::string const& datafilename,
            dimension const& dimx, dimension const& dimy, dimension const& dimz)
        {
            // Create a future, execute the required action,
            // we simply return the initialized future, the caller needs
            // to call get() on the return value to obtain the result
            typedef interpolate3d::server::partition3d::init_action action_type;
            return hpx::async<action_type>(
                gid, datafilename, dimx, dimy, dimz);
        }

        static void init(hpx::naming::id_type const& gid,
            std::string const& datafilename, dimension const& dimx,
            dimension const& dimy, dimension const& dimz)
        {
            init_async(gid, datafilename, dimx, dimy, dimz).get();
        }

        ///////////////////////////////////////////////////////////////////////
        static hpx::lcos::future<double>
        interpolate_async(hpx::naming::id_type const& gid,
            double value_x, double value_y, double value_z)
        {
            // Create a future, execute the required action,
            // we simply return the initialized future, the caller needs
            // to call get() on the return value to obtain the result
            typedef interpolate3d::server::partition3d::interpolate_action action_type;
            return hpx::async<action_type>(gid, value_x, value_y, value_z);
        }

        static double interpolate(hpx::naming::id_type const& gid,
            double value_x, double value_y, double value_z)
        {
            return interpolate_async(gid, value_x, value_y, value_z).get();
        }
    };
}}

#endif


