//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SHENEOS_PARTITION3D_AUG_08_2011_1223PM)
#define HPX_SHENEOS_PARTITION3D_AUG_08_2011_1223PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/runtime/components/client_base.hpp>

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
        // create a new partition instance and initialize it synchronously
        partition3d(std::string const& datafilename, dimension const& dimx, 
                dimension const& dimy, dimension const& dimz) 
          : base_type(sheneos::stubs::partition3d::create_sync(hpx::find_here()))
        {
            init(datafilename, dimx, dimy, dimz);
        }
        partition3d(hpx::naming::id_type gid, std::string const& datafilename, 
                dimension const& dimx, dimension const& dimy, dimension const& dimz) 
          : base_type(sheneos::stubs::partition3d::create_sync(gid))
        {
            init(datafilename, dimx, dimy, dimz);
        }
        partition3d(hpx::naming::id_type gid) 
          : base_type(gid) 
        {}

        // initialize this partition
        hpx::lcos::promise<void>
        init_async(std::string const& datafilename, 
            dimension const& dimx, dimension const& dimy, dimension const& dimz)
        {
            return stubs::partition3d::init_async(this->gid_, datafilename, 
                dimx, dimy, dimz);
        }

        void init(std::string const& datafilename, 
            dimension const& dimx, dimension const& dimy, dimension const& dimz)
        {
            stubs::partition3d::init(this->gid_, datafilename, dimx, dimy, dimz);
        }

        // ask this partition to interpolate, note that value must be in the
        // range valid for this partition
        hpx::lcos::promise<std::vector<double> >
        interpolate_async(double ye, double temp, double rho, 
            boost::uint32_t eosvalues)
        {
            return stubs::partition3d::interpolate_async(this->gid_, 
                ye, temp, rho, eosvalues);
        }

        std::vector<double> interpolate(double ye, double temp, double rho,
            boost::uint32_t eosvalues)
        {
            return stubs::partition3d::interpolate(this->gid_, 
                ye, temp, rho, eosvalues);
        }
    };
}

#endif


