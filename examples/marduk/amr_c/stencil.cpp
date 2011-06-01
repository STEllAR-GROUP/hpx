//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Matthew Anderson
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/lcos/future_wait.hpp>

#include <boost/foreach.hpp>

#include <math.h>

#include "stencil.hpp"
#include "logging.hpp"
#include "stencil_data.hpp"
#include "stencil_functions.hpp"
#include "stencil_data_locking.hpp"
#include "../mesh/unigrid_mesh.hpp"

#if defined(RNPL_FOUND)
#include <sdf.h>
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    ///////////////////////////////////////////////////////////////////////////
    stencil::stencil()
      : numsteps_(0)
    {
    }

    inline bool 
    stencil::floatcmp(double_type const& x1, double_type const& x2) {
      // compare two floating point numbers
      static double_type const epsilon = 1.e-8;
      if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
        // the numbers are close enough for coordinate comparison
        return true;
      } else {
        return false;
      }
    }

    inline bool 
    stencil::floatcmp_le(double_type const& x1, double_type const& x2) {
      // compare two floating point numbers
      static double_type const epsilon = 1.e-8;

      if ( x1 < x2 ) return true;

      if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
        // the numbers are close enough for coordinate comparison
        return true;
      } else {
        return false;
      }
    }

    inline bool 
    stencil::floatcmp_ge(double_type const& x1, double_type const& x2) {
      // compare two floating point numbers
      static double_type const epsilon = 1.e-8;

      if ( x1 > x2 ) return true;

      if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
        // the numbers are close enough for coordinate comparison
        return true;
      } else {
        return false;
      }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Implement actual functionality of this stencil
    // Compute the result value for the current time step
    int stencil::eval(naming::id_type const& result, 
        std::vector<naming::id_type> const& gids, std::size_t row, std::size_t column,
        parameter const& par)
    {
        // make sure all the gids are looking valid
        if (result == naming::invalid_id)
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "stencil::eval", "result gid is invalid");
            return -1;
        }


        // this should occur only after result has been delivered already
        BOOST_FOREACH(naming::id_type gid, gids)
        {
            if (gid == naming::invalid_id)
                return -1;
        }

        // get all input and result memory_block_data instances
        std::vector<access_memory_block<stencil_data> > val;
        access_memory_block<stencil_data> resultval = 
            get_memory_block_async(val, gids, result);

        // lock all user defined data elements, will be unlocked at function exit
        scoped_values_lock<lcos::mutex> l(resultval, val); 

        resultval.get() = val[0].get();
        resultval->level_ = val[0]->level_;

        // Check if this is a prolongation/restriction step
        if ( (row+5)%3 == 0 || ( par->allowedl == 0 && row == 0 ) ) {
          // This is a prolongation/restriction step
          resultval->timestep_ = val[0]->timestep_;
        } else {
          resultval->timestep_ = val[0]->timestep_ + 1.0/pow(2.0,int(val[0]->level_));
        }

#if defined(RNPL_FOUND)
        // Output
        if (par->loglevel > 1 && fmod(resultval->timestep_, par->output) < 1.e-6) {
          std::vector<double> x,y,z,phi,d1phi,d2phi,d3phi,d4phi;   
          applier::applier& appl = applier::get_applier();
          naming::id_type this_prefix = appl.get_runtime_support_gid();
          int locality = get_prefix_from_id( this_prefix );
          double datatime;

          int gi = par->item2gi[column];
          int nx = par->gr_nx[gi];
          int ny = par->gr_ny[gi];
          int nz = par->gr_nz[gi];
          for (int i=0;i<nx;i++) {
            x.push_back(par->gr_minx[gi] + par->gr_h[gi]*i);
          }
          for (int i=0;i<ny;i++) {
            x.push_back(par->gr_miny[gi] + par->gr_h[gi]*i);
          }
          for (int i=0;i<nz;i++) {
            x.push_back(par->gr_minz[gi] + par->gr_h[gi]*i);
          }

          //datatime = resultval->timestep_*par->gr_h[gi]*par->lambda;
          datatime = resultval->timestep_;
          for (int k=0;k<nz;k++) {
          for (int j=0;j<ny;j++) {
          for (int i=0;i<nx;i++) {
            phi.push_back(resultval->value_[i+nx*(j+ny*k)].phi[0][0]);
            d1phi.push_back(resultval->value_[i+nx*(j+ny*k)].phi[0][1]);
            d2phi.push_back(resultval->value_[i+nx*(j+ny*k)].phi[0][2]);
            d3phi.push_back(resultval->value_[i+nx*(j+ny*k)].phi[0][3]);
            d4phi.push_back(resultval->value_[i+nx*(j+ny*k)].phi[0][4]);
          } } }
          int shape[3];
          char cnames[80] = { "x|y|z" };
          char phi_name[80];
          sprintf(phi_name,"%dphi",locality);
          char phi1_name[80];
          sprintf(phi1_name,"%dd1phi",locality);
          char phi2_name[80];
          sprintf(phi2_name,"%dd2phi",locality);
          char phi3_name[80];
          sprintf(phi3_name,"%dd3phi",locality);
          char phi4_name[80];
          sprintf(phi4_name,"%dd4phi",locality);
          shape[0] = nx;
          shape[1] = ny;
          shape[2] = nz;
          gft_out_full(phi_name,datatime,shape,cnames,3,&*x.begin(),&*phi.begin());
          gft_out_full(phi1_name,datatime,shape,cnames,3,&*x.begin(),&*d1phi.begin());
          gft_out_full(phi2_name,datatime,shape,cnames,3,&*x.begin(),&*d2phi.begin());
          gft_out_full(phi3_name,datatime,shape,cnames,3,&*x.begin(),&*d3phi.begin());
          gft_out_full(phi4_name,datatime,shape,cnames,3,&*x.begin(),&*d4phi.begin());
        }
#endif

        //std::cout << " TEST row " << row << " column " << column << " timestep " << resultval->timestep_ << std::endl;
        if ( val[0]->timestep_ >= par->nt0-2 ) {
          return 0;
        }
        return 1;
    }

    hpx::actions::manage_object_action<stencil_data> const manage_stencil_data =
        hpx::actions::manage_object_action<stencil_data>();

    ///////////////////////////////////////////////////////////////////////////
    naming::id_type stencil::alloc_data(int item, int maxitems, int row,
                           parameter const& par)
    {
        naming::id_type here = applier::get_applier().get_runtime_support_gid();
        naming::id_type result = components::stubs::memory_block::create(
            here, sizeof(stencil_data), manage_stencil_data);

        if (-1 != item) {
            // provide initial data for the given data value 
            access_memory_block<stencil_data> val(
                components::stubs::memory_block::checkout(result));

            // call provided (external) function
            generate_initial_data(val.get_ptr(), item, maxitems, row, *par.p);

            if (log_ && par->loglevel > 1)         // send initial value to logging instance
                stubs::logging::logentry(log_, val.get(), row,0, par);
        }
        return result;
    }

    void stencil::init(std::size_t numsteps, naming::id_type const& logging)
    {
        numsteps_ = numsteps;
        log_ = logging;
    }

}}}

