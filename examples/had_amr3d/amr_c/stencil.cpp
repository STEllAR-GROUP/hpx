//  Copyright (c) 2007-2010 Hartmut Kaiser
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
#include "../amr/unigrid_mesh.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    ///////////////////////////////////////////////////////////////////////////
    stencil::stencil()
      : numsteps_(0)
    {
    }

    inline bool 
    stencil::floatcmp(had_double_type const& x1, had_double_type const& x2) {
      // compare two floating point numbers
      static had_double_type const epsilon = 1.e-8;
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
        Parameter const& par)
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

        // Here we give the coordinate value to the result (prior to sending it to the user)
        int compute_index;
        bool boundary = false;
        int bbox[6] = {0,0,0,0,0,0};   // initialize bounding box
#if 0
        if ( val.size()%2 == 0 ) {
          compute_index = (val.size())/2;
        } else {
          compute_index = (val.size()-1)/2;
        }
#endif
        compute_index = 0;

        if ( val.size() == 7 ) {
          compute_index = (val.size()-1)/2;
        } else {
          had_double_type dx = par->dx0;
          int tmp_index = column/par->nx0;
          int c = tmp_index/par->nx0;
          int b = tmp_index%par->nx0;
          int a = column - par->nx0*(b+c*par->nx0);
          had_double_type x = par->minx0 + a*dx*par->granularity;
          had_double_type y = par->minx0 + b*dx*par->granularity;
          had_double_type z = par->minx0 + c*dx*par->granularity;
          compute_index = -1;
          bbox[0] = 1; bbox[1] = 1; bbox[2] = 1;
          bbox[3] = 1; bbox[4] = 1; bbox[5] = 1;
          for (int i=0;i<val.size();i++) {
            // figure out bounding box
            if ( x > val[i]->x_[0] && 
                 floatcmp(y,val[i]->y_[0]) == 1 && 
                 floatcmp(z,val[i]->z_[0]) == 1 ) {
              bbox[0] = 0;
            }

            if ( x < val[i]->x_[0] && 
                 floatcmp(y,val[i]->y_[0]) == 1 && 
                 floatcmp(z,val[i]->z_[0]) == 1 ) {
              bbox[1] = 0;
            }

            if ( floatcmp(x,val[i]->x_[0]) && 
                 y > val[i]->y_[0]  && 
                 floatcmp(z,val[i]->z_[0]) == 1 ) {
              bbox[2] = 0;
            }

            if ( floatcmp(x,val[i]->x_[0]) && 
                 y < val[i]->y_[0]  && 
                 floatcmp(z,val[i]->z_[0]) == 1 ) {
              bbox[3] = 0;
            }

            if ( floatcmp(x,val[i]->x_[0]) && 
                 floatcmp(y,val[i]->y_[0]) == 1 && 
                 z > val[i]->z_[0] ) {
              bbox[4] = 0;
            }

            if ( floatcmp(x,val[i]->x_[0]) && 
                 floatcmp(y,val[i]->y_[0]) == 1 && 
                 z < val[i]->z_[0] ) {
              bbox[5] = 0;
            }


            if ( floatcmp(x,val[i]->x_[0]) == 1 && 
                 floatcmp(y,val[i]->y_[0]) == 1 && 
                 floatcmp(z,val[i]->z_[0]) == 1 ) {
              compute_index = i;
            }
          }
          if ( compute_index == -1 ) {
            BOOST_ASSERT(false);
          }
          boundary = true;
        } 

#if 0
// ------------------------------------------------------
// TEST mode
        resultval.get() = val[compute_index].get();
        resultval->x_ = val[compute_index]->x_;
        resultval->y_ = val[compute_index]->y_;
        resultval->z_ = val[compute_index]->z_;
        resultval->granularity = val[compute_index]->granularity;
        resultval->level_ = val[compute_index]->level_;
        resultval->cycle_ = val[compute_index]->cycle_ + 1;

        resultval->max_index_ = val[compute_index]->max_index_;
        resultval->granularity = val[compute_index]->granularity;
        resultval->index_ = val[compute_index]->index_;

        resultval->g_startx_ = val[compute_index]->g_startx_;
        resultval->g_endx_ = val[compute_index]->g_endx_;
        resultval->g_dx_ = val[compute_index]->g_dx_;
        resultval->timestep_ = val[compute_index]->timestep_ + 1.0/pow(2.0,resultval->level_);
        if (par->loglevel > 1 && fmod(resultval->timestep_, par->output) < 1.e-6) {
          stencil_data data (resultval.get());

        //  unlock_scoped_values_lock<lcos::mutex> ul(l);
          stubs::logging::logentry(log_, data, row, 0, par);
        }
        //if ( val[compute_index]->timestep_ >= par->nt0-1 ) {
        //  return 0;
        //}
        return 1;
// END TEST mode
// ------------------------------------------------------
#endif
//#if 0
        // these vectors are used for ghostwidth treatment
        std::vector< had_double_type > alt_vecx;
        std::vector< nodedata > alt_vecval;

        // put all data into a single array
        std::vector< had_double_type* > vecx;
        std::vector< nodedata* > vecval;
//#if 0
        nodedata3D valcube;
        std::vector<nodedata>::iterator niter;
        valcube.resize(3*par->granularity);
        for (int i=0;i< valcube.size();i++) {
          valcube[i].resize(3*par->granularity);
          for (int j=0;j< valcube[i].size(); j++) {
            valcube[i][j].resize(3*par->granularity);
          }
        }

        for (int i=0;i<val.size();i++) {
          int ii,jj,kk;
          if ( i == compute_index-3 ) {
            ii = 0; jj = 0; kk = -1;
          } else if ( i == compute_index-2 ) {
            ii = 0; jj = -1; kk = 0;
          } else if ( i == compute_index-1 ) {
            ii = -1; jj = 0; kk = 0;
          } else if ( i == compute_index ) {
            ii = 0; jj = 0; kk = 0;
          } else if ( i == compute_index+1 ) {
            ii = 1; jj = 0; kk = 0;
          } else if ( i == compute_index+2 ) {
            ii = 0; jj = 1; kk = 0;
          } else if ( i == compute_index+3 ) {
            ii = 0; jj = 0; kk = 1;
          }

          int count = 0;
          for (niter=val[i]->value_.begin();niter!=val[i]->value_.end();++niter) {
            int tmp_index = count/par->granularity;
            int c = tmp_index/par->granularity;
            int b = tmp_index%par->granularity;
            int a = count - par->granularity*(b+c*par->granularity);

            valcube[a+(ii+1)*par->granularity][b+(jj+1)*par->granularity][c+(kk+1)*par->granularity] = &(*niter); 
            count++;
          }
        }

        //std::cout << " TEST " << valcube[par->granularity][par->granularity][par->granularity]->phi[0][4] << std::endl;
//#endif

        // copy over critical info
        resultval->x_ = val[compute_index]->x_;
        resultval->y_ = val[compute_index]->y_;
        resultval->z_ = val[compute_index]->z_;
        resultval->value_.resize(val[compute_index]->value_.size());
        resultval->granularity = val[compute_index]->granularity;
        resultval->level_ = val[compute_index]->level_;
        resultval->cycle_ = val[compute_index]->cycle_ + 1;

        resultval->max_index_ = val[compute_index]->max_index_;
        resultval->granularity = val[compute_index]->granularity;
        resultval->index_ = val[compute_index]->index_;

        if (val[compute_index]->timestep_ < (int)numsteps_) {

            int level = val[compute_index]->level_;

            had_double_type dt = par->dt0/pow(2.0,level);
            had_double_type dx = par->dx0/pow(2.0,level); 

            // call rk update 
            int adj_index = 0;
            int gft = rkupdate(valcube,resultval.get_ptr(),
                                 boundary,bbox,adj_index,dt,dx,val[compute_index]->timestep_,
                                 level,*par.p);

            if (par->loglevel > 1 && fmod(resultval->timestep_, par->output) < 1.e-6) {
                stencil_data data (resultval.get());
                unlock_scoped_values_lock<lcos::mutex> ul(l);
                stubs::logging::logentry(log_, data, row, 0, par);
            }
        }
        else {
            // the last time step has been reached, just copy over the data
            resultval.get() = val[compute_index].get();
        }
        // set return value difference between actual and required number of
        // timesteps (>0: still to go, 0: last step, <0: overdone)
        if ( val[compute_index]->timestep_ >= par->nt0-2 ) {
          return 0;
        }
        return 1;

//#endif
    }

    hpx::actions::manage_object_action<stencil_data> const manage_stencil_data =
        hpx::actions::manage_object_action<stencil_data>();

    ///////////////////////////////////////////////////////////////////////////
    naming::id_type stencil::alloc_data(int item, int maxitems, int row,
                           Parameter const& par)
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

