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

    int stencil::floatcmp(had_double_type x1,had_double_type x2,had_double_type epsilon = 1.e-8) {
      // compare two floating point numbers
      if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
        // the numbers are close enough for coordinate comparison
        return 1;
      } else {
        return 0;
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

        // TEST mode
        //resultval.get() = val[0].get();
        //resultval->cycle_ += 1;
        //int t = resultval->cycle_;
        //int r = numsteps_ - t;
        //return r;

        // Here we give the coordinate value to the result (prior to sending it to the user)
        int compute_index;
        bool boundary = false;
        int bbox[2] = { 0, 0 };   // initialize bounding box

        if ( val.size()%2 == 0 ) {
          // boundary point
          boundary = true;
          compute_index = 0;
          bbox[0] = 1;
          bbox[1] = 0;
        } else {
          compute_index = (val.size()-1)/2;
        }

        // put all data into a single array
        std::vector< had_double_type > vecx;
        std::vector< nodedata > vecval;

        std::size_t count = 0;
        std::size_t adj_index = -1;
        for (int i = 0; i < val.size(); i++) {
          for (int j = 0; j < val[i]->granularity; j++) {
            vecval.push_back(val[i]->value_[j]);
            vecx.push_back(val[i]->x_[j]);
            if ( i == compute_index && adj_index == -1 ) {
              adj_index = count; 
            }
            count++;
          }
        }

        for (int j = 0; j < val[compute_index]->granularity; j++) {
          resultval->x_.push_back(val[compute_index]->x_[j]);
        }

        // DEBUG
        //char description[80];
        //double dasx = (double) resultval->x_[0];
        //double dast = (double) resultval->timestep_;
        //snprintf(description,sizeof(description),"x: %g t: %g level: %d",dasx,dast,val[0]->level_);
        //threads::thread_self& self = threads::get_self();
        //threads::thread_id_type id = self.get_thread_id();
        //threads::set_thread_description(id,description);

        if (val[0]->timestep_ < numsteps_) {

            // copy over critical info
            resultval->level_ = val[0]->level_;
            resultval->cycle_ = val[0]->cycle_ + 1;
            resultval->max_index_ = val[compute_index]->max_index_;
            resultval->granularity = val[compute_index]->granularity;
            resultval->index_ = val[compute_index]->index_;
            resultval->value_.resize(val[compute_index]->granularity);
            had_double_type dt = par->dt0/pow(2.0,(int) val[0]->level_);
            had_double_type dx = par->dx0/pow(2.0,(int) val[0]->level_); 

            // call rk update 
            int gft = rkupdate(&*vecval.begin(),resultval.get_ptr(),&*vecx.begin(),vecval.size(),
                                 boundary,bbox,adj_index,dt,dx,val[0]->timestep_,
                                 val[0]->iter_,val[0]->level_,*par.p);
            BOOST_ASSERT(gft);
  
            // increase the iteration counter
            if ( val[0]->iter_ == 2 ) {
                resultval->iter_ = 0;
            } 
            else {
                resultval->iter_ = val[0]->iter_ + 1;
            }

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
        int t = resultval->cycle_;
        int r = numsteps_ - t;
        //std::cout << " TEST r " << r << std::endl;
        return r;
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

