//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/lcos/future_wait.hpp>

#include <boost/foreach.hpp>

#include <math.h>

#include "../amr/amr_mesh.hpp"
#include "../amr/amr_mesh_tapered.hpp"
#include "../amr/amr_mesh_left.hpp"

#include "stencil.hpp"
#include "logging.hpp"
#include "stencil_data.hpp"
#include "stencil_functions.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    ///////////////////////////////////////////////////////////////////////////
    stencil::stencil()
      : numsteps_(0)
    {
    }

    int stencil::floatcmp(double x1,double x2) {
      // compare to floating point numbers
      double epsilon = 1.e-4;
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
        std::vector<naming::id_type> const& gids, int row, int column,
        Parameter const& par)
    {
        BOOST_ASSERT(gids.size() <= 5);

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

        // start asynchronous get operations

        // get all input memory_block_data instances
        access_memory_block<stencil_data> val1, val2, val3, val4, val5, resultval;
        if (gids.size() == 3) { 
            boost::tie(val1, val2, val3, resultval) = 
                get_memory_block_async<stencil_data>(gids[0], gids[1], gids[2], result);
        } 
        else if (gids.size() == 2) {
            boost::tie(val1, val2, resultval) = 
                get_memory_block_async<stencil_data>(gids[0], gids[1], result);
        } 
        else if (gids.size() == 5) {
            boost::tie(val1, val2, val3, val4, val5, resultval) = 
                get_memory_block_async<stencil_data>(gids[0], gids[1], gids[2], gids[3], gids[4], result);
        } 
        else {
          boost::tie(val1, resultval) = 
                get_memory_block_async<stencil_data>(gids[0], result);
          resultval.get() = val1.get();
          return -1;
        }

        // the predecessor
        double middle_timestep;
        if (gids.size() == 3) 
          middle_timestep = val2->timestep_;
        else if (gids.size() == 2 && column == 0) 
          middle_timestep = val1->timestep_;      // left boundary point
        else if (gids.size() == 2 && column != 0) {
          middle_timestep = val2->timestep_;      // right boundary point
        } else if ( gids.size() == 3 ) {
          middle_timestep = val2->timestep_;
        } else if ( gids.size() == 5 ) {
          middle_timestep = val3->timestep_;
        }

        if (val1->level_ == 0 && middle_timestep < numsteps_ || val1->level_ > 0) {

            if (gids.size() == 3) {
              // this is the actual calculation, call provided (external) function
              evaluate_timestep(val1.get_ptr(), val2.get_ptr(), val3.get_ptr(), 
                  resultval.get_ptr(), numsteps_,par,gids.size());

              // copy over the coordinate value to the result
              resultval->x_ = val2->x_;

              // copy over subsidiary data from stencil_data
              resultval->overwrite_alloc_ = 0;
              resultval->right_alloc_ = 0;
              resultval->left_alloc_ = 0;
            } else if (gids.size() == 2) {
              // bdry computation
              if ( column == 0 ) {
                evaluate_left_bdry_timestep(val1.get_ptr(), val2.get_ptr(),
                  resultval.get_ptr(), numsteps_,par);

                // copy over the coordinate value to the result
                resultval->x_ = val1->x_;

                // copy over subsidiary data from stencil_data
                resultval->overwrite_alloc_ = 0;
                resultval->right_alloc_ = 0;
                resultval->left_alloc_ = 0;
              } else {
                evaluate_right_bdry_timestep(val1.get_ptr(), val2.get_ptr(),
                  resultval.get_ptr(), numsteps_,par);

                // copy over the coordinate value to the result
                resultval->x_ = val2->x_;

                // copy over subsidiary data from stencil_data
                resultval->overwrite_alloc_ = 0;
                resultval->right_alloc_ = 0;
                resultval->left_alloc_ = 0;
              }
            } else if (gids.size() == 5) {
              // this is the actual calculation, call provided (external) function
              evaluate_timestep(val2.get_ptr(), val3.get_ptr(), val4.get_ptr(), 
                  resultval.get_ptr(), numsteps_,par,gids.size());

              // copy over the coordinate value to the result
              resultval->x_ = val3->x_;

              // copy over subsidiary data from stencil_data
              resultval->overwrite_alloc_ = 0;
              resultval->right_alloc_ = 0;
              resultval->left_alloc_ = 0;
            }

#if 0
            // DEBUGGING
            int st;
            if ( gids.size() == 5 ) {
              st = testpoint(val1,gids[0]); if (st) checkpoint(gids);
              st = testpoint(val2,gids[1]); if (st) checkpoint(gids);
              st = testpoint(val3,gids[2]); if (st) checkpoint(gids);
              st = testpoint(val4,gids[3]); if (st) checkpoint(gids);
              st = testpoint(val5,gids[4]); if (st) checkpoint(gids);
            } else if ( gids.size() == 3 ) {
              st = testpoint(val1,gids[0]); if (st) checkpoint(gids);
              st = testpoint(val2,gids[1]); if (st) checkpoint(gids);
              st = testpoint(val3,gids[2]); if (st) checkpoint(gids);
            } else if ( gids.size() == 2 ) {
              st = testpoint(val1,gids[0]); if (st) checkpoint(gids);
              st = testpoint(val2,gids[1]); if (st) checkpoint(gids);
            }
            st = testpoint(resultval,result);
#endif

            std::size_t allowedl = par.allowedl;
            if ( resultval->refine_ && gids.size() == 5 && resultval->level_ < allowedl 
                 && val1->timestep_ >= 1.e-6  ) {
              finer_mesh_tapered(result, gids, row, column, par);
            } else {
              resultval->overwrite_alloc_ = 0;
            } 

            // One special case: refining at time = 0
            if ( resultval->refine_ && gids.size() == 5 && 
                 val1->timestep_ < 1.e-6 && resultval->level_ < allowedl ) {
              finer_mesh_initial(result, gids, resultval->level_+1, resultval->x_, row, column, par);
            }

            if (log_ && fmod(resultval->timestep_,par.output) < 1.e-6)  
                stubs::logging::logentry(log_, resultval.get(), row,0, par);
        }
        else {
            // the last time step has been reached, just copy over the data
            if (gids.size() == 3) {
              resultval.get() = val2.get();
            } else if (gids.size() == 2) {
              // bdry computation
              if ( column == 0 ) {
                resultval.get() = val1.get();
              } else {
                resultval.get() = val2.get();
              }
            } else if (gids.size() == 5) {
              resultval.get() = val3.get();
            } else {
              BOOST_ASSERT(false);
            }
        }
 
        // set return value difference between actual and required number of
        // timesteps (>0: still to go, 0: last step, <0: overdone)
        if ( val1->level_ > 0 ) {
          if ( row == 1 || row == 2 ) return 0;
          else {
            return 1;
          }
        } else {
          int t = (int) (resultval->timestep_ + 0.5);
          int r = numsteps_ - t;
          return r;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Implement a finer mesh via interpolation of inter-mesh points
    // Compute the result value for the current time step
    int stencil::finer_mesh_tapered(naming::id_type const& result, 
        std::vector<naming::id_type> const& gids, int row,int column, Parameter const& par) 
    {

      int i;
      naming::id_type gval[9];
      access_memory_block<stencil_data> mval[9];
      access_memory_block<stencil_data> edge1,edge2;

      boost::tie(edge1,edge2) = 
          get_memory_block_async<stencil_data>(gids[0],gids[1]);

      if ( !edge1->refine_ || !edge2->refine_ || (row == 1 && column == 1) ) {
        boost::tie(gval[0], gval[2], gval[4], gval[6], gval[8]) = 
                        components::wait(components::stubs::memory_block::clone_async(gids[0]), 
                             components::stubs::memory_block::clone_async(gids[1]),
                             components::stubs::memory_block::clone_async(gids[2]),
                             components::stubs::memory_block::clone_async(gids[3]),
                             components::stubs::memory_block::clone_async(gids[4]));
        boost::tie(gval[1], gval[3], gval[5],gval[7]) = 
                    components::wait(components::stubs::memory_block::clone_async(gids[2]), 
                    components::stubs::memory_block::clone_async(gids[2]),
                    components::stubs::memory_block::clone_async(gids[2]),
                    components::stubs::memory_block::clone_async(gids[2]));
        boost::tie(mval[0], mval[2], mval[4], mval[6], mval[8]) = 
          get_memory_block_async<stencil_data>(gval[0], gval[2], gval[4], gval[6], gval[8]);

        // temporarily store the anchor values before overwriting them
        double t0,t2,t4,t6,t8;
        t0 = mval[0]->value_;
        t2 = mval[2]->value_;
        t4 = mval[4]->value_;
        t6 = mval[6]->value_;
        t8 = mval[8]->value_;

        // the edge of the AMR mesh has been reached.  
        // Use the left mesh class instead of standard tapered
        boost::tie(mval[1], mval[3], mval[5],mval[7]) = 
            get_memory_block_async<stencil_data>(gval[1], gval[3], gval[5],gval[7]);

        // increase the level by one
        for (i=0;i<9;i++) {
          ++mval[i]->level_;
          mval[i]->index_ = i;
        }

        // this updates the coordinate position
        mval[1]->x_ = 0.5*(mval[0]->x_+mval[2]->x_);
        mval[3]->x_ = 0.5*(mval[2]->x_+mval[4]->x_);
        mval[5]->x_ = 0.5*(mval[4]->x_+mval[6]->x_);
        mval[7]->x_ = 0.5*(mval[6]->x_+mval[8]->x_);

        // unset alloc on these gids
        for (i=1;i<9;i=i+2) {
          mval[i]->left_alloc_ = 0;
          mval[i]->right_alloc_ = 0;
          mval[i]->overwrite_alloc_ = 0;
        }

        // avoid interpolation if possible
        int s1,s3,s5,s7;
        s1 = 0; s3 = 0; s5 = 0; s7 = 0;

        s1 = findpoint(mval[0],mval[2],mval[1]);
        s3 = findpoint(mval[2],mval[4],mval[3]);
        s5 = findpoint(mval[4],mval[6],mval[5]);
        s7 = findpoint(mval[6],mval[8],mval[7]);

        if ( par.linearbounds == 1 ) {
          // linear interpolation
          if ( s1 == 0 ) mval[1]->value_ = 0.5*(t0 + t2);
          if ( s3 == 0 ) mval[3]->value_ = 0.5*(t2 + t4);
          if ( s5 == 0 ) mval[5]->value_ = 0.5*(t4 + t6);
          if ( s7 == 0 ) mval[7]->value_ = 0.5*(t6 + t8);

          // DEBUG
          if ( s1 == 0 ) stubs::logging::logentry(log_, mval[1].get(), row,2, par);
          if ( s3 == 0 ) stubs::logging::logentry(log_, mval[3].get(), row,2, par);
          if ( s5 == 0 ) stubs::logging::logentry(log_, mval[5].get(), row,2, par);
          if ( s7 == 0 ) stubs::logging::logentry(log_, mval[7].get(), row,2, par);
        } else {
          // other user defined options not implemented yet
          interpolation();
          BOOST_ASSERT(false);
        }

        // apply refinement criteria test to interpolated/found values
        mval[1]->refine_ = refinement(mval[1]->value_,mval[1]->level_);
        mval[3]->refine_ = refinement(mval[3]->value_,mval[1]->level_);
        mval[5]->refine_ = refinement(mval[5]->value_,mval[1]->level_);
        mval[7]->refine_ = refinement(mval[7]->value_,mval[1]->level_);

        // the initial data for the child mesh comes from the parent mesh
        naming::id_type here = applier::get_applier().get_runtime_support_gid();
        components::component_type logging_type =
                  components::get_component_type<components::amr::server::logging>();
        components::component_type function_type =
                  components::get_component_type<components::amr::stencil>();
        if ( !child_left_mesh[row].get_gid() ) {
            child_left_mesh[row].create(here, 1, true);
        }

        std::vector<naming::id_type> initial_data;
        for (i=0;i<9;i++) {
          initial_data.push_back(gval[i]);
        }

        bool do_logging = false;
        if ( par.loglevel > 0 ) {
          do_logging = true;
        }
        std::vector<naming::id_type> result_data(
            child_left_mesh[row].execute(initial_data, function_type,
              do_logging ? logging_type : components::component_invalid,par));
  
        access_memory_block<stencil_data> overwrite, resultval;
        boost::tie(overwrite, resultval) = 
            get_memory_block_async<stencil_data>(result_data[4], result);

        // overwrite the coarse point computation
        resultval->value_ = overwrite->value_;

        resultval->overwrite_alloc_ = 1;
        resultval->overwrite_ = result_data[4];

        // remember neighbor value
        overwrite->right_alloc_ = 1;
        overwrite->right_ = result_data[6];

        overwrite->left_alloc_ = 1;
        overwrite->left_ = result_data[2];

        // DEBUG -- log the right/left points computed
        access_memory_block<stencil_data> amb1 = 
                       hpx::components::stubs::memory_block::get(result_data[2]);
        stubs::logging::logentry(log_, amb1.get(), row,1, par);

        access_memory_block<stencil_data> amb2 = 
                       hpx::components::stubs::memory_block::get(result_data[6]);
        stubs::logging::logentry(log_, amb2.get(), row,1, par);

        components::stubs::memory_block::free(result_data[0]);
        components::stubs::memory_block::free(result_data[1]);
        components::stubs::memory_block::free(result_data[3]);
        components::stubs::memory_block::free(result_data[5]);
        components::stubs::memory_block::free(result_data[7]);
        components::stubs::memory_block::free(result_data[8]);
        // release result data
        //for (std::size_t i = 0; i < result_data.size(); ++i) 
        //    components::stubs::memory_block::free(result_data[i]);
      } else {
        boost::tie(gval[8], gval[1], gval[3], gval[5], gval[7]) = 
                        components::wait(components::stubs::memory_block::clone_async(gids[0]), 
                             components::stubs::memory_block::clone_async(gids[1]),
                             components::stubs::memory_block::clone_async(gids[2]),
                             components::stubs::memory_block::clone_async(gids[3]),
                             components::stubs::memory_block::clone_async(gids[4]));
        boost::tie(gval[0], gval[2], gval[4],gval[6]) = 
                    components::wait(components::stubs::memory_block::clone_async(gids[2]), 
                    components::stubs::memory_block::clone_async(gids[2]),
                    components::stubs::memory_block::clone_async(gids[2]),
                    components::stubs::memory_block::clone_async(gids[2]));
        boost::tie(mval[8], mval[1], mval[3], mval[5], mval[7]) = 
          get_memory_block_async<stencil_data>(gval[8], gval[1], gval[3], gval[5], gval[7]);

        boost::tie(mval[0], mval[2], mval[4],mval[6]) = 
            get_memory_block_async<stencil_data>(gval[0], gval[2], gval[4],gval[6]);

        // temporarily store the anchor values before overwriting them
        double tm1,t1,t3,t5,t7;
        tm1 = mval[8]->value_;
        t1 = mval[1]->value_;
        t3 = mval[3]->value_;
        t5 = mval[5]->value_;
        t7 = mval[7]->value_;

        // increase the level by one
        for (i=0;i<9;i++) {
          ++mval[i]->level_;
          mval[i]->index_ = i;
        }

        // this updates the coordinate position
        mval[0]->x_ = 0.5*(mval[8]->x_+mval[1]->x_);
        mval[2]->x_ = 0.5*(mval[1]->x_+mval[3]->x_);
        mval[4]->x_ = 0.5*(mval[3]->x_+mval[5]->x_);
        mval[6]->x_ = 0.5*(mval[5]->x_+mval[7]->x_);

        // reset alloc on these gids
        for (i=0;i<8;i=i+2) {
          mval[i]->left_alloc_ = 0;
          mval[i]->right_alloc_ = 0;
          mval[i]->overwrite_alloc_ = 0;
        }

        // avoid interpolation if possible
        int s0,s2,s4,s6;
        s0 = 0; s2 = 0; s4 = 0; s6 = 0;

        s0 = findpoint(mval[8],mval[1],mval[0]);
        s2 = findpoint(mval[1],mval[3],mval[2]);
        s4 = findpoint(mval[3],mval[5],mval[4]);
        s6 = findpoint(mval[5],mval[7],mval[6]);

        if ( par.linearbounds == 1 ) {
          if (s0 == 0) mval[0]->value_ = 0.5*(tm1 + t1);
          if (s2 == 0) mval[2]->value_ = 0.5*(t1 + t3);
          if (s4 == 0) mval[4]->value_ = 0.5*(t3 + t5);
          if (s6 == 0) mval[6]->value_ = 0.5*(t5 + t7);

          // DEBUG
          if ( s0 == 0 ) stubs::logging::logentry(log_, mval[0].get(), row,2, par);
          if ( s2 == 0 ) stubs::logging::logentry(log_, mval[2].get(), row,2, par);
          if ( s4 == 0 ) stubs::logging::logentry(log_, mval[4].get(), row,2, par);
          if ( s6 == 0 ) stubs::logging::logentry(log_, mval[6].get(), row,2, par);

        } else {
          // other user defined options not implemented yet
          interpolation();
          BOOST_ASSERT(false);
        }

        // apply refinement criteria test to interpolated/found values
        mval[0]->refine_ = refinement(mval[0]->value_,mval[0]->level_);
        mval[2]->refine_ = refinement(mval[2]->value_,mval[2]->level_);
        mval[4]->refine_ = refinement(mval[4]->value_,mval[4]->level_);
        mval[6]->refine_ = refinement(mval[6]->value_,mval[6]->level_);

        // the initial data for the child mesh comes from the parent mesh
        naming::id_type here = applier::get_applier().get_runtime_support_gid();
        components::component_type logging_type =
                  components::get_component_type<components::amr::server::logging>();
        components::component_type function_type =
                  components::get_component_type<components::amr::stencil>();
        if ( !child_mesh[row].get_gid() ) {
            child_mesh[row].create(here,1, true);
        }

        std::vector<naming::id_type> initial_data;
        for (i=0;i<8;i++) {
          initial_data.push_back(gval[i]);
        }

        // this cloned gid is not needed anymore
        components::stubs::memory_block::free(gval[8]);

        bool do_logging = false;
        if ( par.loglevel > 0 ) {
          do_logging = true;
        }
        std::vector<naming::id_type> result_data(
            child_mesh[row].execute(initial_data, function_type,
              do_logging ? logging_type : components::component_invalid,par));
  
        access_memory_block<stencil_data> overwrite,resultval;
        boost::tie(overwrite, resultval) = 
            get_memory_block_async<stencil_data>(result_data[3], result);

        // overwrite the coarse point computation
        resultval->value_ = overwrite->value_;

        // remember the overwrite point and the neighbor
        resultval->overwrite_alloc_ = 1;
        resultval->overwrite_ = result_data[3];

        overwrite->right_alloc_ = 1;
        overwrite->right_ = result_data[4];

        overwrite->left_alloc_ = 0;

        // DEBUG -- log the right points computed if no interp was involved
        access_memory_block<stencil_data> amb = 
                         hpx::components::stubs::memory_block::get(result_data[4]);
        stubs::logging::logentry(log_, amb.get(), row,1, par);

        components::stubs::memory_block::free(result_data[0]);
        components::stubs::memory_block::free(result_data[1]);
        components::stubs::memory_block::free(result_data[2]);
        components::stubs::memory_block::free(result_data[5]);
        components::stubs::memory_block::free(result_data[6]);
        components::stubs::memory_block::free(result_data[7]);
        // release result data
        //for (std::size_t i = 0; i < result_data.size(); ++i) 
        //    components::stubs::memory_block::free(result_data[i]);
      }

      return 0;
    }
    
    ///////////////////////////////////////////////////////////////////////////
    // Implement a finer mesh via interpolation of inter-mesh points
    // Compute the result value for the current time step
    int stencil::finer_mesh_initial(naming::id_type const& result, 
        std::vector<naming::id_type> const& gids, std::size_t level, double x, 
        int row, int column, Parameter const& par) 
    {

      // the initial data for the child mesh comes from the parent mesh
      naming::id_type here = applier::get_applier().get_runtime_support_gid();
      components::component_type logging_type =
                components::get_component_type<components::amr::server::logging>();
      components::component_type function_type =
                components::get_component_type<components::amr::stencil>();

      if (!child_left_mesh[row].get_gid())
          child_left_mesh[row].create(here, 1, true);

      bool do_logging = false;
      if ( par.loglevel > 0 ) {
        do_logging = true;
      }
      std::vector<naming::id_type> result_data(
          child_left_mesh[row].init_execute(function_type,
            do_logging ? logging_type : components::component_invalid,
            level, x, par));

      //  using mesh_left
      access_memory_block<stencil_data> overwrite, resultval;
      boost::tie(overwrite, resultval) = 
          get_memory_block_async<stencil_data>(result_data[4], result);

 
      // overwrite the coarse point computation
      resultval->value_ = overwrite->value_;
 
      resultval->overwrite_alloc_ = 1;
      resultval->overwrite_ = result_data[4];
   
      // remember neighbor value
      overwrite->right_alloc_ = 1;
      overwrite->right_ = result_data[6];

      overwrite->left_alloc_ = 1;
      overwrite->left_ = result_data[2];

      resultval->right_alloc_ = 0;
      resultval->left_alloc_ = 0;

      // DEBUG -- log the right/left points computed
      access_memory_block<stencil_data> amb1 = 
                         hpx::components::stubs::memory_block::get(result_data[2]);
      access_memory_block<stencil_data> amb2 = 
                         hpx::components::stubs::memory_block::get(result_data[6]);
      stubs::logging::logentry(log_, amb1.get(), row,1, par);
      stubs::logging::logentry(log_, amb2.get(), row,1, par);

      components::stubs::memory_block::free(result_data[0]);
      components::stubs::memory_block::free(result_data[1]);
      components::stubs::memory_block::free(result_data[3]);
      components::stubs::memory_block::free(result_data[5]);
      components::stubs::memory_block::free(result_data[7]);
      components::stubs::memory_block::free(result_data[8]);
 
      // release result data
      //for (std::size_t i = 0; i < result_data.size(); ++i) 
      //    components::stubs::memory_block::free(result_data[i]);

      return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    naming::id_type stencil::alloc_data(int item, int maxitems, int row,
        std::size_t level, double x, Parameter const& par)
    {
        naming::id_type result = components::stubs::memory_block::create(
            applier::get_applier().get_runtime_support_gid(), sizeof(stencil_data));

        if (-1 != item) {
            // provide initial data for the given data value 
            access_memory_block<stencil_data> val(
                components::stubs::memory_block::checkout(result));

            // call provided (external) function
            generate_initial_data(val.get_ptr(), item, maxitems, row, level, x, par);

            if (par.loglevel > 1)         // send initial value to logging instance
                stubs::logging::logentry(log_, val.get(), row,0, par);
        }
        return result;
    }

    int stencil::findpoint(access_memory_block<stencil_data> const& anchor_to_the_left,
                           access_memory_block<stencil_data> const& anchor_to_the_right, 
                           access_memory_block<stencil_data> & resultval) 
    {
      // the pinball machine
      int s = 0;
      access_memory_block<stencil_data> amb0;
      amb0 = anchor_to_the_left;
      if (s == 0 && amb0->overwrite_alloc_ == 1) {
        access_memory_block<stencil_data> amb1 = hpx::components::stubs::memory_block::get(amb0->overwrite_);

        // look around
        if ( amb1->right_alloc_ == 1 ) {
          access_memory_block<stencil_data> amb2 = hpx::components::stubs::memory_block::get(amb1->right_);
          if ( floatcmp(amb2->x_,resultval->x_) ) {
            resultval->value_ = amb2->value_;
            // transfer overwrite information as well
            if ( amb2->overwrite_alloc_ == 1 ) {
              resultval->overwrite_alloc_ = 1;
              resultval->overwrite_ = amb2->overwrite_;
            }

            s = 1;
            return s;
          } else {
            if ( amb2->x_ > resultval->x_ ) {
              s = findpoint(amb1,amb2,resultval);
            } else {
              s = findpoint(amb2,amb1,resultval);
            }
          }
        }

        if ( s == 0 && amb1->left_alloc_ == 1 ) {
          access_memory_block<stencil_data> amb2 = hpx::components::stubs::memory_block::get(amb1->left_);
          if ( floatcmp(amb2->x_,resultval->x_) ) {
            resultval->value_ = amb2->value_;
            // transfer overwrite information as well
            if ( amb2->overwrite_alloc_ == 1 ) {
              resultval->overwrite_alloc_ = 1;
              resultval->overwrite_ = amb2->overwrite_;
            }
            s = 1;
            return s;
          } else {
            if ( amb2->x_ > resultval->x_ ) {
              s = findpoint(amb1,amb2,resultval);
            } else {
              s = findpoint(amb2,amb1,resultval);
            }
          }
        }

      }


      amb0 = anchor_to_the_right;
      if (s == 0 && amb0->overwrite_alloc_ == 1) {
        access_memory_block<stencil_data> amb1 = hpx::components::stubs::memory_block::get(amb0->overwrite_);

        // look around
        if ( amb1->right_alloc_ == 1 ) {
          access_memory_block<stencil_data> amb2 = hpx::components::stubs::memory_block::get(amb1->right_);
          if ( floatcmp(amb2->x_,resultval->x_) ) {
            resultval->value_ = amb2->value_;
            // transfer overwrite information as well
            if ( amb2->overwrite_alloc_ == 1 ) {
              resultval->overwrite_alloc_ = 1;
              resultval->overwrite_ = amb2->overwrite_;
            }
            s = 1;
            return s;
          } else {
            if ( amb2->x_ > resultval->x_ ) {
              s = findpoint(amb1,amb2,resultval);
            } else {
              s = findpoint(amb2,amb1,resultval);
            }
          }
        }

        if (s == 0 && amb1->left_alloc_ == 1 ) {
          access_memory_block<stencil_data> amb2 = hpx::components::stubs::memory_block::get(amb1->left_);
          if ( floatcmp(amb2->x_,resultval->x_) ) {
            resultval->value_ = amb2->value_;
            // transfer overwrite information as well
            if ( amb2->overwrite_alloc_ == 1 ) {
              resultval->overwrite_alloc_ = 1;
              resultval->overwrite_ = amb2->overwrite_;
            }
            s = 1;
            return s;
          } else {
            if ( amb2->x_ > resultval->x_ ) {
              s = findpoint(amb1,amb2,resultval);
            } else {
              s = findpoint(amb2,amb1,resultval);
            }
          }
        }
      }

      return s;
    }

    void stencil::init(std::size_t numsteps, naming::id_type const& logging)
    {
        numsteps_ = numsteps;
        log_ = logging;
    }

    // This routine is for debugging
    int stencil::testpoint(access_memory_block<stencil_data> const& val,
                            naming::id_type const& gid)
    {
       if ( floatcmp(val->x_,3.3333333333333333) == 1 ) {
           printf(" TEST overwrite %d timestep: %g value %g index %d id %d level %d x %g right_alloc %d left_alloc %d refine %d\n",
               val->overwrite_alloc_,val->timestep_,
               val->value_,val->index_,gid.id_lsb_,val->level_,
               val->x_,val->right_alloc_,val->left_alloc_,val->refine_);
           //if ( gid.id_lsb_ == 549233 ) {
           //  return 1;
           //}
       }
       return 0;
    }

    // This routine is for debugging
    void stencil::checkpoint(std::vector<naming::id_type> const& gids)
    {
      int i;
      for (i=0;i<gids.size();i++) {
        access_memory_block<stencil_data> amb = hpx::components::stubs::memory_block::get(gids[i]);
        printf(" gid: %d location: %g overwrite: %d\n",gids[i].id_lsb_,amb->x_,amb->overwrite_alloc_);
        if ( amb->overwrite_alloc_ == 1 ) {
          access_memory_block<stencil_data> amb2 = hpx::components::stubs::memory_block::get(amb->overwrite_);
          printf(" overwrite      location: %g overwrite: %d : %d %d : level %d\n",amb2->x_,amb2->overwrite_alloc_,amb2->left_alloc_,amb2->right_alloc_,amb2->level_);
          if ( amb2->overwrite_alloc_ == 1 ) {
            access_memory_block<stencil_data> amb3 = hpx::components::stubs::memory_block::get(amb2->overwrite_);
          printf("  overwrite overwrite    location: %g overwrite: %d : %d %d : level %d\n",amb3->x_,amb3->overwrite_alloc_,amb3->left_alloc_,amb3->right_alloc_,amb3->level_);
          }
          if ( amb2->right_alloc_ == 1 ) {
            access_memory_block<stencil_data> amb3 = hpx::components::stubs::memory_block::get(amb2->right_);
          printf(" overwrite right       location: %g overwrite: %d : %d %d : level %d\n",amb3->x_,amb3->overwrite_alloc_,amb3->left_alloc_,amb3->right_alloc_,amb3->level_);
          }
          if ( amb2->left_alloc_ == 1 ) {
            access_memory_block<stencil_data> amb3 = hpx::components::stubs::memory_block::get(amb2->left_);
          printf(" overwrite left       location: %g overwrite: %d : %d %d : level %d\n",amb3->x_,amb3->overwrite_alloc_,amb3->left_alloc_,amb3->right_alloc_,amb3->level_);
          }
        }
      }

    }

    // This routine is for debugging -- pass in any gid
    // and it traverses the entire grid available at that moment
    void stencil::traverse_grid(naming::id_type const& start,int firstcall)
    {
      int i;
      int found;
#if 0
      if (firstcall == 1) lsb_count = 0;

      access_memory_block<stencil_data> amb = hpx::components::stubs::memory_block::get(start);
      printf("stencil::traverse_grid x: %g lsb: %d\n",amb->x_,start.id_lsb_);
      if ( amb->overwrite_alloc_ == 1 ) {
        // check if the lsb has already been recorded
        found = 0;
        for (i=0;i<lsb_count;i++) {
          if (amb->overwrite_.id_lsb_ == unique_lsb[i]) {
            found = 1;
            break;
          }
        }
        if ( found == 0 ) {
          unique_lsb[lsb_count] = amb->overwrite_.id_lsb_;
          lsb_count++;
          printf("stencil::traverse_grid overwrite\n");
          traverse_grid(amb->overwrite_,0);
        }
      }

      if ( amb->right_alloc_ == 1 ) {
        // check if the lsb has already been recorded
        found = 0;
        for (i=0;i<lsb_count;i++) {
          if (amb->right_.id_lsb_ == unique_lsb[i]) {
            found = 1;
            break;
          }
        }
        if ( found == 0 ) {
          unique_lsb[lsb_count] = amb->right_.id_lsb_;
          lsb_count++;
          printf("stencil::traverse_grid right\n");
          traverse_grid(amb->right_,0);
        }
      }

      if ( amb->left_alloc_ == 1 ) {
        // check if the lsb has already been recorded
        found = 0;
        for (i=0;i<lsb_count;i++) {
          if (amb->left_.id_lsb_ == unique_lsb[i]) {
            found = 1;
            break;
          }
        }
        if ( found == 0 ) {
          unique_lsb[lsb_count] = amb->left_.id_lsb_;
          lsb_count++;
          printf("stencil::traverse_grid left\n");
          traverse_grid(amb->left_,0);
        }
      }
#endif
    }


}}}

