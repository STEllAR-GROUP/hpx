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
#include "../amr/rk_left.hpp"

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

    int stencil::floatcmp(had_double_type x1,had_double_type x2) {
      // compare to floating point numbers
      had_double_type epsilon = 1.e-8;
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
        access_memory_block<stencil_data> resultval;
        std::vector<access_memory_block<stencil_data> > val;

        int i;
        std::vector< stencil_data * > vecval;
        resultval = get_memory_block_async(val,gids,result);

        // the gids may originate from different timesteps (due to rk tapering)
        if ( val[0]->iter_ != val[val.size()-1]->iter_ ) {
          for (i=0;i<val.size();i++) {
            if ( val[0]->iter_ == val[i]->iter_ ) vecval.push_back(val[i].get_ptr());
          }
        } else {
          for (i=0;i<val.size();i++) vecval.push_back(val[i].get_ptr());
        }

        // Here we give the coordinate value to the result (prior to sending it to the user)
        int compute_index;
        bool boundary = false;
        int bbox[2];

        if ( val[0]->level_ == 0 ) {
          if ( column == 4 ) {
            // indicate a physical boundary
            boundary = true;
            compute_index = 1;
            bbox[0] = 1;
            bbox[1] = 0;
          } else if ( column == par->nx0 - 1) {
            // indicate a physical boundary
            boundary = true;
            compute_index = vecval.size()-1;
            bbox[0] = 0;
            bbox[1] = 1;
          } else if ( column == 0 ) {
            compute_index = 0;
          }
        } 

        if (vecval.size()%2 != 0 ) {
          if ( gids.size() == 1 ) { 
            resultval.get() = val[0].get();
            return -1;
          }
          BOOST_ASSERT( (vecval.size())%2 == 1 );
          if ( !boundary ) compute_index = (vecval.size()-1)/2;
        } else {
          // This should only happen at a physical boundary or the left edge of the mesh
          BOOST_ASSERT( (column == 0 || boundary) );
        }

        // update x position
        resultval->x_ = val[compute_index]->x_;

        //boundary for r = 0
        if ( floatcmp(resultval->x_,0.0) ) {
          // indicate a physical boundary
          boundary = true;
          bbox[0] = 1;
          bbox[1] = 0;
        }

        // initialize result 
        resultval->overwrite_alloc_ = false;
        resultval->right_alloc_ = false;
        resultval->left_alloc_ = false;

        // the first two input values should have the same timestep
        BOOST_ASSERT(val[0]->timestep_== val[1]->timestep_);

        if (val[0]->level_ == 0 && val[0]->timestep_ < numsteps_ || val[0]->level_ > 0) {

            // call rk update 
            int gft = rkupdate(&*vecval.begin(),resultval.get_ptr(),vecval.size(),boundary,bbox,compute_index,par.p);
            BOOST_ASSERT(gft);
            // refine only after rk subcycles are finished (we don't refine in the midst of rk subcycles)
            if ( resultval->iter_ == 0 ) resultval->refine_ = refinement(&*vecval.begin(),vecval.size(),&resultval->value_,resultval->level_,resultval->x_,compute_index,boundary,bbox,par.p);
            else resultval->refine_ = false;

            std::size_t allowedl = par->allowedl;

            // eliminate unrefinable cases
            if ( gids.size() != 5 && par->stencilsize == 3 && par->integrator == 0 ) resultval->refine_ = false;
            if ( par->stencilsize == 3 && par->integrator == 1 ) {
              if ( gids.size() == vecval.size() && gids.size() != 9 ) resultval->refine_ = false; 
              if ( gids.size() != vecval.size() && gids.size() - vecval.size() != 9 ) resultval->refine_ = false; 
            }

            if ( resultval->refine_ && resultval->level_ < allowedl 
                 && val[0]->timestep_ >= 1.e-6  ) {
              finer_mesh_tapered(result, gids,vecval.size(), row, column, par);
            } else {
              resultval->overwrite_alloc_ = 0;
            } 

            // One special case: refining at time = 0
            if ( resultval->refine_ && 
                 val[0]->timestep_ < 1.e-6 && resultval->level_ < allowedl ) {
              finer_mesh_initial(result, gids, resultval->level_+1, resultval->x_, row, column, par);
            }

            if (log_ && fmod(resultval->timestep_,par->output) < 1.e-6) 
                stubs::logging::logentry(log_, resultval.get(), row,0, par);
        }
        else {
            // the last time step has been reached, just copy over the data
            resultval.get() = val[compute_index].get();
        }
 
        // set return value difference between actual and required number of
        // timesteps (>0: still to go, 0: last step, <0: overdone)
        if ( val[0]->level_ > 0 ) {
          if ( row > 0 ) return 0;
          else {
            return 1;
          }
        } else {
          int t = resultval->cycle_;
          int r = numsteps_ - t;
          return r;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Implement a finer mesh via interpolation of inter-mesh points
    // Compute the result value for the current time step
    int stencil::finer_mesh_tapered(naming::id_type const& result, 
        std::vector<naming::id_type> const& gids,int vecvalsize, int row,int column, Parameter const& par) 
    {
      naming::id_type gval[9];
      access_memory_block<stencil_data> mval[9];

      bool left = left_tapered_mesh(gids,row,column,par);

      if ( left ) {
        // -------------------- Left (unbiased) Tapered Mesh --------------------------
        std::vector<naming::id_type> initial_data;
        left_tapered_prep_initial_data(initial_data,gids,vecvalsize,row,column,par);

        // mesh object setup
        components::component_type logging_type =
                  components::get_component_type<components::amr::server::logging>();
        components::component_type function_type =
                  components::get_component_type<components::amr::stencil>();
        // create the mesh only if you need to, otherwise reuse (reduce overhead)
        if ( par->integrator == 0 ) {
          // Euler not supported anymore
          BOOST_ASSERT(false);
        } else if ( par->integrator == 1 ) {
          if ( !rk_left_mesh[row].get_gid() ) {
              rk_left_mesh[row].create(applier::get_applier().get_runtime_support_gid());
          }
        } else {
          BOOST_ASSERT(false);
        }

        bool do_logging = false;
        if ( par->loglevel > 0 ) {
          do_logging = true;
        }

        std::vector<naming::id_type> result_data;
        if ( par->integrator == 0 ) {
          // Euler not supported anymore
        } else if ( par->integrator == 1 ) {
          result_data =  rk_left_mesh[row].execute(initial_data, function_type,
                do_logging ? logging_type : components::component_invalid,par);
        } else {
          BOOST_ASSERT(false);
        }
  
        // -------------------------------------------------------------------
        // You get 3 values out: left, center, and right -- that's it.  overwrite the coarse grid point and
        // tell the neighbors to remember the left and right values.
        access_memory_block<stencil_data> overwrite, resultval;
        int mid; 
        if ( (result_data.size())%2 == 1 ) {
          mid = (result_data.size()-1)/2;
        } else {
          BOOST_ASSERT(false);
        }
        boost::tie(overwrite, resultval) = 
            get_memory_block_async<stencil_data>(result_data[mid], result);

        // overwrite the coarse point computation
        resultval->value_ = overwrite->value_;

        resultval->overwrite_alloc_ = true;
        resultval->overwrite_ = result_data[mid];

        // remember neighbor value
        overwrite->right_alloc_ = true;
        overwrite->right_ = result_data[result_data.size()-1];

        overwrite->left_alloc_ = true;
        overwrite->left_ = result_data[0];

        // DEBUG -- log the right/left points computed
        access_memory_block<stencil_data> amb1 = 
                       hpx::components::stubs::memory_block::get(result_data[0]);
        if (log_)
            stubs::logging::logentry(log_, amb1.get(), row,1, par);

        access_memory_block<stencil_data> amb2 = 
                       hpx::components::stubs::memory_block::get(result_data[result_data.size()-1]);
        if (log_)
            stubs::logging::logentry(log_, amb2.get(), row,1, par);

        // release result data
        for (std::size_t i = 1; i < result_data.size()-1; ++i) { 
          if ( i != mid ) components::stubs::memory_block::free(result_data[i]);
        }
      } else {
        // -------------------- Right (biased) Tapered Mesh --------------------------
        // not implemented
        BOOST_ASSERT(false);
      }

      return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Decide whether to use a left or right biased tapered mesh
    bool stencil::left_tapered_mesh(std::vector<naming::id_type> const& gids, int row,int column, Parameter const& par) 
    {
      if ( par->integrator == 0 ) {
        BOOST_ASSERT(false);
        //access_memory_block<stencil_data> edge1,edge2;
        //boost::tie(edge1,edge2) = get_memory_block_async<stencil_data>(gids[0],gids[1]);
        //if ( !edge1->refine_ || !edge2->refine_ || (row == 1 && column == 1) ) 
        //    return true;
        //return false;
      } else if (par->integrator == 1) {
        return true;
      }
      BOOST_ASSERT(false);
      return false;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Prep initial data for left (unbiased) tapered mesh
    int stencil::left_tapered_prep_initial_data(std::vector<naming::id_type> & initial_data, 
        std::vector<naming::id_type> const& gids,int vecvalsize, int row,int column, Parameter const& par) 
    {
      int i;
      if ( par->integrator == 0 ) {
        // not implemented
        BOOST_ASSERT(false);
      } else if (par->integrator == 1) {
        // rk3 {{{
        BOOST_ASSERT(gids.size()-vecvalsize == 9 || gids.size() == 9);
        naming::id_type gval[17];
        access_memory_block<stencil_data> mval[17];

        // inputs may include different timestamps; separate these out
        int std_index;
        if ( gids.size() != vecvalsize ) {
          std_index = vecvalsize;
        } else {
          std_index = 0;
        }
        boost::tie(gval[0], gval[2], gval[4], gval[6], gval[8]) = 
                        components::wait(components::stubs::memory_block::clone_async(gids[std_index]), 
                             components::stubs::memory_block::clone_async(gids[std_index+1]),
                             components::stubs::memory_block::clone_async(gids[std_index+2]),
                             components::stubs::memory_block::clone_async(gids[std_index+3]),
                             components::stubs::memory_block::clone_async(gids[std_index+4]));
        boost::tie(gval[10], gval[12], gval[14], gval[16]) = 
                        components::wait(components::stubs::memory_block::clone_async(gids[std_index+5]), 
                             components::stubs::memory_block::clone_async(gids[std_index+6]),
                             components::stubs::memory_block::clone_async(gids[std_index+7]),
                             components::stubs::memory_block::clone_async(gids[std_index+8]));
        boost::tie(gval[1], gval[3], gval[5],gval[7]) = 
                    components::wait(components::stubs::memory_block::clone_async(gids[4]), 
                    components::stubs::memory_block::clone_async(gids[4]),
                    components::stubs::memory_block::clone_async(gids[4]),
                    components::stubs::memory_block::clone_async(gids[4]));
        boost::tie(gval[9], gval[11], gval[13],gval[15]) = 
                    components::wait(components::stubs::memory_block::clone_async(gids[4]), 
                    components::stubs::memory_block::clone_async(gids[4]),
                    components::stubs::memory_block::clone_async(gids[4]),
                    components::stubs::memory_block::clone_async(gids[4]));
        boost::tie(mval[0], mval[2], mval[4], mval[6], mval[8]) = 
          get_memory_block_async<stencil_data>(gval[0], gval[2], gval[4], gval[6], gval[8]);
        boost::tie(mval[10], mval[12], mval[14], mval[16]) = 
          get_memory_block_async<stencil_data>(gval[10], gval[12], gval[14], gval[16]);

        // the edge of the AMR mesh has been reached.  
        // Use the left mesh class instead of standard tapered
        boost::tie(mval[1], mval[3], mval[5],mval[7]) = 
            get_memory_block_async<stencil_data>(gval[1], gval[3], gval[5],gval[7]);
        boost::tie(mval[9], mval[11], mval[13],mval[15]) = 
            get_memory_block_async<stencil_data>(gval[9], gval[11], gval[13],gval[15]);

        for (i=0;i<17;i++) {
          // increase the level by one
          ++mval[i]->level_;
          mval[i]->index_ = i;

          mval[i]->iter_ = 0;
        }

        // this updates the coordinate position
        for (i=1;i<17;i=i+2) {
          mval[i]->x_ = 0.5*(mval[i-1]->x_+mval[i+1]->x_);
        }

        // unset alloc on these gids
        for (i=1;i<17;i=i+2) {
          mval[i]->left_alloc_ = false;
          mval[i]->right_alloc_ = false;
          mval[i]->overwrite_alloc_ = false;
        }

        // avoid interpolation if possible
        int s;
        bool boundary = false;
        int bbox[2];
        s = 0;
        for (i=1;i<17;i=i+2) {
          s = findpoint(mval[i-1],mval[i+1],mval[i]);
          if ( s == 0 ) { 
            std::vector< had_double_type > x_val;
            std::vector< nodedata > n_val;
            x_val.push_back(mval[0]->x_); n_val.push_back(mval[0]->value_);
            x_val.push_back(mval[2]->x_); n_val.push_back(mval[2]->value_);
            x_val.push_back(mval[4]->x_); n_val.push_back(mval[4]->value_);
            x_val.push_back(mval[6]->x_); n_val.push_back(mval[6]->value_);
            x_val.push_back(mval[8]->x_); n_val.push_back(mval[8]->value_);
            x_val.push_back(mval[10]->x_); n_val.push_back(mval[10]->value_);
            x_val.push_back(mval[12]->x_); n_val.push_back(mval[12]->value_);
            x_val.push_back(mval[14]->x_); n_val.push_back(mval[14]->value_);
            x_val.push_back(mval[16]->x_); n_val.push_back(mval[16]->value_);
            // pass in everything -- let the user decide how to interpolate using all the available anchors
            int gft = interpolation(mval[i]->x_,&(mval[i]->value_),
                          &*x_val.begin(),x_val.size(),
                          &*n_val.begin(),n_val.size());
            BOOST_ASSERT(gft);

            std::vector< stencil_data * > vecval;
            vecval.push_back(mval[i-1].get_ptr());
            vecval.push_back(mval[i].get_ptr());
            vecval.push_back(mval[i+1].get_ptr());
            mval[i]->refine_ = refinement(&*vecval.begin(),vecval.size(),&(mval[i]->value_),mval[i]->level_,mval[i]->x_,1,boundary,bbox,par.p);

            // DEBUG
            if (log_)
                stubs::logging::logentry(log_, mval[i].get(), row,2, par);
          }
        }

        for (i=0;i<17;i++) {
          initial_data.push_back(gval[i]);
        }
        // }}}
      }

      return 0;
    }
    
    ///////////////////////////////////////////////////////////////////////////
    // Prep initial data for right biased tapered mesh
    int stencil::right_tapered_prep_initial_data(std::vector<naming::id_type> & initial_data, 
        std::vector<naming::id_type> const& gids,int vecvalsize, int row,int column, Parameter const& par) 
    {
      int i;
      if ( par->integrator == 0 ) {
        // not implemented yet
        BOOST_ASSERT(false);
      } else if (par->integrator == 1) {
        // not implemented yet
        BOOST_ASSERT(false);
      }

      return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Implement a finer mesh via interpolation of inter-mesh points
    // Compute the result value for the current time step
    int stencil::finer_mesh_initial(naming::id_type const& result, 
        std::vector<naming::id_type> const& gids, std::size_t level, had_double_type x, 
        int row, int column, Parameter const& par) 
    {
      // the initial data for the child mesh comes from the parent mesh
      naming::id_type here = applier::get_applier().get_runtime_support_gid();
      components::component_type logging_type =
                components::get_component_type<components::amr::server::logging>();
      components::component_type function_type =
                components::get_component_type<components::amr::stencil>();

      if ( par->integrator == 0 ) {
        BOOST_ASSERT(false);
      } else if ( par->integrator == 1 ) {
        if ( !rk_left_mesh[row].get_gid() ) {
            rk_left_mesh[row].create(here);
        }
      } else {
        BOOST_ASSERT(false);
      }

      bool do_logging = false;
      if ( par->loglevel > 0 ) {
        do_logging = true;
      }

      std::vector<naming::id_type> result_data;
      if ( par->integrator == 0 ) {
        BOOST_ASSERT(false);
      } else if ( par->integrator == 1 ) {
        result_data = rk_left_mesh[row].init_execute(function_type,
              do_logging ? logging_type : components::component_invalid,
              level, x, par);
      } else {
        BOOST_ASSERT(false);
      }


      //  using mesh_left
      access_memory_block<stencil_data> overwrite, resultval;
      int mid; 
      if ( (result_data.size())%2 == 1 ) {
        mid = (result_data.size()-1)/2;
      } else {
        BOOST_ASSERT(false);
      }

      boost::tie(overwrite, resultval) = 
          get_memory_block_async<stencil_data>(result_data[mid], result);

 
      // overwrite the coarse point computation
      resultval->value_ = overwrite->value_;
 
      resultval->overwrite_alloc_ = true;
      resultval->overwrite_ = result_data[mid];
   
      // remember neighbor value
      overwrite->right_alloc_ = true;
      overwrite->right_ = result_data[0];

      overwrite->left_alloc_ = true;
      overwrite->left_ = result_data[result_data.size()-1];

      resultval->right_alloc_ = false;
      resultval->left_alloc_ = false;

      // DEBUG -- log the right/left points computed
      access_memory_block<stencil_data> amb1 = 
                         hpx::components::stubs::memory_block::get(result_data[0]);
      access_memory_block<stencil_data> amb2 = 
                         hpx::components::stubs::memory_block::get(result_data[result_data.size()-1]);
      if (log_) {
          stubs::logging::logentry(log_, amb1.get(), row,1, par);
          stubs::logging::logentry(log_, amb2.get(), row,1, par);
      }

      for (std::size_t i = 1; i < result_data.size()-1; ++i) {
        // free all but the overwrite and end value
        if ( i != mid ) components::stubs::memory_block::free(result_data[i]);
      }

      return 0;
    }

    hpx::actions::manage_object_action<stencil_data> const manage_stencil_data =
        hpx::actions::manage_object_action<stencil_data>();

    ///////////////////////////////////////////////////////////////////////////
    naming::id_type stencil::alloc_data(int item, int maxitems, int row,
        std::size_t level, had_double_type x, Parameter const& par)
    {
        naming::id_type here = applier::get_applier().get_runtime_support_gid();
        naming::id_type result = components::stubs::memory_block::create(
            here, sizeof(stencil_data), manage_stencil_data);

        if (-1 != item) {
            // provide initial data for the given data value 
            access_memory_block<stencil_data> val(
                components::stubs::memory_block::checkout(result));

            // call provided (external) function
            generate_initial_data(val.get_ptr(), item, maxitems, row, level, x, par.p);

            if (log_ && par->loglevel > 1)         // send initial value to logging instance
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
            resultval->refine_ = amb2->refine_;
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

        if ( s == 0 && amb1->left_alloc_) {
          access_memory_block<stencil_data> amb2 = hpx::components::stubs::memory_block::get(amb1->left_);
          if ( floatcmp(amb2->x_,resultval->x_) ) {
            resultval->value_ = amb2->value_;
            resultval->refine_ = amb2->refine_;
            // transfer overwrite information as well
            if ( amb2->overwrite_alloc_) {
              resultval->overwrite_alloc_ = true;
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
            resultval->refine_ = amb2->refine_;
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

        if (s == 0 && amb1->left_alloc_) {
          access_memory_block<stencil_data> amb2 = hpx::components::stubs::memory_block::get(amb1->left_);
          if ( floatcmp(amb2->x_,resultval->x_) ) {
            resultval->value_ = amb2->value_;
            resultval->refine_ = amb2->refine_;
            // transfer overwrite information as well
            if ( amb2->overwrite_alloc_) {
              resultval->overwrite_alloc_ = true;
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
          // printf(" TEST overwrite %d timestep: %g index %d id %d level %d x %g right_alloc %d left_alloc %d refine %d\n",
          //     val->overwrite_alloc_,val->timestep_,
          //     val->index_, gid.get_gid().get_lsb(),val->level_,
          //     val->x_,val->right_alloc_,val->left_alloc_,val->refine_);
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
     //   printf(" gid: %d location: %g overwrite: %d\n", gids[i].get_gid().get_lsb(), amb->x_, amb->overwrite_alloc_);
        if ( amb->overwrite_alloc_ == 1 ) {
          access_memory_block<stencil_data> amb2 = hpx::components::stubs::memory_block::get(amb->overwrite_);
     //     printf(" overwrite      location: %g overwrite: %d : %d %d : level %d\n",amb2->x_,amb2->overwrite_alloc_,amb2->left_alloc_,amb2->right_alloc_,amb2->level_);
          if ( amb2->overwrite_alloc_ == 1 ) {
            access_memory_block<stencil_data> amb3 = hpx::components::stubs::memory_block::get(amb2->overwrite_);
     //     printf("  overwrite overwrite    location: %g overwrite: %d : %d %d : level %d\n",amb3->x_,amb3->overwrite_alloc_,amb3->left_alloc_,amb3->right_alloc_,amb3->level_);
          }
          if ( amb2->right_alloc_ == 1 ) {
            access_memory_block<stencil_data> amb3 = hpx::components::stubs::memory_block::get(amb2->right_);
     //     printf(" overwrite right       location: %g overwrite: %d : %d %d : level %d\n",amb3->x_,amb3->overwrite_alloc_,amb3->left_alloc_,amb3->right_alloc_,amb3->level_);
          }
          if ( amb2->left_alloc_ == 1 ) {
            access_memory_block<stencil_data> amb3 = hpx::components::stubs::memory_block::get(amb2->left_);
     //     printf(" overwrite left       location: %g overwrite: %d : %d %d : level %d\n",amb3->x_,amb3->overwrite_alloc_,amb3->left_alloc_,amb3->right_alloc_,amb3->level_);
          }
        }
      }

    }

    // This routine is for debugging -- pass in any gid
    // and it traverses the entire grid available at that moment
    void stencil::traverse_grid(naming::id_type const& start,int firstcall)
    {
#if 0
      int i;
      int found;
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

      if ( amb->left_alloc_ ) {
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

