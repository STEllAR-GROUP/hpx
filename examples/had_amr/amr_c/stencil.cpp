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
#include "../amr/unigrid_mesh.hpp"

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
        std::vector<access_memory_block<stencil_data> > val,tval;

        int i,j;
//         had_double_type timestep;
        std::vector< nodedata > vecval;
        std::vector< had_double_type > vecx;
        resultval = get_memory_block_async(val,gids,result);

        // Here we give the coordinate value to the result (prior to sending it to the user)
        int compute_index;
        bool boundary = false;
        int bbox[2];
        int numvals = par->nx0/par->granularity;
        int asize = (int) pow(2.,(int)val[0]->level_);

        // initialize bounding box
        bbox[0] = 0;
        bbox[1] = 0;

        if ( val[0]->iter_ != val[val.size()-1]->iter_ ) {
          for (i=0;i<val.size();i++) {
            if ( val[0]->iter_ == val[i]->iter_ ) tval.push_back(val[i]);
          }
        } else {
          for (i=0;i<val.size();i++) tval.push_back(val[i]);
        }
        //std::cout << " TEST " << val[0]->iter_ << " " << val[val.size()-1]->iter_ << " " << val.size() << " " << tval.size() << std::endl;

        if ( tval[0]->level_ == 0 ) {
          if ( column == 0 ) {
            // indicate a physical boundary
            boundary = true;
            compute_index = 0;
            bbox[0] = 1;
           }
          if ( column == numvals - 1) {
            // indicate a physical boundary
            boundary = true;
            compute_index = tval.size()-1;
            bbox[1] = 1;
          } 
          if ( !boundary ) {
            if ( (tval.size()-1)%2 == 0 ) {
              compute_index = (tval.size()-1)/2;
              if ( column == 1 && par->granularity == 1 ) {
                boundary = true;
                bbox[0] = 2;
                bbox[1] = 0;
              }
            } else {
              BOOST_ASSERT(false);
            }
          }
        } else {
          if ( column == 0  ) {
            compute_index = 0;
          } else if ( column == val[0]->max_index_ - 1) {
            compute_index = tval.size()-1;
          } else if ( (tval.size()-1)%2 == 0 ) {
            compute_index = (tval.size()-1)/2;
          } else {
            BOOST_ASSERT(false);
          } 

          // Decide if the compute_index point is a boundary
          if ( floatcmp(0.0,tval[compute_index]->x_[0]) ) {
            // indicate a physical boundary
            boundary = true;
            bbox[0] = 1;
          } else if ( column == 1 && par->granularity == 1 && floatcmp(0.0,tval[compute_index-1]->x_[0]) ) {
            boundary = true;
            bbox[0] = 2;
            bbox[1] = 0;
          } else if ( column == val[0]->max_index_-1 && floatcmp(par->maxx0,tval[compute_index]->x_[par->granularity-1]) ) {
            boundary = true;
            bbox[1] = 1;
          }
        } 

        // put all data into a single array
        int count = 0;
        int adj_index = -1;
        for (i=0;i<tval.size();i++) {
          for (j=0;j<asize*par->granularity;j++) {
            vecval.push_back(tval[i]->value_[j]);
            vecx.push_back(tval[i]->x_[j]);
            if ( i == compute_index && adj_index == -1 ) {
              adj_index = count; 
            }
            count++;
          }
        }

        for (j=0;j<asize*par->granularity;j++) {
          resultval->x_.push_back(tval[compute_index]->x_[j]);
        }

        if (val[0]->level_ == 0 && val[0]->timestep_ < numsteps_ || val[0]->level_ > 0) {

            // copy over critical info
            resultval->level_ = val[0]->level_;
            resultval->cycle_ = val[0]->cycle_ + 1;
            resultval->max_index_ = tval[compute_index]->max_index_;
            resultval->index_ = tval[compute_index]->index_;
            resultval->value_.resize(asize*par->granularity);
            had_double_type dt = par->dt0/pow(2.0,(int) val[0]->level_);
            had_double_type dx = par->dx0/pow(2.0,(int) val[0]->level_); 
            
            // call rk update 
            int gft = rkupdate(&*vecval.begin(),resultval.get_ptr(),&*vecx.begin(),vecval.size(),
                                 boundary,bbox,adj_index,dt,dx,val[0]->timestep_,
                                 val[0]->iter_,val[0]->level_,*par.p);
            BOOST_ASSERT(gft);

            // increase the iteration counter
            if ( val[0]->iter_ == 5 ) {
              resultval->iter_ = 0;
            } else {
              resultval->iter_ = val[0]->iter_ + 1;
            }

            // refine only after rk subcycles are finished (we don't refine in the midst of rk subcycles)
            if ( resultval->iter_ == 0 ) resultval->refine_ = refinement(&*vecval.begin(),vecval.size(),resultval.get_ptr(),compute_index,boundary,bbox,*par.p);
            else resultval->refine_ = false;

            std::size_t allowedl = par->allowedl;

         //   if ( resultval->refine_ && resultval->level_ < allowedl 
         //        && val[0]->timestep_ >= 1.e-6  ) {
         //     finer_mesh_tapered(result, gids,vecval.size(), row, column, par);
         //   }

            // One special case: refining at time = 0
            if ( resultval->refine_ && 
                 val[0]->timestep_ < 1.e-6 && resultval->level_ < allowedl ) {
              finer_mesh_initial(result, gids, resultval->level_+1,resultval->x_[0],row, column, par);
            }

            if (par->loglevel > 1 && fmod(resultval->timestep_,par->output) < 1.e-6) 
                stubs::logging::logentry(log_, resultval.get(), row,0, par);
        }
        else {
            // the last time step has been reached, just copy over the data
            resultval.get() = val[compute_index].get();
        }
 
        // set return value difference between actual and required number of
        // timesteps (>0: still to go, 0: last step, <0: overdone)
        if ( val[0]->level_ > 0 ) {
          return 0;
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
      //naming::id_type gval[9];
      //access_memory_block<stencil_data> mval[9];

      std::vector<naming::id_type> initial_data;

      naming::id_type here = applier::get_applier().get_runtime_support_gid();
      components::component_type logging_type =
                components::get_component_type<components::amr::server::logging>();
      components::component_type function_type =
                components::get_component_type<components::amr::stencil>();

      bool do_logging = false;
      if ( par->loglevel > 0 ) {
        do_logging = true;
      }

      std::vector<naming::id_type> result_data;
      int numsteps = 2 * 6; // six subcycles each step
      int numvals;
      if ( par->granularity == 1 ) {
        numvals = 17;
      } else if ( par->granularity == 2 ) {
        numvals = 9;
      } else {
        numvals = 6;
      }

      hpx::components::amr::unigrid_mesh unigrid_mesh;
      unigrid_mesh.create(here);

      result_data = unigrid_mesh.execute(initial_data,function_type, numvals, numsteps,
            do_logging ? logging_type : components::component_invalid, par);

      for (std::size_t i = 0; i < result_data.size(); ++i) {
        // free all
        components::stubs::memory_block::free(result_data[i]);
      }

      return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Prep initial data for left (unbiased) tapered mesh
    int stencil::prep_initial_data(std::vector<naming::id_type> & initial_data, 
        std::vector<naming::id_type> const& gids,int vecvalsize, int row,int column, Parameter const& par) 
    {
#if 0
      int i;
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
          mval[i]->refine_ = refinement(&*vecval.begin(),vecval.size(),&(mval[i]->value_),mval[i]->level_,mval[i]->x_,1,boundary,bbox,*par.p);

          // DEBUG
          if (log_)
              stubs::logging::logentry(log_, mval[i].get(), row,2, par);
        }
      }

      for (i=0;i<17;i++) {
        initial_data.push_back(gval[i]);
      }
      return 0;
#endif
    }
    
    ///////////////////////////////////////////////////////////////////////////
    // Implement a finer mesh via interpolation of inter-mesh points
    // Compute the result value for the current time step
    int stencil::finer_mesh_initial(naming::id_type const& result, 
        std::vector<naming::id_type> const& gids, std::size_t level, had_double_type xmin, 
        int row, int column, Parameter const& par) 
    {
      // the initial data for the child mesh comes from the parent mesh
      naming::id_type here = applier::get_applier().get_runtime_support_gid();
      components::component_type logging_type =
                components::get_component_type<components::amr::server::logging>();
      components::component_type function_type =
                components::get_component_type<components::amr::stencil>();

      bool do_logging = false;
      if ( par->loglevel > 0 ) {
        do_logging = true;
      }

      std::vector<naming::id_type> result_data;
      int numsteps = 2 * 6; // six subcycles each step

      int left_half;  // this variable depends upon how close we are to the origin, r=0
      int std_ghostwidth = 8; // standard tapering for stencilsize=7
      had_double_type dx = par->dx0/pow(2.0,level);
      double tmp = (double) ( xmin- par->minx0)/dx; // we have to have an intermediate cast on account of mpfr 
      left_half = (int) tmp;
      if ( left_half > par->ghostwidth/2 + 8 ) left_half = par->ghostwidth/2 + std_ghostwidth;

      int almost_numvals = left_half + par->ghostwidth/2 + 2*par->granularity-1 + std_ghostwidth;

      // to find numvals, divide almost_numvals by the granularity
      int remainder = almost_numvals%par->granularity;
      int numvals = (almost_numvals+remainder)/par->granularity;

      hpx::components::amr::unigrid_mesh unigrid_mesh;
      unigrid_mesh.create(here);
      result_data = unigrid_mesh.init_execute(function_type, numvals, numsteps,
            do_logging ? logging_type : components::component_invalid,level,xmin, par);

      for (std::size_t i = 0; i < result_data.size(); ++i) {
        // free all
        components::stubs::memory_block::free(result_data[i]);
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
            generate_initial_data(val.get_ptr(), item, maxitems, row, level, x, *par.p);

            if (log_ && par->loglevel > 1)         // send initial value to logging instance
                stubs::logging::logentry(log_, val.get(), row,0, par);
        }
        return result;
    }

    int stencil::findpoint(access_memory_block<stencil_data> const& anchor_to_the_left,
                           access_memory_block<stencil_data> const& anchor_to_the_right, 
                           access_memory_block<stencil_data> & resultval) 
    {
#if 0
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
#endif
      return 0;
    }

    void stencil::init(std::size_t numsteps, naming::id_type const& logging)
    {
        numsteps_ = numsteps;
        log_ = logging;
    }

}}}

