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
        int compute_index(0);

        if ( val.size() == 3 ) {
          compute_index = (val.size()-1)/2;
        } else if ( val.size() == 2 ) {
          had_double_type mid = 0.5*(par->x1 + par->x2);
          if ( val[0]->x_[0] < mid ) {
            compute_index = 0;
          } else if ( val[1]->x_[0] > mid ) {
            compute_index = 1;
          } else {
            BOOST_ASSERT(false);
          }
        } else {
          BOOST_ASSERT(false);
        }

#if 0
// ------------------------------------------------------
// TEST mode
        resultval.get() = val[compute_index].get();
        resultval->x_ = val[compute_index]->x_;
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

          unlock_scoped_values_lock<lcos::mutex> ul(l);
          stubs::logging::logentry(log_, data, row, 0, par);
        }
        //if ( val[compute_index]->timestep_ >= par->nt0-1 ) {
        //  return 0;
        //}
        return 1;
// END TEST mode
// ------------------------------------------------------
#endif

        // these vectors are used for ghostwidth treatment
        std::vector< had_double_type > alt_vecx;
        std::vector< nodedata > alt_vecval;

        // put all data into a single array
        std::vector< had_double_type* > vecx;
        std::vector< nodedata* > vecval;
 
        std::size_t adj_index(0);
        if ( compute_index == 1 ) adj_index = val[0]->granularity;
        else if ( compute_index == 0 ) {
          adj_index = 2;
        } else {
          BOOST_ASSERT(false);
        }

        std::vector<had_double_type>::iterator iter;
        for (iter=val[0]->x_.begin();iter!=val[0]->x_.end();++iter) vecx.push_back( &(*iter) ); 
        for (iter=val[1]->x_.begin();iter!=val[1]->x_.end();++iter) vecx.push_back( &(*iter) ); 
        if ( val.size() == 3 ) {
          for (iter=val[2]->x_.begin();iter!=val[2]->x_.end();++iter) vecx.push_back( &(*iter) ); 
        }

        std::vector<nodedata>::iterator n_iter;
        for (n_iter=val[0]->value_.begin();n_iter!=val[0]->value_.end();++n_iter) vecval.push_back( &(*n_iter) ); 
        for (n_iter=val[1]->value_.begin();n_iter!=val[1]->value_.end();++n_iter) vecval.push_back( &(*n_iter) ); 
        if ( val.size() == 3 ) {
          for (n_iter=val[2]->value_.begin();n_iter!=val[2]->value_.end();++n_iter) vecval.push_back( &(*n_iter) ); 
        }

        // copy over critical info
        resultval->x_ = val[compute_index]->x_;
        resultval->granularity = val[compute_index]->granularity;
        resultval->level_ = val[compute_index]->level_;
        resultval->cycle_ = val[compute_index]->cycle_ + 1;

        resultval->max_index_ = val[compute_index]->max_index_;
        resultval->granularity = val[compute_index]->granularity;
        resultval->index_ = val[compute_index]->index_;

        resultval->g_startx_ = val[compute_index]->g_startx_;
        resultval->g_endx_ = val[compute_index]->g_endx_;
        resultval->g_dx_ = val[compute_index]->g_dx_;

        if ( val.size() == 3 ) {
          // ghostwidth {{{
          // ghostwidth interpolation can only occur when rows contain aligned timesteps are aligned
          // During other iterations, points are treated as artificial boundaries and eventually tapered.
          if ( val[0]->level_ != val[1]->level_ || val[1]->level_ != val[2]->level_ ) {
            // sanity checks
            BOOST_ASSERT(compute_index == 1);
            BOOST_ASSERT(adj_index == val[0]->granularity);

            resultval->g_startx_ = val[compute_index]->x_[0];
            resultval->g_endx_ = val[compute_index]->x_[val[compute_index]->granularity-1];
            resultval->g_dx_ = val[compute_index]->x_[1]-val[compute_index]->x_[0];

            // There are four cases:  each involve either a downsampling or interpolation
            if (val[0]->level_ != val[1]->level_ && val[1]->level_ == val[2]->level_ ) { 
              if ( val[0]->level_ > val[1]->level_ ) {
                // Case I {{{
//                had_double_type dx = *vecx[1] - *vecx[0];
                // downsample val[0]
                int half;
                if ( val[0]->granularity%2 == 0 ) {
                  half = val[0]->granularity/2;
                } else {
                  half = (val[0]->granularity-1)/2;
                }
                alt_vecval.resize(half + val[1]->granularity + val[2]->granularity);
                alt_vecx.resize(half + val[1]->granularity + val[2]->granularity);

                // points in val[1] and val[2] remain the same
                int count = half;
                for (int j=adj_index;j<vecx.size();j++) {
                  alt_vecx[count] = *vecx[j];
                  alt_vecval[count] = *vecval[j];
                  count++;
                }

                count = half-1;
                for (int j=adj_index-1;j>=0;j=j-2) {
                  alt_vecx[count] = *vecx[j];
                  alt_vecval[count] = *vecval[j];
                  count--;
                }

                adj_index = half;

                vecx.resize(0);
                vecval.resize(0);
                for (iter=alt_vecx.begin();iter!=alt_vecx.end();++iter) vecx.push_back( &(*iter) ); 
                for (n_iter=alt_vecval.begin();n_iter!=alt_vecval.end();++n_iter) vecval.push_back( &(*n_iter) ); 
                // }}}
              } else if ( val[0]->level_ < val[1]->level_ ) {
                // Case II {{{
                // interpolate val[0]
                had_double_type dx = *vecx[vecx.size()-1] - *vecx[vecx.size()-2];

                alt_vecval.resize(2*val[0]->granularity-1 + val[1]->granularity + val[2]->granularity);
                alt_vecx.resize(2*val[0]->granularity-1 + val[1]->granularity + val[2]->granularity);

                // no interpolation needed for points in val[1], and the first point in val[2]
                std::size_t start;
                std::size_t offset;
                start = val[0]->granularity+val[1]->granularity-1;
                offset = 2*val[0]->granularity-1;
                for (int j=0;j<=start;j++) {
                  alt_vecx[j+offset] = *vecx[j+val[0]->granularity];
                  alt_vecval[j+offset] = *vecval[j+val[0]->granularity];
                }

                // set up the new 'x' vector
                for (int j=0;j<offset;j++) {
                  alt_vecx[j] = *vecx[0] + j*dx;
                }

                // set up the new 'values' vector
                int count = 1;
                for (int j=1;j<offset;j++) {
                  if ( count%2 == 0 ) {
                    alt_vecval[j] = *vecval[count/2];
                  } else {
                    // linear interpolation
                    for (int i=0;i<num_eqns;i++) {
                      alt_vecval[j].phi[0][i] = 0.5*vecval[(count-1)/2]->phi[0][i] 
                                              + 0.5*vecval[(count+1)/2]->phi[0][i];
                      // note that we do not interpolate the phi[1] variables since interpolation
                      // only occurs after the 3 rk steps (i.e. rk_iter = 0).  phi[1] has not impact at rk_iter=0.
                    }
                  }
                  count++;
                }
 
                adj_index = 2;

                // temporarily change the resultval->granularity
                resultval->granularity = 2*val[0]->granularity-1 + val[1]->granularity;
                resultval->x_.resize(resultval->granularity);
                for (int j = 0; j < resultval->granularity; j++) {
                  resultval->x_[j] = alt_vecx[j];
                }

                vecx.resize(0);
                vecval.resize(0);
                for (iter=alt_vecx.begin();iter!=alt_vecx.end();++iter) vecx.push_back( &(*iter) ); 
                for (n_iter=alt_vecval.begin();n_iter!=alt_vecval.end();++n_iter) vecval.push_back( &(*n_iter) ); 
                // }}}
              } else {
                BOOST_ASSERT(false);
              }
            } else if (val[2]->level_ != val[1]->level_ && val[0]->level_ == val[1]->level_) {
              if ( val[2]->level_ < val[1]->level_ ) {
                // Case III {{{
                had_double_type dx = *vecx[1] - *vecx[0];

                // interpolate val[2]
                alt_vecval.resize(val[0]->granularity + val[1]->granularity + 2*val[2]->granularity-1);
                alt_vecx.resize(val[0]->granularity + val[1]->granularity + 2*val[2]->granularity-1);

                // no interpolation needed for points in val[0], val[1], and the first point in val[2]
                std::size_t start;
                start = val[0]->granularity+val[1]->granularity-1;
                for (int j=0;j<=start;j++) {
                  alt_vecx[j] = *vecx[j];
                  alt_vecval[j] = *vecval[j];
                }

                // set up the new 'x' vector
                for (int j=start;j<alt_vecx.size();j++) {
                  alt_vecx[j] = *vecx[start] + (j-start)*dx;
                }

                // set up the new 'values' vector
                int count = 1;
                for (int j=start+1;j<alt_vecx.size();j++) {
                  if ( count%2 == 0 ) {
                    alt_vecval[j] = *vecval[start+count/2];
                  } else {
                    // linear interpolation
                    for (int i=0;i<num_eqns;i++) {
                      alt_vecval[j].phi[0][i] = 0.5*vecval[start+(count-1)/2]->phi[0][i] 
                                              + 0.5*vecval[start+(count+1)/2]->phi[0][i];
                      // note that we do not interpolate the phi[1] variables since interpolation
                      // only occurs after the 3 rk steps (i.e. rk_iter = 0).  phi[1] has not impact at rk_iter=0.
                    }
                  }
                  count++;
                }

                // temporarily change the resultval->granularity
                resultval->granularity = val[1]->granularity + 2*val[2]->granularity-1;
                resultval->x_.resize(resultval->granularity);
                for (int j = 0; j < resultval->granularity; j++) {
                  resultval->x_[j] = alt_vecx[j+adj_index];
                }

                vecx.resize(0);
                vecval.resize(0);
                for (iter=alt_vecx.begin();iter!=alt_vecx.end();++iter) vecx.push_back( &(*iter) ); 
                for (n_iter=alt_vecval.begin();n_iter!=alt_vecval.end();++n_iter) vecval.push_back( &(*n_iter) ); 
                // }}}
              } else if ( val[2]->level_ > val[1]->level_ ) {
                // Case IV {{{
                // downsample val[2]
//                had_double_type dx = *vecx[vecx.size()-1] - *vecx[vecx.size()-2];

                int half;
                if ( val[2]->granularity%2 == 0 ) {
                  half = val[2]->granularity/2;
                } else {
                  half = (val[2]->granularity-1)/2;
                }
                alt_vecval.resize(val[0]->granularity + val[1]->granularity + half);
                alt_vecx.resize(val[0]->granularity + val[1]->granularity + half);

                // points in val[0] and val[1] remain the same
                std::size_t start = val[0]->granularity + val[1]->granularity;
                for (int j=0;j<start;j++) {
                  alt_vecx[j] = *vecx[j];
                  alt_vecval[j] = *vecval[j];
                }

                std::size_t count = start;
                for (int j=start;j<vecx.size();j=j+2) {
                  alt_vecx[count] = *vecx[j];
                  alt_vecval[count] = *vecval[j];
                  count++;
                }

                vecx.resize(0);
                vecval.resize(0);
                for (iter=alt_vecx.begin();iter!=alt_vecx.end();++iter) vecx.push_back( &(*iter) ); 
                for (n_iter=alt_vecval.begin();n_iter!=alt_vecval.end();++n_iter) vecval.push_back( &(*n_iter) ); 
                // }}}
              } else {
                BOOST_ASSERT(false);
              }
            }
          }
          // }}}
        }
        if ( val.size() == 2 ) {
          if ( val[0]->level_ != val[1]->level_ ) {
            // This protects the user from picking a granularity too large with nx0 too small
            BOOST_ASSERT(floatcmp(*vecx[1] - *vecx[0],*vecx[vecx.size()-1]-*vecx[vecx.size()-2]));
          }
        }

        // DEBUG
        //char description[80];
        //double dasx = (double) resultval->x_[0];
        //double dast = (double) resultval->timestep_;
        //snprintf(description,sizeof(description),"x: %g t: %g level: %d",dasx,dast,val[0]->level_);
        //threads::thread_self& self = threads::get_self();
        //threads::thread_id_type id = self.get_thread_id();
        //threads::set_thread_description(id,description);

        if (val[compute_index]->timestep_ < (int)numsteps_) {

            // copy over critical info
            resultval->x_.resize(resultval->granularity);
            resultval->value_.resize(resultval->granularity);

            int level = val[compute_index]->level_;

            had_double_type dt = par->dt0/pow(2.0,level);
            had_double_type dx = par->dx0/pow(2.0,level); 

            // DEBUG
            //for (int j=0;j<vecx.size()-1;j++) {
            //  if ( floatcmp(*vecx[j+1]-*vecx[j],dx) == 0 ) {
            //     BOOST_ASSERT(false);
            //  }
            //}

            int gft = 0;

            // call rk update 
            gft = rkupdate(vecval,resultval.get_ptr(),vecx,vecval.size(),
                           adj_index,dt,dx,val[compute_index]->timestep_,
                           level,*par.p);

            BOOST_ASSERT(gft);

            // ghostwidth resizing
            if ( val.size() == 2 ) {
              if ( resultval->granularity != par->granularity ) {
                  // tapering {{{

                  int count = 0;
                  for (int j=resultval->x_.size()-1;j>=0;j--) {
                    if ( floatcmp(resultval->x_[j],resultval->g_endx_ - count*resultval->g_dx_) == 1 ) {
                      count++;
                    }  else {
                      resultval->x_.erase(resultval->x_.begin()+j);
                      resultval->value_.erase(resultval->value_.begin()+j);
                    }
                  }

                  resultval->granularity = par->granularity;

                  BOOST_ASSERT(floatcmp(resultval->x_[0],resultval->g_startx_) == 1);
                  BOOST_ASSERT(floatcmp(resultval->x_[par->granularity-1],resultval->g_endx_) == 1);
                  BOOST_ASSERT(resultval->x_.size() == par->granularity);
                  // }}}
              }
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
        if ( val[compute_index]->timestep_ >= par->nt0-2 ) {
          return 0;
        }
        return 1;
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

