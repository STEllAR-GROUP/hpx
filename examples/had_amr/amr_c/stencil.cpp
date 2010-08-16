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
        std::vector<access_memory_block<stencil_data> > val, tval;
        access_memory_block<stencil_data> resultval = 
            get_memory_block_async(val, gids, result);

        // lock all user defined data elements, will be unlocked at function exit
        scoped_values_lock<lcos::mutex> l(resultval, val);

        // Here we give the coordinate value to the result (prior to sending it to the user)
        int compute_index;
        bool boundary = false;
        int bbox[2] = { 0, 0 };   // initialize bounding box

        if ( val[0]->iter_ != val[val.size()-1]->iter_ ) {
          for (int i = 0; i < val.size(); i++) {
            if ( val[0]->iter_ == val[i]->iter_ ) 
                tval.push_back(val[i]);
          }
        } 
        else {
          for (int i = 0; i < val.size(); i++) 
              tval.push_back(val[i]);
        }

        if ( tval[0]->level_ == 0 ) {
          int numvals = par->nx0/par->granularity;
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
            } else {
              BOOST_ASSERT(false);
            }
          }
        } 
        else {
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
          } else if ( column == val[0]->max_index_-1 && floatcmp(par->maxx0,tval[compute_index]->x_[tval[compute_index]->granularity-1]) ) {
            boundary = true;
            bbox[1] = 1;
          }
        } 

        // put all data into a single array
        std::vector< had_double_type > vecx;
        std::vector< nodedata > vecval;

        std::size_t count = 0;
        std::size_t adj_index = -1;
        for (int i = 0; i < tval.size(); i++) {
          for (int j = 0; j < tval[i]->granularity; j++) {
            vecval.push_back(tval[i]->value_[j]);
            vecx.push_back(tval[i]->x_[j]);
            if ( i == compute_index && adj_index == -1 ) {
              adj_index = count; 
            }
            count++;
          }
        }

        for (int j = 0; j < tval[compute_index]->granularity; j++) {
          resultval->x_.push_back(tval[compute_index]->x_[j]);
        }

        // initialize result 
        resultval->overwrite_alloc_ = false;
        resultval->right_alloc_ = false;

        // DEBUG
        char description[80];
        double dasx = (double) resultval->x_[0];
        double dast = (double) resultval->timestep_;
        snprintf(description,sizeof(description),"x: %g t: %g level: %d",dasx,dast,val[0]->level_);
        threads::thread_self& self = threads::get_self();
        threads::thread_id_type id = self.get_thread_id();
        threads::set_thread_description(id,description);

        if ((val[0]->level_ == 0 && val[0]->timestep_ < numsteps_) || val[0]->level_ > 0) {

            // copy over critical info
            resultval->level_ = val[0]->level_;
            resultval->cycle_ = val[0]->cycle_ + 1;
            resultval->max_index_ = tval[compute_index]->max_index_;
            resultval->granularity = tval[compute_index]->granularity;
            resultval->index_ = tval[compute_index]->index_;
            resultval->value_.resize(tval[compute_index]->granularity);
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

            // refine only after rk subcycles are finished (we don't refine in the midst of rk subcycles)
            if ( resultval->iter_ == 0 ) {
                resultval->refine_ = refinement(&*vecval.begin(), vecval.size(),
                    resultval.get_ptr(),compute_index,boundary,bbox,*par.p);
            }
            else {
                resultval->refine_ = false;
            }

            if ( resultval->refine_ && resultval->level_ < par->allowedl ) {
                unlock_scoped_values_lock<lcos::mutex> ul(l);
                finer_mesh(gids, resultval, val, vecval.size(), tval.size(),
                    compute_index, row, column, par);
            } 
            else {
                resultval->overwrite_alloc_ = false;
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
        if ( val[0]->level_ > 0 ) {
            return 0;
        } 
        else {
          BOOST_ASSERT(numsteps_%6 == 0);
          int t = resultval->cycle_;
          int r = numsteps_ - t;
          int m = r/6;
//           std::cout << " TEST t " << t << " r " << r << " numsteps " << numsteps_ << " m " << m << std::endl;
          return m;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Implement a finer mesh via interpolation of inter-mesh points
    // Compute the result value for the current time step
    int stencil::finer_mesh(std::vector<naming::id_type> const& gids,
        access_memory_block<stencil_data>& resultval,
        std::vector<access_memory_block<stencil_data> >& vals,   // needed for locking purposes only
        std::size_t vecvalsize, std::size_t size, 
        std::size_t compute_index, std::size_t row, std::size_t column, 
        Parameter const& par) 
    {
      naming::id_type here = applier::get_applier().get_runtime_support_gid();
      components::component_type logging_type =
          components::get_component_type<components::amr::server::logging>();
      components::component_type function_type =
          components::get_component_type<components::amr::stencil>();

      std::size_t numsteps = 2 * 3;     // three subcycles each step
      std::size_t numvals = 2 * size;

      //BOOST_ASSERT(size*par->granularity == vecvalsize);

      std::vector<naming::id_type> initial_data;

      {
          scoped_values_lock<lcos::mutex> l(vals);
          prep_initial_data(initial_data, gids, vecvalsize, size, row, column, numvals, par);
      }

      hpx::components::amr::unigrid_mesh unigrid_mesh;
      unigrid_mesh.create(here);

      std::vector<naming::id_type> result_data = 
          unigrid_mesh.execute(initial_data, function_type, numvals, numsteps, 
              (par->loglevel > 0) ? logging_type : components::component_invalid, par);

      // prepare for restriction
      prep_restriction_data(result_data, numvals, size, par);

      access_memory_block<stencil_data> overwrite = 
          hpx::components::stubs::memory_block::get(result_data[compute_index]);

      {
          scoped_values_lock<lcos::mutex> l(overwrite, resultval);

          resultval->overwrite_alloc_ = true;
          resultval->overwrite_ = result_data[compute_index];
          resultval->value_ = overwrite->value_;

          // remember neighbors
          overwrite->right_alloc_ = true;
          overwrite->right_ = result_data[compute_index+size];
      }

      // DEBUG
      if ( par->loglevel > 0 ) {
        access_memory_block<stencil_data> amb1 =
                         hpx::components::stubs::memory_block::get(result_data[compute_index+size]);
        stubs::logging::logentry(log_, amb1.get(), row,1, par);
      }

      // TEST
      //if ( floatcmp(amb1->x_[0],2.60593,1.e-4) ) {
      //  std::cout << "WRITE TEST : " << amb1->x_[0] << " time " << amb1->timestep_  << std::endl;
      //  std::cout << "WRITE TEST neighbor : " << resultval->x_[0] << " chi value " << resultval->value_[0].phi[0][0] << std::endl;
      //  std::cout << " level " << resultval->level_  << " dx " << amb1->x_[0] - resultval->x_[0] << std::endl;
      //}
      // END TEST

      return 0;
    }
    ///////////////////////////////////////////////////////////////////////////
    // Prep restriction data 
    int stencil::prep_restriction_data(
        std::vector<naming::id_type> & result_data,
        std::size_t numvals, std::size_t size, Parameter const& par)
    {
      std::vector<access_memory_block<stencil_data> > mval;
      get_memory_block_async(mval, result_data);

      scoped_values_lock<lcos::mutex> l(mval);

      BOOST_ASSERT(numvals == result_data.size());
      std::vector<had_double_type> phi,x;
      phi.resize(numvals*par->granularity*num_eqns);
      x.resize(numvals*par->granularity);

      for (int i = 0; i < numvals; i++) {
        for (int j = 0; j < mval[i]->granularity; j++) {
          x[j + i*par->granularity] = mval[i]->x_[j];
          for (int k = 0; k < num_eqns; k++) {
            phi[k + num_eqns*(j+i*par->granularity)] = mval[i]->value_[j].phi[0][k];
          }
        }
      }

      std::size_t count = 0;
      std::size_t count1 = 0;
      std::size_t count2 = 0;
      std::size_t step1 = 0;
      std::size_t step2 = size;

      for (int i = 0; i < numvals; i++) {
        for (int j = 0; j < mval[i]->granularity; j++) {
          if (count % 2 == 0) {
            if ( step1 == numvals ) {
              BOOST_ASSERT(false);
            }
            mval[step1]->x_[count1] = x[j+i*par->granularity];
            for (int k = 0; k < num_eqns; k++) {
              mval[step1]->value_[count1].phi[0][k] = phi[k + num_eqns*(j+i*par->granularity)];
            }
            count1++;
            if (count1 == mval[step1]->granularity) {
              count1 = 0;
              step1++;
            }
            
          } else {
            if ( step2 == numvals ) {
              BOOST_ASSERT(false);
            }
            mval[step2]->x_[count2] = x[j+i*par->granularity];
            for (int k = 0; k < num_eqns; k++) {
              mval[step2]->value_[count2].phi[0][k] = phi[k + num_eqns*(j+i*par->granularity)];
            }
            count2++;
            if (count2 == mval[step2]->granularity) {
              count2 = 0;
              step2++;
            }
          }
          count++;
        }
      }

      return 0;
    };

    ///////////////////////////////////////////////////////////////////////////
    // Prep initial data 
    int stencil::prep_initial_data(std::vector<naming::id_type> & initial_data, 
        std::vector<naming::id_type> const& gids, std::size_t vecvalsize, 
        std::size_t size, std::size_t row, std::size_t column, 
        std::size_t numvals, Parameter const& par) 
    {
      naming::id_type gval[6];
      std::vector<access_memory_block<stencil_data> > mval;

      int std_index;
      if ( gids.size() != vecvalsize ) {
        std_index = size;
      } else {
        std_index = 0;
      }

      if ( size == 3 ) {
        boost::tie(gval[0], gval[1], gval[2], gval[3], gval[4], gval[5]) = 
            components::wait(
                components::stubs::memory_block::clone_async(gids[std_index]), 
                components::stubs::memory_block::clone_async(gids[std_index+1]),
                components::stubs::memory_block::clone_async(gids[std_index+2]),
                components::stubs::memory_block::clone_async(gids[std_index]),
                components::stubs::memory_block::clone_async(gids[std_index]),
                components::stubs::memory_block::clone_async(gids[std_index]));

        mval.resize(6);
        boost::tie(mval[0], mval[1], mval[2], mval[3], mval[4], mval[5]) = 
          get_memory_block_async<stencil_data>(gval[0], gval[1], gval[2], gval[3], gval[4], gval[5]);
   
      } 
      else if ( size == 2 ) {
        boost::tie(gval[0], gval[1], gval[2], gval[3]) = 
            components::wait(
                components::stubs::memory_block::clone_async(gids[std_index]), 
                components::stubs::memory_block::clone_async(gids[std_index+1]),
                components::stubs::memory_block::clone_async(gids[std_index]),
                components::stubs::memory_block::clone_async(gids[std_index]));

        mval.resize(4);
        boost::tie(mval[0], mval[1], mval[2], mval[3]) = 
          get_memory_block_async<stencil_data>(gval[0], gval[1], gval[2], gval[3]);
      } 
      else {
        BOOST_ASSERT(false);
      }

      {
        scoped_values_lock<lcos::mutex> l(mval);

        had_double_type dx = mval[0]->x_[1]- mval[0]->x_[0];

        // make sure the cloned gids have a consistent granularity
        //for (i=0;i<size;i++) {
        //  mval[i+size]->granularity = mval[i]->granularity;
        //}

        // the last gid of the AMR mesh has a slightly smaller granularity
        mval[2*size-1]->granularity = mval[size-1]->granularity-1;

        for (int i = 0; i < 2*size; i++) {
          // increase the level by one
          ++mval[i]->level_;
          mval[i]->index_ = i;
          mval[i]->iter_ = 0;
          mval[i]->max_index_ = 2*size;
          if ( i >= size ) {
            mval[i]->right_alloc_ = false;
            mval[i]->overwrite_alloc_ = false;
            for (int j = 0; j < mval[i]->granularity; j++) {
              mval[i]->x_[j]  = mval[i-size]->x_[j] + 0.5*dx;
            }
          }
        }

      for (int i = 0; i < size; i++) {
        int s = findpoint(mval[i],mval[i+1],mval[i+size]);
        //std::cout << " TEST findpoint s: " << s << " i " << i << " " << mval[i]->overwrite_alloc_ << " x " << mval[i+size]->x_[0] << std::endl;
        if ( mval[0]->timestep_ < 1.e-6 ) {
          // don't interpolate initial data
          initial_data_aux(mval[i+size].get_ptr(),*par.p);
          s = 1;
        }
        if ( s == 0 ) { 
          // DEBUG
          if ( par->loglevel > 0 ) {
            stubs::logging::logentry(log_, mval[i+size].get(), row,2, par);
          }

          // TEST
          //if ( floatcmp(mval[i+size]->x_[0],2.60593,1.e-4) ) {
          //  std::cout << " TEST neighbors: " << mval[i]->x_[0] << " " << mval[i+1]->x_[0] << " i " << i << " size " << size << std::endl;
          //  std::cout << " TEST alloc: " << mval[i]->overwrite_alloc_ << " right alloc " << mval[i]->right_alloc_ << " timestep " << mval[i+size]->timestep_ << " level " << mval[i+size]->level_ << std::endl;
        //std::cout << "TEST chi value " << mval[i]->value_[0].phi[0][0] << std::endl;
        //    std::cout << " TEST: dx " << mval[i+size]->x_[0] - mval[i]->x_[0]  << std::endl;
        //    BOOST_ASSERT(false);
        //  }
          // END TEST
          
            // point not found -- interpolate
            for (int j = 0; j < mval[i]->granularity-1; j++) {
              for (int k = 0; k < num_eqns; k++) {
                mval[i+size]->value_[j].phi[0][k] = 0.5*(mval[i]->value_[j].phi[0][k] + mval[i]->value_[j+1].phi[0][k] );
              }
            }

            if ( mval[i+size]->granularity > mval[i]->granularity-1 ) {
              // last point to fill
              BOOST_ASSERT(i+1 < size );
              for (int k = 0; k < num_eqns; k++) {
                mval[i+size]->value_[ mval[i+size]->granularity-1  ].phi[0][k] = 
                  0.5*(mval[i]->value_[mval[i]->granularity-1].phi[0][k] + mval[i+1]->value_[0].phi[0][k]);
              }
            }
          }
        }
  
        // re-order things so they can be used
        std::vector<had_double_type> phi, x;
        phi.resize(2*size*par->granularity*num_eqns);
        x.resize(2*size*par->granularity);

        std::size_t ct = 0;
        std::size_t ct1 = 0;
        std::size_t stp1 = 0;
        std::size_t ct2 = 0;
        std::size_t stp2 = size;
        std::size_t ct3 = 0;
        std::size_t stp3 = 0;
        for (int i = 0; i < 2*size; i++) {
          for (int j = 0; j < mval[i]->granularity; j++) {
            if (ct % 2 == 0) {
              x[ct3 + stp3*par->granularity] = mval[stp1]->x_[ct1];
              for (int k = 0; k < num_eqns; k++) {
                phi[k + num_eqns*(ct3+stp3*par->granularity)] = mval[stp1]->value_[ct1].phi[0][k];
              }
              ct1++;
              if ( ct1 == mval[stp1]->granularity ) {
                stp1++;
                ct1 = 0;
              }
            } else {
              x[ct3 + stp3*par->granularity] = mval[stp2]->x_[ct2];
              for (int k = 0; k < num_eqns; k++) {
                phi[k + num_eqns*(ct3+stp3*par->granularity)] = mval[stp2]->value_[ct2].phi[0][k];
              }
              ct2++;
              if ( ct2 == mval[stp2]->granularity ) {
                stp2++;
                ct2 = 0;
              }
            } 
            ct++;

            ct3++;
            if ( ct3 == par->granularity ) {
              stp3++;
              ct3 = 0;
            }
          }
        }

        std::size_t count = 0;
        std::size_t step = 0;

        for (int i = 0; i < 2*size; i++) {
          for (int j = 0; j < mval[i]->granularity; j++) {
            mval[i]->x_[j] = x[count+step*par->granularity];
            for (int k = 0; k < num_eqns; k++) {
              mval[i]->value_[j].phi[0][k] = phi[k + num_eqns*(count+step*par->granularity)];
            }
            count++;
            if ( count == par->granularity ) {
              step++;
              count = 0;
            }
          }
        }
      }

      if ( mval[0]->timestep_ < 1.e-6 && par->loglevel > 1) {
        for (int i = 0; i < 2*size; i++) {
          stubs::logging::logentry(log_, mval[i].get(),0,0, par);
        }
      }

      for (int i = 0; i < 2*size; i++) {
        initial_data.push_back(gval[i]);
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
      // the pinball machine
      int s = 0;
      int sup = resultval->granularity-1;
      access_memory_block<stencil_data> amb0;
      amb0 = anchor_to_the_left;
      if (s == 0 && amb0->overwrite_alloc_ == 1) {
        access_memory_block<stencil_data> amb1 = hpx::components::stubs::memory_block::get(amb0->overwrite_);

        // look around
        if ( amb1->right_alloc_ == 1 ) {
          access_memory_block<stencil_data> amb2 = hpx::components::stubs::memory_block::get(amb1->right_);
          if ( floatcmp(amb2->x_[0],resultval->x_[0]) || floatcmp(amb2->x_[0],resultval->x_[sup]) ) {
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
      }


      amb0 = anchor_to_the_right;
      if (s == 0 && amb0->overwrite_alloc_ == 1) {
        access_memory_block<stencil_data> amb1 = hpx::components::stubs::memory_block::get(amb0->overwrite_);

        // look around
        if ( amb1->right_alloc_ == 1 ) {
          access_memory_block<stencil_data> amb2 = hpx::components::stubs::memory_block::get(amb1->right_);
          if ( floatcmp(amb2->x_[0],resultval->x_[0]) || floatcmp(amb2->x_[0],resultval->x_[sup]) ) {
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
      }

      return s;
    }

    void stencil::init(std::size_t numsteps, naming::id_type const& logging)
    {
        numsteps_ = numsteps;
        log_ = logging;
    }

}}}

