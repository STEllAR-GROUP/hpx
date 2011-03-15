//  Copyright (c) 2007-2011 Hartmut Kaiser
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
#include "../nbody/unigrid_mesh.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace nbody 
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

    inline bool 
    stencil::floatcmp_le(had_double_type const& x1, had_double_type const& x2) {
      // compare two floating point numbers
      static had_double_type const epsilon = 1.e-8;

      if ( x1 < x2 ) return true;

      if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
        // the numbers are close enough for coordinate comparison
        return true;
      } else {
        return false;
      }
    }

    inline bool 
    stencil::floatcmp_ge(had_double_type const& x1, had_double_type const& x2) {
      // compare two floating point numbers
      static had_double_type const epsilon = 1.e-8;

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
        hpx::memory::default_vector<access_memory_block<stencil_data> >::type val;
        access_memory_block<stencil_data> resultval = 
            get_memory_block_async(val, gids, result);

        // lock all user defined data elements, will be unlocked at function exit
        scoped_values_lock<lcos::mutex> l(resultval, val); 

        // Here we give the coordinate value to the result (prior to sending it to the user)
        int compute_index;
        
        std::cout << "row: " << row << " column : " << column << std::endl;
        

        if ( val.size() == 0 ) {
          // This should not happen
          BOOST_ASSERT(false);
        }

        if ( val.size() == 1 ) {
          // no restriction needed
          resultval.get() = val[0].get();
          return 0;
        } else {
          compute_index = -1;
          for (int i=0;i<val.size();i++) {
            if ( column == val[i]->column ) {
              compute_index = i;
              break;
            }   
          }    
          BOOST_ASSERT(compute_index != -1);

          resultval->ax = 0.0;
          resultval->ay = 0.0;
          resultval->az = 0.0;
          
          for (int i=0;i<val.size();i++)
          {
              
              if ( i != compute_index ) {
                double dx = val[compute_index]->x - val[i]->x;
                double dy = val[compute_index]->y - val[i]->y;
                double dz = val[compute_index]->z - val[i]->z;
                
                double inv_dr = sqrt (1/(((dx * dx) + (dy * dy) + (dz * dz))+par->softening_2));
                double acc_factor = par->part_mass * inv_dr * inv_dr * inv_dr;
                resultval->ax += dx + acc_factor;
                resultval->ay += dy + acc_factor;
                resultval->az += dz + acc_factor;
              }
              std::cout << "Result Val" << resultval->ax <<" "<<resultval->ay << " " << resultval->az << std::endl;
          }
          
          double vel_dt_half_x, vel_dt_half_y, vel_dt_half_z;
          double v_half_x, v_half_y, v_half_z;
          
          resultval->x = val[compute_index]->x;
          resultval->y = val[compute_index]->y;
          resultval->z = val[compute_index]->z;            
          resultval->vx = val[compute_index]->vx;
          resultval->vy = val[compute_index]->vy;
          resultval->vz = val[compute_index]->vz;
            
          vel_dt_half_x = resultval->ax * par->half_dt;
          vel_dt_half_y = resultval->ay * par->half_dt;
          vel_dt_half_z = resultval->az * par->half_dt;
            
          v_half_x = resultval->vx * par->half_dt;
          v_half_y = resultval->vy * par->half_dt;
          v_half_z = resultval->vz * par->half_dt;
            
          resultval->x += v_half_x * par->dtime;
          resultval->y += v_half_y * par->dtime;
          resultval->z += v_half_z * par->dtime;
            
          resultval->vx += v_half_x + vel_dt_half_x;
          resultval->vy += v_half_y + vel_dt_half_y;
          resultval->vz += v_half_z + vel_dt_half_z;

          std::cout << "Result Val X" << resultval->x <<" "<<resultval->y << " " << resultval->z << std::endl;
          std::cout << "Result Val VX" << resultval->vx <<" "<<resultval->vy << " " << resultval->vz << std::endl;

          ////TODO Put the move function here 
//           for (int i=0;i<val.size();i++)
//           {
//              if ( i != compute_index ) {
//                 
//              }
//           }

       //   resultval.get() = val[compute_index].get();
          return 0;
        }
        BOOST_ASSERT(false);
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

