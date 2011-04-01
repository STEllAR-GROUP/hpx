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
        
//        std::cout << "stencil::eval:: EVAL row: " << row << " column : " << column  << " val.size() " << val.size() << std::endl;
        

        if ( val.size() == 0 ) {
          // This should not happen
          BOOST_ASSERT(false);
        }
/*
        if ( val.size() == 1 ) {
          // no restriction needed
  //        resultval.get() = val[0].get();
          return 0;
        } else*/ 
        {
          compute_index = -1;
          for (int i=0;i<val.size();i++) {
//              std::cout << "stencil::eval:: column: " << column << " val ["<<i<<"] column: " << val[i]->column << " val.size() " << val.size() << std::endl;
            if ( column == val[i]->column ) {
              compute_index = i;
//               std::cout<< "i: " << i << " Compute_index " << compute_index  << " column: " << column << "val ["<<i<<"] column: " << val[i]->column << std::endl;
              break; 
            }   
            
            if (compute_index == -1)
            {
//                std::cout << "stencil::eval::  compute_index = -1 column: " << column << "val ["<<i<<"] column:" << val[i]->column << " val.size() " << val.size() << std::endl;
            }
          }    
          

          BOOST_ASSERT(compute_index != -1);
          resultval.get() = val[compute_index].get();
          

      //    if (resultval->node_type == 1){
          
//           resultval->ax = 0.0;
//           resultval->ay = 0.0;
//           resultval->az = 0.0;
          
          
            unsigned long ci_num_par = 0;
            if (par->extra_pxpar != 0)
            {
                if (compute_index < par->num_pxpar-1)
                    ci_num_par = par->granularity;
                else if (compute_index == par->num_pxpar-1)
                    ci_num_par = par->extra_pxpar;
                else if (compute_index >= par->num_pxpar)
                    BOOST_ASSERT("ERROR: Compute_index is more than number of PX particles");
//                         std::cout << "stencil::eval:: num actual particles in px_par(compute_index) " << compute_index << " is " << ci_num_par << std::endl;
            }
            else if(par->extra_pxpar == 0)
            {
                if (compute_index < par->num_pxpar)
                {
                    ci_num_par = par->granularity;
                    //std::cout << "stencil::evla:: ci_num_par " << ci_num_par << std::endl;
                }
                else if (compute_index >= par->num_pxpar)
                    BOOST_ASSERT("ERROR: Compute_index is more than number of PX particles");
//                         std::cout << "stencil::eval:: num actual particles in px_par(compute_index) " << compute_index << " is " << ci_num_par << std::endl;         
            }
        
         
          for (int i=0;i< val.size();++i)
          {
//               std::cout << "\n VAL i " << i <<  "COMPUTE_INDEX " << compute_index << " val.size() " << val.size() <<std::endl;
//               std::cout << "resultval sizes node_type " << resultval->node_type.size() << " x " << resultval->x.size() <<std::endl;

              
              //int global_idx[ci_num_par]; 
              std::vector<unsigned long> global_idx(ci_num_par,0);
              for(unsigned long d = 0; d < ci_num_par; ++d)
              {
                  global_idx[d] = (compute_index * par->granularity) + d;
                   //std::cout << "stencil::eval:: compute index "<< compute_index << " global_idx " << global_idx[d] << std::endl;
              }
              
              unsigned long i_num_par = 0;
              if (par->extra_pxpar != 0)
              {
                    if (i < par->num_pxpar-1)
                        i_num_par = par->granularity;
                    else if (i == par->num_pxpar-1)
                        i_num_par = par->extra_pxpar;
                    else if (i >= par->num_pxpar)
                        BOOST_ASSERT("ERROR: i is more than number of PX particles");
//                     std::cout << "stencil::eval:: num actual particles in px_par(i) " << i << " is " << i_num_par << std::endl;
              }
              else if(par->extra_pxpar == 0)
              {
                    if (i < par->num_pxpar)
                        i_num_par = par->granularity;
                    else if (i >= par->num_pxpar)
                        BOOST_ASSERT("ERROR: i is more than number of PX particles");
//                     std::cout << "stencil::eval:: num actual particles in px_par(i) " << i << " is " << i_num_par << std::endl;         
              }
              //int remote_idx[i_num_par];
              std::vector<unsigned long> remote_idx(i_num_par,0);
              for(unsigned long d = 0; d < i_num_par; ++d)
              {                   
                  remote_idx[d] = (i * par->granularity) + d;
//                   std::cout << "stencil::eval:: i "<< i << " remote_idx " << remote_idx[d] << std::endl;
              }
              
                        
//           for(int d = 0; d < ci_num_par; ++d)
//           {
//                                 resultval->ax[d] = 0;
//                                 resultval->ay[d] = 0;
//                                 resultval->az[d] = 0;
//           }
              
              if (i != compute_index)
              {
              for(unsigned long d = 0; d < ci_num_par; ++d)
              {
                  if(val[compute_index]->node_type[d] == 1)
                  {
                            for(unsigned long f=0; f < i_num_par; ++f)
                            {
                        for(unsigned long e=0; e < par->iList[global_idx[d]].size(); ++e)
                        {
//                             std::cout << "stencil::eval::  E " << e << " par->iList[global_idx[d]].size() " << par->iList[global_idx[d]].size() << std::endl;

//                                 std::cout << "ilist size " << par->iList[global_idx[d]].size() << " val[compute_index]->node_type " << val[compute_index]->node_type.size() << " d " << d << " ci_num_par " << ci_num_par << " f " << f << " i_num_par " << i_num_par << std::endl;
                                if(par->iList[global_idx[d]][e] == remote_idx[f] && val[compute_index]->node_type[d] == 1 && global_idx[d] != remote_idx[f]) 
                                {
                                std::cout << "stencil::eval:: " << global_idx[d] << " iteracts with " << remote_idx[f] << std::endl;
                                
                                double dx = val[i]->x[f] - val[compute_index]->x[d] ;
                                double dy = val[i]->y[f] - val[compute_index]->y[d] ;
                                double dz = val[i]->z[f] - val[compute_index]->z[d] ;
                                double inv_dr = (1/ (sqrt ((((dx * dx) + (dy * dy) + (dz * dz))+par->softening_2))));
                                double acc_factor = val[i]->mass[f] * inv_dr * inv_dr * inv_dr;
                                std::cout << " dx " << dx << " dy "<< dy <<" dz " << dz << " inv_dr " << inv_dr << " accFactor " << acc_factor << std::endl;
                                resultval->ax[d] += dx * acc_factor;
                                resultval->ay[d] += dy * acc_factor;
                                resultval->az[d] += dz * acc_factor;
                                }
                            }
                        }
                  }
              }
              }

          }
          
//           for(int d = 0; d < ci_num_par; ++d)
//           {
// //               if (resultval->node_type[d] == 1)
// //               {
// //                     double vel_dt_half_x, vel_dt_half_y, vel_dt_half_z;
// //                     double v_half_x, v_half_y, v_half_z;
// //                     resultval->node_type[d] = val[compute_index]->node_type[d];
// //                     resultval->x[d] = val[compute_index]->x[d];
// //                     resultval->y[d] = val[compute_index]->y[d];
// //                     resultval->z[d] = val[compute_index]->z[d];            
// //                     resultval->vx[d] = val[compute_index]->vx[d];
// //                     resultval->vy[d] = val[compute_index]->vy[d];
// //                     resultval->vz[d] = val[compute_index]->vz[d]; 
// //                     
// //                         
// //                     vel_dt_half_x = resultval->ax[d] * par->half_dt;
// //                     vel_dt_half_y = resultval->ay[d] * par->half_dt;
// //                     vel_dt_half_z = resultval->az[d] * par->half_dt;
// //                         
// // //                     v_half_x = resultval->vx[d] * par->half_dt;
// // //                     v_half_y = resultval->vy[d] * par->half_dt;
// // //                     v_half_z = resultval->vz[d] * par->half_dt;
// //                     
// //                     v_half_x = resultval->vx[d] + vel_dt_half_x;
// //                     v_half_y = resultval->vy[d] + vel_dt_half_y;
// //                     v_half_z = resultval->vz[d] + vel_dt_half_z;  
// //                     
// //                     resultval->x[d] += v_half_x * par->dtime;
// //                     resultval->y[d] += v_half_y * par->dtime;
// //                     resultval->z[d] += v_half_z * par->dtime;
// //                         
// //                     resultval->vx[d] += v_half_x + vel_dt_half_x;
// //                     resultval->vy[d] += v_half_y + vel_dt_half_y;
// //                     resultval->vz[d] += v_half_z + vel_dt_half_z;
//                     
//                     std::cout << "Result Val Type: " << resultval->node_type[d] << " compute_index " << compute_index << " d " << d <<std::endl;
//                     std::cout << "Result Val AX Global ID " << (compute_index * par->granularity) + d <<" "<< resultval->ax[d] <<" "<<resultval->ay[d] << " " << resultval->az[d] << std::endl;                    
//                     
// //               }
//               
//           }


        //  } // if node_type == 1 (par)

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

