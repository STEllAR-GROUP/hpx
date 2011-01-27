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

    inline std::size_t stencil::findlevel3D(std::size_t step, std::size_t item, 
                                            std::size_t &a, std::size_t &b, std::size_t &c, Parameter const& par)
    {
      int ll = par->level_row[step];
      // discover what level to which this point belongs
      int level = -1;
      if ( ll == par->allowedl ) {
        level = ll;
        // get 3D coordinates from 'i' value
        // i.e. i = a + nx*(b+c*nx);
        int tmp_index = item/par->nx[ll];
        c = tmp_index/par->nx[ll];
        b = tmp_index%par->nx[ll];
        a = item - par->nx[ll]*(b+c*par->nx[ll]);
        BOOST_ASSERT(item == a + par->nx[ll]*(b+c*par->nx[ll]));
      } else {
        if ( item < par->rowsize[par->allowedl] ) {
          level = par->allowedl;
        } else {
          for (int j=par->allowedl-1;j>=ll;j--) {
            if ( item < par->rowsize[j] && item >= par->rowsize[j+1] ) {
              level = j;
              break;
            }
          }
        }

        if ( level < par->allowedl ) {
          int tmp_index = (item - par->rowsize[level+1])/par->nx[level];
          c = tmp_index/par->nx[level];
          b = tmp_index%par->nx[level];
          a = (item-par->rowsize[level+1]) - par->nx[level]*(b+c*par->nx[level]);
          BOOST_ASSERT(item-par->rowsize[level+1] == a + par->nx[level]*(b+c*par->nx[level]));
        } else {
          int tmp_index = item/par->nx[level];
          c = tmp_index/par->nx[level];
          b = tmp_index%par->nx[level];
          a = item - par->nx[level]*(b+c*par->nx[level]);
        }
      }
      BOOST_ASSERT(level >= 0);
      return level;
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

        if ( val.size() == 0 ) {
          // This should not happen
          BOOST_ASSERT(false);
        }

        // Check if this is a prolongation/restriction step
        if ( (row+5)%3 == 0 && par->allowedl != 0 ) {
          // This is a prolongation/restriction step
          if ( val.size() == 1 ) {
            // no restriction needed
            resultval.get() = val[0].get();
            if ( val[0]->timestep_ >= par->nt0-2 ) {
              return 0;
            } 
            return 1;
          } else {
            std::size_t a,b,c;
            int level = findlevel3D(row,column,a,b,c,par);
            had_double_type dx = par->dx0/pow(2.0,level);
            had_double_type x = par->min[level] + a*dx*par->granularity;
            had_double_type y = par->min[level] + b*dx*par->granularity;
            had_double_type z = par->min[level] + c*dx*par->granularity;
            compute_index = -1;
            for (int i=0;i<val.size();i++) {
              if ( floatcmp(x,val[i]->x_[0]) == 1 && 
                   floatcmp(y,val[i]->y_[0]) == 1 && 
                   floatcmp(z,val[i]->z_[0]) == 1 ) {
                compute_index = i;
                break;
              }
            }
            if ( compute_index == -1 ) {
              std::cout << " PROBLEM LOCATING x " << x << " y " << y << " z " << z << " val size " << val.size() << " level " << level << std::endl;
              BOOST_ASSERT(false);
            }
            resultval.get() = val[compute_index].get();
            if ( val[compute_index]->timestep_ >= par->nt0-2 ) {
              return 0;
            }
            return 1;
#if 0
            // restriction needed
            std::size_t a,b,c;
            int level = findlevel3D(row,column,a,b,c,par);
            had_double_type dx = par->dx0/pow(2.0,level);
            had_double_type x = par->min[level] + a*dx*par->granularity;
            had_double_type y = par->min[level] + b*dx*par->granularity;
            had_double_type z = par->min[level] + c*dx*par->granularity;
            compute_index = -1;
            for (int i=0;i<val.size();i++) {
              if ( floatcmp(x,val[i]->x_[0]) == 1 && 
                   floatcmp(y,val[i]->y_[0]) == 1 && 
                   floatcmp(z,val[i]->z_[0]) == 1 ) {
                compute_index = i;
                break;
              }
            }
            if ( compute_index == -1 ) {
              std::cout << " PROBLEM LOCATING x " << x << " y " << y << " z " << z << " val size " << val.size() << " level " << level << std::endl;
              BOOST_ASSERT(false);
            }

            // copy over critical info
            resultval->x_ = val[compute_index]->x_;
            resultval->y_ = val[compute_index]->y_;
            resultval->z_ = val[compute_index]->z_;
            resultval->value_.resize(val[compute_index]->value_.size());
            resultval->granularity = val[compute_index]->granularity;
            resultval->level_ = val[compute_index]->level_;
  
            resultval->max_index_ = val[compute_index]->max_index_;
            resultval->granularity = val[compute_index]->granularity;
            resultval->index_ = val[compute_index]->index_;

            // restriction
            int n = par->granularity;
            int last_time = -1;
            bool found = false;
            had_double_type xx,yy,zz;
            for (int k=0;k<n;k++) {
              zz = z + k*dx;
            for (int j=0;j<n;j++) {
              yy = y + j*dx;
            for (int i=0;i<n;i++) {
              xx = x + i*dx;
              found = false;

              // NOTE:: be sure you are getting the highest level of the point available
              // this still needs to be implemented

              if ( last_time != -1 ) {
                // this might save some time -- see if the point is here
                if ( xx >= val[last_time]->x_[0] && xx <= val[last_time]->x_[par->granularity-1] &&
                     yy >= val[last_time]->y_[0] && yy <= val[last_time]->y_[par->granularity-1] &&
                     zz >= val[last_time]->z_[0] && zz <= val[last_time]->z_[par->granularity-1] ) {
                  found = true;
                }
              }

              if ( !found ) {
                // find who has this point
                for (int ii=0;ii<val.size();ii++) {
                  if ( (xx >= val[ii]->x_[0] || floatcmp(xx,val[ii]->x_[0])==1) && 
                       (xx <= val[ii]->x_[par->granularity-1] || floatcmp(xx,val[ii]->x_[par->granularity-1])==1) &&
                       (yy >= val[ii]->y_[0] || floatcmp(yy,val[ii]->y_[0])==1)  && 
                       (yy <= val[ii]->y_[par->granularity-1] || floatcmp(yy,val[ii]->y_[par->granularity-1])==1) &&
                       (zz >= val[ii]->z_[0] || floatcmp(zz,val[ii]->z_[0])==1) && 
                       (zz <= val[ii]->z_[par->granularity-1] || floatcmp(zz,val[ii]->z_[par->granularity-1])==1) ) {
                    found = true;
                    last_time = ii;
                    break;
                  }
                }
              }

              if ( !found ) {
                std::cout << " DEBUG coords " << xx << " " << yy << " " << zz <<  std::endl;
                for (int ii=0;ii<val.size();ii++) {
                  std::cout << " DEBUG available x " << val[ii]->x_[0] << " " << val[ii]->x_[par->granularity-1] << " " <<  std::endl;
                  std::cout << " DEBUG available y " << val[ii]->y_[0] << " " << val[ii]->y_[par->granularity-1] << " " <<  std::endl;
                  std::cout << " DEBUG available z " << val[ii]->z_[0] << " " << val[ii]->z_[par->granularity-1] << " " <<  std::endl;
                  std::cout << " " << std::endl;
                }
              }
              BOOST_ASSERT(found);

              // identify the finer mesh index
              int aa = -1;
              int bb = -1;
              int cc = -1;
              for (int ii=0;ii<par->granularity;ii++) {
                if ( floatcmp(xx,val[last_time]->x_[ii]) == 1 ) aa = ii;
                if ( floatcmp(yy,val[last_time]->y_[ii]) == 1 ) bb = ii;
                if ( floatcmp(zz,val[last_time]->z_[ii]) == 1 ) cc = ii;
                if ( aa != -1 && bb != -1 && cc != -1 ) break;
              }
              BOOST_ASSERT(aa != -1); 
              BOOST_ASSERT(bb != -1); 
              BOOST_ASSERT(cc != -1); 
              
              // restriction
              for (int ll=0;ll<num_eqns;ll++) {
                resultval->value_[i+n*(j+n*k)].phi[0][ll] = val[last_time]->value_[aa+n*(bb+n*cc)].phi[0][ll]; 
              }
            }}}
            if ( val[compute_index]->timestep_ >= par->nt0-2 ) {
              return 0;
            }
            return 1;
#endif
          }
        } else {
          compute_index = -1;
          if ( val.size() == 27 ) {
            compute_index = (val.size()-1)/2;
          } else {
            std::size_t a,b,c;
            int level = findlevel3D(row,column,a,b,c,par);
            had_double_type dx = par->dx0/pow(2.0,level);
            had_double_type x = par->min[level] + a*dx*par->granularity;
            had_double_type y = par->min[level] + b*dx*par->granularity;
            had_double_type z = par->min[level] + c*dx*par->granularity;
            compute_index = -1;
            for (int i=0;i<val.size();i++) {
              if ( floatcmp(x,val[i]->x_[0]) == 1 && 
                   floatcmp(y,val[i]->y_[0]) == 1 && 
                   floatcmp(z,val[i]->z_[0]) == 1 ) {
                compute_index = i;
                break;
              }
            }
            if ( compute_index == -1 ) {
              for (int i=0;i<val.size();i++) {
                std::cout << " DEBUG " << val[i]->x_[0] << " " << val[i]->y_[0] << " "<< val[i]->z_[0] << std::endl;
              }
              std::cout << " PROBLEM LOCATING x " << x << " y " << y << " z " << z << " val size " << val.size() << " level " << level << std::endl;
              BOOST_ASSERT(false);
            }
            boundary = true;
          } 

          std::vector<nodedata* > vecval;
          std::vector<nodedata>::iterator niter;
          // this is really a 3d array
          vecval.resize(3*par->granularity * 3*par->granularity * 3*par->granularity);

          int count_i = 0;
          int count_j = 0;
          if ( boundary ) {
            bbox[0] = 1; bbox[1] = 1; bbox[2] = 1;
            bbox[3] = 1; bbox[4] = 1; bbox[5] = 1;
          }
          for (int i=0;i<val.size();i++) {
            int ii,jj,kk;
            if ( val.size() == 27 ) {
              kk = i/9 - 1;
              jj = count_j - 1;
              ii = count_i - 1;
              count_i++;
              if ( count_i%3 == 0 ) {
                count_i = 0; 
                count_j++;
              }
              if ( count_j%3 == 0 ) count_j = 0;
            } else {
              had_double_type x = val[compute_index]->x_[0];
              had_double_type y = val[compute_index]->y_[0];
              had_double_type z = val[compute_index]->z_[0];

              bool xchk = floatcmp(x,val[i]->x_[0]);
              bool ychk = floatcmp(y,val[i]->y_[0]);
              bool zchk = floatcmp(z,val[i]->z_[0]);

              if ( xchk ) ii = 0;
              else if ( x > val[i]->x_[0] ) ii = -1;
              else if ( x < val[i]->x_[0] ) ii = 1;
              else BOOST_ASSERT(false);

              if ( ychk ) jj = 0;
              else if ( y > val[i]->y_[0] ) jj = -1;
              else if ( y < val[i]->y_[0] ) jj = 1;
              else BOOST_ASSERT(false);

              if ( zchk ) kk = 0;
              else if ( z > val[i]->z_[0] ) kk = -1;
              else if ( z < val[i]->z_[0] ) kk = 1;
              else BOOST_ASSERT(false);

              // figure out bounding box
              if ( x > val[i]->x_[0] && ychk && zchk ) {
                bbox[0] = 0;
              }

              if ( x < val[i]->x_[0] && ychk && zchk ) {
                bbox[1] = 0;
              }

              if ( xchk && y > val[i]->y_[0]  && zchk ) {
                bbox[2] = 0;
              }

              if ( xchk && y < val[i]->y_[0]  && zchk ) {
                bbox[3] = 0;
              }

              if ( xchk && ychk && z > val[i]->z_[0] ) {
                bbox[4] = 0;
              }

              if ( xchk && ychk && z < val[i]->z_[0] ) {
                bbox[5] = 0;
              }
            }

            int count = 0;
            for (niter=val[i]->value_.begin();niter!=val[i]->value_.end();++niter) {
              int tmp_index = count/par->granularity;
              int c = tmp_index/par->granularity;
              int b = tmp_index%par->granularity;
              int a = count - par->granularity*(b+c*par->granularity);

              vecval[a+(ii+1)*par->granularity 
                        + 3*par->granularity*( 
                             (b+(jj+1)*par->granularity)
                                +3*par->granularity*(c+(kk+1)*par->granularity) )] = &(*niter); 
              count++;
            }
          }

          // copy over critical info
          resultval->x_ = val[compute_index]->x_;
          resultval->y_ = val[compute_index]->y_;
          resultval->z_ = val[compute_index]->z_;
          resultval->value_.resize(val[compute_index]->value_.size());
          resultval->granularity = val[compute_index]->granularity;
          resultval->level_ = val[compute_index]->level_;

          resultval->max_index_ = val[compute_index]->max_index_;
          resultval->granularity = val[compute_index]->granularity;
          resultval->index_ = val[compute_index]->index_;

          if (val[compute_index]->timestep_ < (int)numsteps_) {

              int level = val[compute_index]->level_;

              had_double_type dt = par->dt0/pow(2.0,level);
              had_double_type dx = par->dx0/pow(2.0,level); 

              // call rk update 
              int adj_index = 0;
              int gft = rkupdate(vecval,resultval.get_ptr(),
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

