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

    had_double_type stencil::interp_linear(had_double_type y1, had_double_type y2,
                                           had_double_type x, had_double_type x1, had_double_type x2) {
      had_double_type xx1 = x - x1;
      had_double_type xx2 = x - x2;
      had_double_type result = xx2*y1/( (x1-x2) ) + xx1*y2/( (x2-x1) );
  
      return result;
    }

    void stencil::interp3d(had_double_type &x,had_double_type &y, had_double_type &z,
                                      access_memory_block<stencil_data> &val, 
                                      access_memory_block<stencil_data> &resultval,
                                      int index, Parameter const& par) {

      int ii,jj,kk;

      // set up index bounds
      for (int i=0;i<val->x_.size();i++) {
        if ( floatcmp_ge(val->x_[i],x) ) {
          ii = i;
          break;
        }         
      }
      for (int i=0;i<val->y_.size();i++) {
        if ( floatcmp_ge(val->y_[i],y) ) {
          jj = i;
          break;
        }         
      }
      for (int i=0;i<val->z_.size();i++) {
        if ( floatcmp_ge(val->z_[i],z) ) {
          kk = i;
          break;
        }         
      }

      int nx = val->x_.size();
      int ny = val->y_.size();
      int nz = val->z_.size();

      bool no_interp_x = false;
      bool no_interp_y = false;
      bool no_interp_z = false;
      if ( ii == 0 ) {
        // we may have a problem unless x doesn't need to be interpolated -- check
        BOOST_ASSERT( floatcmp(val->x_[ii],x) );
        no_interp_x = true;
      }
      if ( jj == 0 ) {
        // we may have a problem unless y doesn't need to be interpolated -- check
        BOOST_ASSERT( floatcmp(val->y_[jj],y) );
        no_interp_y = true;
      }
      if ( kk == 0 ) {
        // we may have a problem unless z doesn't need to be interpolated -- check
        BOOST_ASSERT( floatcmp(val->z_[kk],z) );
        no_interp_z = true;
      }

      if ( no_interp_x && no_interp_y && no_interp_z ) {
        // no interp needed -- this probably will never be called but is added for completeness
        for (int ll=0;ll<num_eqns;ll++) {
          resultval->pvalue_[index].phi[0][ll] = val->value_[ii+nx*(jj+ny*kk)].phi[0][ll];
        }
        return;
      }

      // Quick sanity check to be sure we have bracketed the point we wish to interpolate
      if ( !no_interp_x  && !no_interp_y && !no_interp_z ) {
        BOOST_ASSERT(floatcmp_le(val->x_[ii-1],x) && floatcmp_ge(val->x_[ii],x) );
        BOOST_ASSERT(floatcmp_le(val->y_[jj-1],y) && floatcmp_ge(val->y_[jj],y) );
        BOOST_ASSERT(floatcmp_le(val->z_[kk-1],z) && floatcmp_ge(val->z_[kk],z) );
      }

      had_double_type tmp2[2][2][num_eqns];
      had_double_type tmp3[2][num_eqns];

      // interpolate in x {{{
      if ( !no_interp_x && !no_interp_y && !no_interp_z ) {
        for (int k=kk-1;k<kk+1;k++) {
          for (int j=jj-1;j<jj+1;j++) {
            for (int ll=0;ll<num_eqns;ll++) {
              tmp2[j-(jj-1)][k-(kk-1)][ll] = interp_linear(val->value_[ii-1+nx*(j+ny*k)].phi[0][ll],
                                                   val->value_[ii  +nx*(j+ny*k)].phi[0][ll],
                                                   x,
                                                   val->x_[ii-1],val->x_[ii]);
            }
          }
        }
      } else if ( no_interp_x && !no_interp_y && !no_interp_z ) {
        for (int k=kk-1;k<kk+1;k++) {
          for (int j=jj-1;j<jj+1;j++) {
            for (int ll=0;ll<num_eqns;ll++) {
              tmp2[j-(jj-1)][k-(kk-1)][ll] = val->value_[ii+nx*(j+ny*k)].phi[0][ll];
            }
          }
        }
      } else if ( !no_interp_x && no_interp_y && !no_interp_z ) {
        for (int k=kk-1;k<kk+1;k++) {
          for (int ll=0;ll<num_eqns;ll++) {
            tmp2[0][k-(kk-1)][ll] = interp_linear(val->value_[ii-1+nx*(jj+ny*k)].phi[0][ll],
                                              val->value_[ii  +nx*(jj+ny*k)].phi[0][ll],
                                              x,
                                              val->x_[ii-1],val->x_[ii]);
          }
        }
      } else if ( !no_interp_x && !no_interp_y && no_interp_z ) {
        for (int j=jj-1;j<jj+1;j++) {
          for (int ll=0;ll<num_eqns;ll++) {
            tmp2[j-(jj-1)][0][ll] = interp_linear(val->value_[ii-1+nx*(j+ny*kk)].phi[0][ll],
                                              val->value_[ii  +nx*(j+ny*kk)].phi[0][ll],
                                              x,
                                              val->x_[ii-1],val->x_[ii]);
          }
        }
      } else if ( no_interp_x && no_interp_y && !no_interp_z ) {
        for (int k=kk-1;k<kk+1;k++) {
          for (int ll=0;ll<num_eqns;ll++) {
            tmp2[0][k-(kk-1)][ll] = val->value_[ii+nx*(jj+ny*k)].phi[0][ll];
          }
        }
      } else if ( no_interp_x && !no_interp_y && no_interp_z ) {
        for (int j=jj-1;j<jj+1;j++) {
          for (int ll=0;ll<num_eqns;ll++) {
            tmp2[j-(jj-1)][0][ll] = val->value_[ii+nx*(j+ny*kk)].phi[0][ll];
          }
        }
      } else if ( !no_interp_x && no_interp_y && no_interp_z ) {
        for (int ll=0;ll<num_eqns;ll++) {
          resultval->pvalue_[index].phi[0][ll] = interp_linear(val->value_[ii-1+nx*(jj+ny*kk)].phi[0][ll],
                                                              val->value_[ii  +nx*(jj+ny*kk)].phi[0][ll],
                                                              x,
                                                              val->x_[ii-1],val->x_[ii]);
        }
        return;
      } else {
        BOOST_ASSERT(false);
      }
      // }}}

      // interpolate in y {{{
      if ( !no_interp_y && !no_interp_z ) {
        for (int k=0;k<2;k++) {
          for (int ll=0;ll<num_eqns;ll++) {
            tmp3[k][ll] = interp_linear(tmp2[0][k][ll],tmp2[1][k][ll],y,
                                         val->y_[jj-1],val->y_[jj]);
          }
        }
      } else if ( no_interp_y && !no_interp_z ) {
        for (int k=0;k<2;k++) {
          for (int ll=0;ll<num_eqns;ll++) {
            tmp3[k][ll] = tmp2[0][k][ll];
          }
        }
      } else if ( !no_interp_y && no_interp_z ) {
        for (int ll=0;ll<num_eqns;ll++) {
          resultval->pvalue_[index].phi[0][ll] = interp_linear(tmp2[0][0][ll],tmp2[1][0][ll],y,
                                                              val->y_[jj-1],val->y_[jj]);
        }
        return;
      } else {
        BOOST_ASSERT(false);
      }
      // }}}

      // interpolate in z {{{
      if ( !no_interp_z ) {
        for (int ll=0;ll<num_eqns;ll++) {
          resultval->pvalue_[index].phi[0][ll] = interp_linear(tmp3[0][ll],tmp3[1][ll],
                                                              z,
                                                              val->z_[kk-1],val->z_[kk]);
        } 
        return;
      } else {
        BOOST_ASSERT(false);
      }
      // }}}

      return;
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
        int pbox[6] = {0,0,0,0,0,0};   // initialize bounding box

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

            // TEST
            resultval.get() = val[compute_index].get();

            // copy over critical info
            resultval->x_ = val[compute_index]->x_;
            resultval->y_ = val[compute_index]->y_;
            resultval->z_ = val[compute_index]->z_;
            resultval->value_.resize(val[compute_index]->value_.size());
            resultval->granularity = val[compute_index]->granularity;
            resultval->level_ = val[compute_index]->level_;
            resultval->max_index_ = val[compute_index]->max_index_;
            resultval->index_ = val[compute_index]->index_;

            // We may be dealing with either restriction or prolongation (both are performed at the same time)
            bool restriction = false;
            bool prolongation = false;
            for (int i=0;i<val.size();i++) {
              if ( resultval->level_ < val[i]->level_ ) restriction = true;
              if ( resultval->level_ > val[i]->level_ ) prolongation = true;
              if ( restriction && prolongation ) break;
            }

            if ( prolongation ) {
              // prolongation {{{
              // interpolation
              had_double_type xmin = val[compute_index]->x_[0];
              had_double_type xmax = val[compute_index]->x_[par->granularity-1];
              had_double_type ymin = val[compute_index]->y_[0];
              had_double_type ymax = val[compute_index]->y_[par->granularity-1];
              had_double_type zmin = val[compute_index]->z_[0];
              had_double_type zmax = val[compute_index]->z_[par->granularity-1];

              if ( floatcmp(xmin,par->min[level]) == 1 ) {
                pbox[0] = 1;
              }
              if ( floatcmp(xmax,par->max[level]) == 1 ) {
                pbox[1] = 1;
              }
              if ( floatcmp(ymin,par->min[level]) == 1 ) {
                pbox[2] = 1;
              }
              if ( floatcmp(ymax,par->max[level]) == 1 ) {
                pbox[3] = 1;
              }
              if ( floatcmp(zmin,par->min[level]) == 1 ) {
                pbox[4] = 1;
              }
              if ( floatcmp(zmax,par->max[level]) == 1 ) {
                pbox[5] = 1;
              }

              if ( pbox[0] == 0 && pbox[1] == 0 &&
                   pbox[2] == 0 && pbox[3] == 0 &&
                   pbox[4] == 0 && pbox[5] == 0 ) {
                // no prolongation needed in this case.  Something somewhere has gone wrong
                BOOST_ASSERT(false);
              }

              // an alternative 3D vector with all the prolongation data in the right place
              if ( (pbox[0] == 1 && pbox[1] == 1) || (pbox[2] == 1 && pbox[3] == 1) || (pbox[4] == 1 && pbox[5] == 1) ) {
                // this shouldn't happen
                BOOST_ASSERT(false);
              }

              // 27 cases {{{
              if (        pbox[0] == 1 && pbox[1] == 0 && pbox[2] == 0 && pbox[3] == 0 && pbox[4] == 0 && pbox[5] == 0  ) {
                // {{{
                resultval->pvalue_.resize(2*par->granularity * par->granularity * par->granularity);
                resultval->px_.resize(2*par->granularity);
                resultval->py_.resize(par->granularity);
                resultval->pz_.resize(par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->px_[i+par->granularity] = xmin + i*dx;
                  if (i >= 0) {
                    resultval->py_[i] = ymin + i*dx;
                    resultval->pz_[i] = zmin + i*dx;
                  }
                }
                // }}}
              } else if ( pbox[0] == 0 && pbox[1] == 1 && pbox[2] == 0 && pbox[3] == 0 && pbox[4] == 0 && pbox[5] == 0  ) {
                // {{{
                resultval->pvalue_.resize(2*par->granularity * par->granularity * par->granularity);
                resultval->px_.resize(2*par->granularity);
                resultval->py_.resize(par->granularity);
                resultval->pz_.resize(par->granularity);
                for (int i=0;i<2*par->granularity;i++) {
                  resultval->px_[i] = xmin + i*dx;
                  if ( i < par->granularity ) {
                    resultval->py_[i] = ymin + i*dx;
                    resultval->pz_[i] = zmin + i*dx;
                  }
                }
                // }}}
              } else if ( pbox[0] == 0 && pbox[1] == 0 && pbox[2] == 1 && pbox[3] == 0 && pbox[4] == 0 && pbox[5] == 0  ) {
                // {{{
                resultval->pvalue_.resize(par->granularity * 2*par->granularity * par->granularity);
                resultval->px_.resize(par->granularity);
                resultval->py_.resize(2*par->granularity);
                resultval->pz_.resize(par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->py_[i+par->granularity] = ymin + i*dx;
                  if (i >= 0) {
                    resultval->px_[i] = xmin + i*dx;
                    resultval->pz_[i] = zmin + i*dx;
                  }
                }
                // }}}
              } else if ( pbox[0] == 0 && pbox[1] == 0 && pbox[2] == 0 && pbox[3] == 1 && pbox[4] == 0 && pbox[5] == 0  ) {
                // {{{
                resultval->pvalue_.resize(par->granularity * 2*par->granularity * par->granularity);
                resultval->px_.resize(par->granularity);
                resultval->py_.resize(2*par->granularity);
                resultval->pz_.resize(par->granularity);
                for (int i=0;i<2*par->granularity;i++) {
                  resultval->py_[i] = ymin + i*dx;
                  if ( i < par->granularity ) {
                    resultval->px_[i] = xmin + i*dx;
                    resultval->pz_[i] = zmin + i*dx;
                  }
                }
                // }}}
              } else if ( pbox[0] == 0 && pbox[1] == 0 && pbox[2] == 0 && pbox[3] == 0 && pbox[4] == 1 && pbox[5] == 0  ) {
                // {{{
                resultval->pvalue_.resize(par->granularity * par->granularity * 2*par->granularity);
                resultval->px_.resize(par->granularity);
                resultval->py_.resize(par->granularity);
                resultval->pz_.resize(2*par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->pz_[i+par->granularity] = zmin + i*dx;
                  if (i >= 0) {
                    resultval->px_[i] = xmin + i*dx;
                    resultval->py_[i] = ymin + i*dx;
                  }
                }
                // }}}
              } else if ( pbox[0] == 0 && pbox[1] == 0 && pbox[2] == 0 && pbox[3] == 0 && pbox[4] == 0 && pbox[5] == 1  ) {
                // {{{
                resultval->pvalue_.resize(par->granularity * par->granularity * 2*par->granularity);
                resultval->px_.resize(par->granularity);
                resultval->py_.resize(par->granularity);
                resultval->pz_.resize(2*par->granularity);
                for (int i=0;i<2*par->granularity;i++) {
                  resultval->pz_[i] = zmin + i*dx;
                  if ( i < par->granularity ) {
                    resultval->px_[i] = xmin + i*dx;
                    resultval->py_[i] = ymin + i*dx;
                  }
                }
                // }}}
//
//
//
              } else if ( pbox[0] == 1 && pbox[1] == 0 && pbox[2] == 1 && pbox[3] == 0 && pbox[4] == 0 && pbox[5] == 0  ) {
                // {{{
                resultval->pvalue_.resize(2*par->granularity * 2*par->granularity * par->granularity);
                resultval->px_.resize(2*par->granularity);
                resultval->py_.resize(2*par->granularity);
                resultval->pz_.resize(par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->px_[i+par->granularity] = xmin + i*dx;
                  resultval->py_[i+par->granularity] = ymin + i*dx;
                  if (i >= 0) {
                    resultval->pz_[i] = zmin + i*dx;
                  }
                }
                // }}}
              } else if ( pbox[0] == 1 && pbox[1] == 0 && pbox[2] == 0 && pbox[3] == 1 && pbox[4] == 0 && pbox[5] == 0  ) {
                // {{{
                resultval->pvalue_.resize(2*par->granularity * 2*par->granularity * par->granularity);
                resultval->px_.resize(2*par->granularity);
                resultval->py_.resize(2*par->granularity);
                resultval->pz_.resize(par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->px_[i+par->granularity] = xmin + i*dx;
                  resultval->py_[i+par->granularity] = ymin + (i+par->granularity)*dx;
                  if (i >= 0) {
                    resultval->pz_[i] = zmin + i*dx;
                  }
                }
                // }}}
              } else if ( pbox[0] == 1 && pbox[1] == 0 && pbox[2] == 0 && pbox[3] == 0 && pbox[4] == 1 && pbox[5] == 0  ) {
                // {{{
                resultval->pvalue_.resize(2*par->granularity * par->granularity * 2*par->granularity);
                resultval->px_.resize(2*par->granularity);
                resultval->py_.resize(par->granularity);
                resultval->pz_.resize(2*par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->px_[i+par->granularity] = xmin + i*dx;
                  resultval->pz_[i+par->granularity] = zmin + i*dx;
                  if (i >= 0) {
                    resultval->py_[i] = ymin + i*dx;
                  }
                }
                // }}}
              } else if ( pbox[0] == 1 && pbox[1] == 0 && pbox[2] == 0 && pbox[3] == 0 && pbox[4] == 0 && pbox[5] == 1  ) {
                // {{{
                resultval->pvalue_.resize(2*par->granularity * par->granularity * 2*par->granularity);
                resultval->px_.resize(2*par->granularity);
                resultval->py_.resize(par->granularity);
                resultval->pz_.resize(2*par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->px_[i+par->granularity] = xmin + i*dx;
                  resultval->pz_[i+par->granularity] = zmin + (i+par->granularity)*dx;
                  if (i >= 0) {
                    resultval->py_[i] = ymin + i*dx;
                  }
                }
                // }}}
//
//
//
              } else if ( pbox[0] == 0 && pbox[1] == 1 && pbox[2] == 1 && pbox[3] == 0 && pbox[4] == 0 && pbox[5] == 0  ) {
                // {{{
                resultval->pvalue_.resize(2*par->granularity * 2*par->granularity * par->granularity);
                resultval->px_.resize(2*par->granularity);
                resultval->py_.resize(2*par->granularity);
                resultval->pz_.resize(par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->px_[i+par->granularity] = xmin + (i+par->granularity)*dx;
                  resultval->py_[i+par->granularity] = ymin + i*dx;
                  if (i >= 0) {
                    resultval->pz_[i] = zmin + i*dx;
                  }
                }
                // }}}
              } else if ( pbox[0] == 0 && pbox[1] == 1 && pbox[2] == 0 && pbox[3] == 1 && pbox[4] == 0 && pbox[5] == 0  ) {
                // {{{
                resultval->pvalue_.resize(2*par->granularity * 2*par->granularity * par->granularity);
                resultval->px_.resize(2*par->granularity);
                resultval->py_.resize(2*par->granularity);
                resultval->pz_.resize(par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->px_[i+par->granularity] = xmin + (i+par->granularity)*dx;
                  resultval->py_[i+par->granularity] = ymin + (i+par->granularity)*dx;
                  if (i >= 0) {
                    resultval->pz_[i] = zmin + i*dx;
                  }
                }
                // }}}
              } else if ( pbox[0] == 0 && pbox[1] == 1 && pbox[2] == 0 && pbox[3] == 0 && pbox[4] == 1 && pbox[5] == 0  ) {
                // {{{
                resultval->pvalue_.resize(2*par->granularity * par->granularity * 2*par->granularity);
                resultval->px_.resize(2*par->granularity);
                resultval->py_.resize(par->granularity);
                resultval->pz_.resize(2*par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->px_[i+par->granularity] = xmin + (i+par->granularity)*dx;
                  resultval->pz_[i+par->granularity] = zmin + i*dx;
                  if (i >= 0) {
                    resultval->py_[i] = ymin + i*dx;
                  }
                }
                // }}}
              } else if ( pbox[0] == 0 && pbox[1] == 1 && pbox[2] == 0 && pbox[3] == 0 && pbox[4] == 0 && pbox[5] == 1  ) {
                // {{{
                resultval->pvalue_.resize(2*par->granularity * par->granularity * 2*par->granularity);
                resultval->px_.resize(2*par->granularity);
                resultval->py_.resize(par->granularity);
                resultval->pz_.resize(2*par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->px_[i+par->granularity] = xmin + (i+par->granularity)*dx;
                  resultval->pz_[i+par->granularity] = zmin + (i+par->granularity)*dx;
                  if (i >= 0) {
                    resultval->py_[i] = ymin + i*dx;
                  }
                }
                // }}}
//
//
//
              } else if ( pbox[0] == 0 && pbox[1] == 0 && pbox[2] == 1 && pbox[3] == 0 && pbox[4] == 1 && pbox[5] == 0  ) {
                // {{{
                resultval->pvalue_.resize(par->granularity * 2*par->granularity * 2*par->granularity);
                resultval->px_.resize(par->granularity);
                resultval->py_.resize(2*par->granularity);
                resultval->pz_.resize(2*par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->py_[i+par->granularity] = ymin + i*dx;
                  resultval->pz_[i+par->granularity] = zmin + i*dx;
                  if (i >= 0) {
                    resultval->px_[i] = xmin + i*dx;
                  }
                }
                // }}}
              } else if ( pbox[0] == 0 && pbox[1] == 0 && pbox[2] == 1 && pbox[3] == 0 && pbox[4] == 0 && pbox[5] == 1  ) {
                // {{{
                resultval->pvalue_.resize(par->granularity * 2*par->granularity * 2*par->granularity);
                resultval->px_.resize(par->granularity);
                resultval->py_.resize(2*par->granularity);
                resultval->pz_.resize(2*par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->py_[i+par->granularity] = ymin + i*dx;
                  resultval->pz_[i+par->granularity] = zmin + (i+par->granularity)*dx;
                  if (i >= 0) {
                    resultval->px_[i] = xmin + i*dx;
                  }
                }
                // }}}
              } else if ( pbox[0] == 0 && pbox[1] == 0 && pbox[2] == 0 && pbox[3] == 1 && pbox[4] == 1 && pbox[5] == 0  ) {
                // {{{
                resultval->pvalue_.resize(par->granularity * 2*par->granularity * 2*par->granularity);
                resultval->px_.resize(par->granularity);
                resultval->py_.resize(2*par->granularity);
                resultval->pz_.resize(2*par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->py_[i+par->granularity] = ymin + (i+par->granularity)*dx;
                  resultval->pz_[i+par->granularity] = zmin + i*dx;
                  if (i >= 0) {
                    resultval->px_[i] = xmin + i*dx;
                  }
                }
                // }}}
              } else if ( pbox[0] == 0 && pbox[1] == 0 && pbox[2] == 0 && pbox[3] == 1 && pbox[4] == 0 && pbox[5] == 1  ) {
                // {{{
                resultval->pvalue_.resize(par->granularity * 2*par->granularity * 2*par->granularity);
                resultval->px_.resize(par->granularity);
                resultval->py_.resize(2*par->granularity);
                resultval->pz_.resize(2*par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->py_[i+par->granularity] = ymin + (i+par->granularity)*dx;
                  resultval->pz_[i+par->granularity] = zmin + (i+par->granularity)*dx;
                  if (i >= 0) {
                    resultval->px_[i] = xmin + i*dx;
                  }
                }
                // }}}
//
//
//
              } else if ( pbox[0] == 1 && pbox[1] == 0 && pbox[2] == 1 && pbox[3] == 0 && pbox[4] == 1 && pbox[5] == 0  ) {
                // {{{
                resultval->pvalue_.resize(2*par->granularity * 2*par->granularity * 2*par->granularity);
                resultval->px_.resize(2*par->granularity);
                resultval->py_.resize(2*par->granularity);
                resultval->pz_.resize(2*par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->px_[i+par->granularity] = xmin + i*dx;
                  resultval->py_[i+par->granularity] = ymin + i*dx;
                  resultval->pz_[i+par->granularity] = zmin + i*dx;
                }
                // }}}
              } else if ( pbox[0] == 1 && pbox[1] == 0 && pbox[2] == 0 && pbox[3] == 1 && pbox[4] == 1 && pbox[5] == 0  ) {
                // {{{
                resultval->pvalue_.resize(2*par->granularity * 2*par->granularity * 2*par->granularity);
                resultval->px_.resize(2*par->granularity);
                resultval->py_.resize(2*par->granularity);
                resultval->pz_.resize(2*par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->px_[i+par->granularity] = xmin + i*dx;
                  resultval->py_[i+par->granularity] = ymin + (i+par->granularity)*dx;
                  resultval->pz_[i+par->granularity] = zmin + i*dx;
                }
                // }}}
              } else if ( pbox[0] == 1 && pbox[1] == 0 && pbox[2] == 1 && pbox[3] == 0 && pbox[4] == 0 && pbox[5] == 1  ) {
                // {{{
                resultval->pvalue_.resize(2*par->granularity * 2*par->granularity * 2*par->granularity);
                resultval->px_.resize(2*par->granularity);
                resultval->py_.resize(2*par->granularity);
                resultval->pz_.resize(2*par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->px_[i+par->granularity] = xmin + i*dx;
                  resultval->py_[i+par->granularity] = ymin + i*dx;
                  resultval->pz_[i+par->granularity] = zmin + (i+par->granularity)*dx;
                }
                // }}}
              } else if ( pbox[0] == 1 && pbox[1] == 0 && pbox[2] == 0 && pbox[3] == 1 && pbox[4] == 0 && pbox[5] == 1  ) {
                // {{{
                resultval->pvalue_.resize(2*par->granularity * 2*par->granularity * 2*par->granularity);
                resultval->px_.resize(2*par->granularity);
                resultval->py_.resize(2*par->granularity);
                resultval->pz_.resize(2*par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->px_[i+par->granularity] = xmin + i*dx;
                  resultval->py_[i+par->granularity] = ymin + (i+par->granularity)*dx;
                  resultval->pz_[i+par->granularity] = zmin + (i+par->granularity)*dx;
                }
                // }}}
//
//
//
              } else if ( pbox[0] == 0 && pbox[1] == 1 && pbox[2] == 1 && pbox[3] == 0 && pbox[4] == 1 && pbox[5] == 0  ) {
                // {{{
                resultval->pvalue_.resize(2*par->granularity * 2*par->granularity * 2*par->granularity);
                resultval->px_.resize(2*par->granularity);
                resultval->py_.resize(2*par->granularity);
                resultval->pz_.resize(2*par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->px_[i+par->granularity] = xmin + (i+par->granularity)*dx;
                  resultval->py_[i+par->granularity] = ymin + i*dx;
                  resultval->pz_[i+par->granularity] = zmin + i*dx;
                }
                // }}}
              } else if ( pbox[0] == 0 && pbox[1] == 1 && pbox[2] == 0 && pbox[3] == 1 && pbox[4] == 1 && pbox[5] == 0  ) {
                // {{{
                resultval->pvalue_.resize(2*par->granularity * 2*par->granularity * 2*par->granularity);
                resultval->px_.resize(2*par->granularity);
                resultval->py_.resize(2*par->granularity);
                resultval->pz_.resize(2*par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->px_[i+par->granularity] = xmin + (i+par->granularity)*dx;
                  resultval->py_[i+par->granularity] = ymin + (i+par->granularity)*dx;
                  resultval->pz_[i+par->granularity] = zmin + i*dx;
                }
                // }}}
              } else if ( pbox[0] == 0 && pbox[1] == 1 && pbox[2] == 1 && pbox[3] == 0 && pbox[4] == 0 && pbox[5] == 1  ) {
                // {{{
                resultval->pvalue_.resize(2*par->granularity * 2*par->granularity * 2*par->granularity);
                resultval->px_.resize(2*par->granularity);
                resultval->py_.resize(2*par->granularity);
                resultval->pz_.resize(2*par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->px_[i+par->granularity] = xmin + (i+par->granularity)*dx;
                  resultval->py_[i+par->granularity] = ymin + i*dx;
                  resultval->pz_[i+par->granularity] = zmin + (i+par->granularity)*dx;
                }
                // }}}
              } else if ( pbox[0] == 0 && pbox[1] == 1 && pbox[2] == 0 && pbox[3] == 1 && pbox[4] == 0 && pbox[5] == 1  ) {
                // {{{
                resultval->pvalue_.resize(2*par->granularity * 2*par->granularity * 2*par->granularity);
                resultval->px_.resize(2*par->granularity);
                resultval->py_.resize(2*par->granularity);
                resultval->pz_.resize(2*par->granularity);
                for (int i=-par->granularity;i<par->granularity;i++) {
                  resultval->px_[i+par->granularity] = xmin + (i+par->granularity)*dx;
                  resultval->py_[i+par->granularity] = ymin + (i+par->granularity)*dx;
                  resultval->pz_[i+par->granularity] = zmin + (i+par->granularity)*dx;
                }
                // }}}
              } else {
                std::cout << " PROBLEM " << pbox[0] << " " << pbox[1] << std::endl;
                std::cout << "         " << pbox[2] << " " << pbox[3] << std::endl;
                std::cout << "         " << pbox[4] << " " << pbox[5] << std::endl;
                BOOST_ASSERT(false);
              }
              // }}}

              // interpolate
              int nx = resultval->px_.size();
              int ny = resultval->py_.size();
              int nz = resultval->pz_.size();
              int n = par->granularity;

              // get the bounding box of the input
              had_double_type vxmin,vymin,vzmin,vxmax,vymax,vzmax; 
              vxmin = 9999;
              vymin = 9999;
              vzmin = 9999;
              vxmax = -9999;
              vymax = -9999;
              vzmax = -9999;
              for (int ii=0;ii<val.size();ii++) {
                if ( val[ii]->x_[0] < vxmin ) vxmin = val[ii]->x_[0]; 
                if ( val[ii]->y_[0] < vymin ) vymin = val[ii]->y_[0]; 
                if ( val[ii]->z_[0] < vzmin ) vzmin = val[ii]->z_[0]; 
                if ( val[ii]->x_[par->granularity-1] > vxmax ) vxmax = val[ii]->x_[par->granularity-1];
                if ( val[ii]->y_[par->granularity-1] > vymax ) vymax = val[ii]->y_[par->granularity-1];
                if ( val[ii]->z_[par->granularity-1] > vzmax ) vzmax = val[ii]->z_[par->granularity-1];
              }

              // sanity check
              if ( vxmin > resultval->px_[0] || vymin > resultval->py_[0] || vzmin > resultval->pz_[0] ||
                   vxmax < resultval->px_[nx-1] || vymax < resultval->py_[ny-1] || vzmax < resultval->pz_[nz-1] ) {
                std::cout << " PROBLEM COARSE INPUT      x " << vxmin << " " << vxmax << std::endl;
                std::cout << "                           y " << vymin << " " << vymax << std::endl;
                std::cout << "                           z " << vzmin << " " << vzmax << std::endl;
                std::cout << " PROBLEM FINE MESH FILL    x " << resultval->px_[0]  << " " << resultval->px_[nx-1] << std::endl;
                std::cout << "                           y " << resultval->py_[0]  << " " << resultval->py_[ny-1] << std::endl;
                std::cout << "                           z " << resultval->pz_[0]  << " " << resultval->pz_[nz-1] << std::endl;
                std::cout << " " << std::endl;
                BOOST_ASSERT(false);
              }

              for (int k=0;k<nz;k++) {
                had_double_type zt =  resultval->pz_[k];              
              for (int j=0;j<ny;j++) {
                had_double_type yt =  resultval->py_[j];              
              for (int i=0;i<nx;i++) {
                had_double_type xt =  resultval->px_[i];              
 
                bool found = false;
                // check compute_index first to see if this point is contained in it
                if ( floatcmp_ge(xt,xmin) && floatcmp_le(xt,xmax) &&
                     floatcmp_ge(yt,ymin) && floatcmp_le(yt,ymax) &&
                     floatcmp_ge(zt,zmin) && floatcmp_le(zt,zmax) ) {
                  // no interp need -- just grap the point  
                  // figure out the index to grab
                  had_double_type daa =  (xt-xmin)/dx + 0.5;
                  had_double_type dbb =  (yt-ymin)/dx + 0.5;
                  had_double_type dcc =  (zt-zmin)/dx + 0.5;
                  int aa = (int) floor(daa); 
                  int bb = (int) floor(dbb); 
                  int cc = (int) floor(dcc); 
                  BOOST_ASSERT( floatcmp(xt,val[compute_index]->x_[aa] ) );
                  BOOST_ASSERT( floatcmp(yt,val[compute_index]->y_[bb] ) );
                  BOOST_ASSERT( floatcmp(zt,val[compute_index]->z_[cc] ) );
                  for (int ll=0;ll<num_eqns;ll++) {
                    resultval->pvalue_[i+nx*(j+ny*k)].phi[0][ll] = val[compute_index]->value_[aa+n*(bb+n*cc)].phi[0][ll]; 
                  }
                } else {
                  // check the other input
                  found = false; 
                  for (int ii=0;ii<val.size();ii++) {
                    if ( ii != compute_index ) {
                      if ( floatcmp_ge(xt,val[ii]->x_[0])  && floatcmp_le(xt,val[ii]->x_[par->granularity-1]) &&
                           floatcmp_ge(yt,val[ii]->y_[0])  && floatcmp_le(yt,val[ii]->y_[par->granularity-1]) &&
                           floatcmp_ge(zt,val[ii]->z_[0])  && floatcmp_le(zt,val[ii]->z_[par->granularity-1]) ) {
                        found = true;
                        // interpolate
                        interp3d(xt,yt,zt,val[ii],resultval,i+nx*(j+ny*k),par); 
                        break;
                      }
                    }
                  }

                  int anchorlist[27];
                  int anchor_index[27];
                  int has_corner[27] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
                  int interp_options[30] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
                  int counter = 0;
                  if ( !found ) {
                    // find the interpolating the anchors needed
                    for (int lk=-1;lk<2;lk=lk+2) {
                      had_double_type zn = zt + lk*dx;
                    for (int lj=-1;lj<2;lj=lj+2) {
                      had_double_type yn = yt + lj*dx;
                    for (int li=-1;li<2;li=li+2) {
                      had_double_type xn = xt + li*dx;
                      for (int ii=0;ii<val.size();ii++) {
                        if ( floatcmp_ge(xn,val[ii]->x_[0])  && floatcmp_le(xn,val[ii]->x_[par->granularity-1]) &&
                             floatcmp_ge(yn,val[ii]->y_[0])  && floatcmp_le(yn,val[ii]->y_[par->granularity-1]) &&
                             floatcmp_ge(zn,val[ii]->z_[0])  && floatcmp_le(zn,val[ii]->z_[par->granularity-1]) ) {
                          anchorlist[counter] = ii;
                          if (li == -1 && lj == -1 && lk == -1 ) {
                            has_corner[0] = 1;
                            anchor_index[0] = counter;
                          }
                          if (li ==  1 && lj == -1 && lk == -1 ) {
                            has_corner[1] = 1;
                            anchor_index[1] = counter;
                          }
                          if (li ==  1 && lj ==  1 && lk == -1 ) {
                            has_corner[2] = 1;
                            anchor_index[2] = counter;
                          }
                          if (li == -1 && lj ==  1 && lk == -1 ) {
                            has_corner[3] = 1;
                            anchor_index[3] = counter;
                          }
                          if (li == -1 && lj == -1 && lk ==  1 ) {
                            has_corner[4] = 1;
                            anchor_index[4] = counter;
                          }
                          if (li ==  1 && lj == -1 && lk ==  1 ) {
                            has_corner[5] = 1;
                            anchor_index[5] = counter;
                          }
                          if (li ==  1 && lj ==  1 && lk ==  1 ) {
                            has_corner[6] = 1;
                            anchor_index[6] = counter;
                          }
                          if (li == -1 && lj ==  1 && lk ==  1 ) {
                            has_corner[7] = 1;
                            anchor_index[7] = counter;
                          }
                          counter++;
                        }
                      }
                    } } }

                    if ( counter == 8 ) {
                      // all interpolation anchors found -- interpolate
                      found = true;
                    } else { 
                      for (int lk=-1;lk<2;lk++) {
                        had_double_type zn = zt + lk*dx;
                      for (int lj=-1;lj<2;lj++) {
                        had_double_type yn = yt + lj*dx;
                      for (int li=-1;li<2;li++) {
                        had_double_type xn = xt + li*dx;
                        // let's not duplicate what we already know
                        if ( li == 0 || lj == 0 || lk == 0 ) {
                          for (int ii=0;ii<val.size();ii++) {
                            if ( floatcmp_ge(xn,val[ii]->x_[0])  && floatcmp_le(xn,val[ii]->x_[par->granularity-1]) &&
                                 floatcmp_ge(yn,val[ii]->y_[0])  && floatcmp_le(yn,val[ii]->y_[par->granularity-1]) &&
                                 floatcmp_ge(zn,val[ii]->z_[0])  && floatcmp_le(zn,val[ii]->z_[par->granularity-1]) ) {
                              anchorlist[counter] = ii;
                              if (li == -1 && lj == -1 && lk == 0 ) {
                                has_corner[8]   = 1;
                                anchor_index[8] = counter;
                              }
                              if (li ==  1 && lj == -1 && lk == 0 ) {
                                has_corner[9]  = 1;
                                anchor_index[9] = counter;
                              }
                              if (li ==  1 && lj ==  1 && lk == 0 ) {
                                has_corner[10] = 1;
                                anchor_index[10] = counter;
                              }
                              if (li == -1 && lj ==  1 && lk == 0 ) {
                                has_corner[11] = 1;
                                anchor_index[11] = counter;
                              }
                          
                              if (li == 0 && lj == -1 && lk == -1 ) {
                                has_corner[12] = 1;
                                anchor_index[12] = counter;
                              }
                              if (li == 1 && lj == 0 && lk == -1 ) {
                                has_corner[13] = 1;
                                anchor_index[13] = counter;
                              }
                              if (li == 0 && lj == 1 && lk == -1 ) {
                                has_corner[14] = 1;
                                anchor_index[14] = counter;
                              }
                              if (li == -1 && lj == 0 && lk == -1 ) {
                                has_corner[15] = 1;
                                anchor_index[15] = counter;
                              }

                              if (li == 0 && lj == -1 && lk == 0 ) {
                                has_corner[16] = 1;
                                anchor_index[16] = counter;
                              }
                              if (li == 1 && lj == 0 && lk ==  0 ) {
                                has_corner[17] = 1;
                                anchor_index[17] = counter;
                              }
                              if (li == 0 && lj == 1 && lk ==  0 ) {
                                has_corner[18] = 1;
                                anchor_index[18] = counter;
                              }
                              if (li == -1 && lj == 0 && lk == 0 ) {
                                has_corner[19] = 1;
                                anchor_index[19] = counter;
                              }

                              if (li == 0 && lj == -1 && lk == 1 ) {
                                has_corner[20] = 1;
                                anchor_index[20] = counter;
                              }
                              if (li == 1 && lj == 0 && lk ==  1 ) {
                                has_corner[21] = 1;
                                anchor_index[21] = counter;
                              }
                              if (li == 0 && lj == 1 && lk ==  1 ) {
                                has_corner[22] = 1;
                                anchor_index[22] = counter;
                              }
                              if (li == -1 && lj == 0 && lk == 1 ) {
                                has_corner[23] = 1;
                                anchor_index[23] = counter;
                              }

                              if (li == 0 && lj == 0 && lk == 1 ) {
                                has_corner[24] = 1;
                                anchor_index[24] = counter;
                              }

                              if (li == 0 && lj == 0 && lk == -1 ) {
                                has_corner[25] = 1;
                                anchor_index[25] = counter;
                              }

                              counter++;
                            }
                          }
                        }
                      } } } 


                      // Now we have the complete picture.  Determine what the interpolation options are and proceed.  
                      if ( has_corner[0] != 1 ) {
                        interp_options[4] = -1;
                        interp_options[2] = -1;
                        interp_options[0] = -1;
                        interp_options[14] = -1;
                        interp_options[17] = -1;
                        interp_options[25] = -1;
                        interp_options[22] = -1;
                      }

                    }

                  } // end if found

                //  if ( !found ) {
                //     std::cout << " PROBLEM: point " << xt << " " << yt << " " << zt << " not found in prolongation." << std::endl;
                //     std::cout << " Available data: " << std::endl;
                //     for (int ii=0;ii<val.size();ii++) {
                //       std::cout << val[ii]->x_[0] << " " << val[ii]->x_[par->granularity-1] << std::endl;
                //       std::cout << val[ii]->y_[0] << " " << val[ii]->y_[par->granularity-1] << std::endl;
                //       std::cout << val[ii]->z_[0] << " " << val[ii]->z_[par->granularity-1] << std::endl;
                //       std::cout << " " << std::endl;
                //     }
                //     BOOST_ASSERT(false);
                //  }
                }
              } } }
              // }}}
            }

            if ( restriction ) {
              // restriction {{{
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

                //if ( last_time != -1 ) {
                  // this might save some time -- see if the point is here
                //  if ( xx >= val[last_time]->x_[0] && xx <= val[last_time]->x_[par->granularity-1] &&
                //       yy >= val[last_time]->y_[0] && yy <= val[last_time]->y_[par->granularity-1] &&
                //       zz >= val[last_time]->z_[0] && zz <= val[last_time]->z_[par->granularity-1] &&
                //       val[last_time]->level_ >= resultval->level_) {
                //    found = true;
                //  }
                //}

                int highest_level = resultval->level_;
                if ( !found ) {
                  // find the highest level who has this point
                  highest_level -= 1;
                  for (int ii=0;ii<val.size();ii++) {
                    if ( (xx >= val[ii]->x_[0] || floatcmp(xx,val[ii]->x_[0])==1) && 
                         (xx <= val[ii]->x_[par->granularity-1] || floatcmp(xx,val[ii]->x_[par->granularity-1])==1) &&
                         (yy >= val[ii]->y_[0] || floatcmp(yy,val[ii]->y_[0])==1)  && 
                         (yy <= val[ii]->y_[par->granularity-1] || floatcmp(yy,val[ii]->y_[par->granularity-1])==1) &&
                         (zz >= val[ii]->z_[0] || floatcmp(zz,val[ii]->z_[0])==1) && 
                         (zz <= val[ii]->z_[par->granularity-1] || floatcmp(zz,val[ii]->z_[par->granularity-1])==1) ) {
                      int val_level = val[ii]->level_;
                      if ( val_level > highest_level ) {
                        found = true;
                        last_time = ii;
                        highest_level = val_level;
                      }
                    }
                  }
                }

                if ( !found ) {
                  std::cout << " DEBUG coords " << xx << " " << yy << " " << zz <<  std::endl;
                  for (int ii=0;ii<val.size();ii++) {
                    std::cout << " DEBUG available x " << val[ii]->x_[0] << " " << val[ii]->x_[par->granularity-1] << " " <<  std::endl;
                    std::cout << " DEBUG available y " << val[ii]->y_[0] << " " << val[ii]->y_[par->granularity-1] << " " <<  std::endl;
                    std::cout << " DEBUG available z " << val[ii]->z_[0] << " " << val[ii]->z_[par->granularity-1] << " " <<  std::endl;
                    std::cout << " DEBUG level: " << val[ii]->level_ << std::endl;
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
                
                for (int ll=0;ll<num_eqns;ll++) {
                  resultval->value_[i+n*(j+n*k)].phi[0][ll] = val[last_time]->value_[aa+n*(bb+n*cc)].phi[0][ll]; 
                }
              } } }
              // }}}
            }
            if ( val[compute_index]->timestep_ >= par->nt0-2 ) {
              return 0;
            }
            return 1;
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

