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

    inline int 
    stencil::findindex(had_double_type &x,had_double_type &y, had_double_type &z,
                       access_memory_block<stencil_data> &val,
                       int &xindex,int &yindex,int &zindex,int n) {

      // find the index that has this point
      int i;
      for (i=0;i<n;i++) {
        if ( floatcmp(x,val->x_[i]) == 1 ) {
          xindex = i;
          break;
        }
      }      
      for (i=0;i<n;i++) {
        if ( floatcmp(y,val->y_[i]) == 1 ) {
          yindex = i;
          break;
        }
      }      
      for (i=0;i<n;i++) {
        if ( floatcmp(z,val->z_[i]) == 1 ) {
          zindex = i;
          break;
        }
      }      

      if ( xindex < 0 || yindex < 0 || zindex < 0 ) return 1;
      else return 0;
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

    // interp3d {{{
    void stencil::interp3d(had_double_type &x,had_double_type &y, had_double_type &z,
                                      access_memory_block<stencil_data> &val, 
                                      nodedata &result, Parameter const& par) {

      int ii = -1;
      int jj = -1;
      int kk = -1;

      // set up index bounds
      for (int i=0;i<par->granularity;i++) {
        if ( floatcmp_ge(val->x_[i],x) ) {
          ii = i;
          break;
        }         
      }
      for (int i=0;i<par->granularity;i++) {
        if ( floatcmp_ge(val->y_[i],y) ) {
          jj = i;
          break;
        }         
      }
      for (int i=0;i<par->granularity;i++) {
        if ( floatcmp_ge(val->z_[i],z) ) {
          kk = i;
          break;
        }         
      }
      BOOST_ASSERT(ii > -1 && jj > -1 && kk > -1);

      int nx = par->granularity;
      int ny = par->granularity;
      int nz = par->granularity;

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
          result.phi[0][ll] = val->value_[ii+nx*(jj+ny*kk)].phi[0][ll];
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
          result.phi[0][ll] = interp_linear(val->value_[ii-1+nx*(jj+ny*kk)].phi[0][ll],
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
          result.phi[0][ll] = interp_linear(tmp2[0][0][ll],tmp2[1][0][ll],y,
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
          result.phi[0][ll] = interp_linear(tmp3[0][ll],tmp3[1][ll],
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
    // }}}

    // special interp 3d {{{
    void stencil::special_interp3d(had_double_type &xt,had_double_type &yt, had_double_type &zt,had_double_type &dx,
                                      access_memory_block<stencil_data> &val0, 
                                      access_memory_block<stencil_data> &val1, 
                                      access_memory_block<stencil_data> &val2, 
                                      access_memory_block<stencil_data> &val3, 
                                      access_memory_block<stencil_data> &val4, 
                                      access_memory_block<stencil_data> &val5, 
                                      access_memory_block<stencil_data> &val6, 
                                      access_memory_block<stencil_data> &val7, 
                                      nodedata &result, Parameter const& par) {

      // we know the patch that has our data; now we need to find the index of the anchor we need inside the patch
      had_double_type x[8],y[8],z[8];  
      int li,lj,lk;
      int i,j,k;
      int xindex[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
      int yindex[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
      int zindex[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
      int rc;

      nodedata work[8];
 
      int n = par->granularity;

      // find the index points {{{
      li = -1; lj = -1; lk = -1;
      z[0] = zt + lk*dx; y[0] = yt + lj*dx; x[0] = xt + li*dx;
      rc = findindex(x[0],y[0],z[0],val0,xindex[0],yindex[0],zindex[0],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[0].phi[0][ll] = val0->value_[xindex[0]+n*(yindex[0]+n*zindex[0])].phi[0][ll];
        }
      } else {
        interp3d(x[0],y[0],z[0],val0,work[0],par);
      }

      li =  1; lj = -1; lk = -1;
      z[1] = zt + lk*dx; y[1] = yt + lj*dx; x[1] = xt + li*dx;
      rc = findindex(x[1],y[1],z[1],val1,xindex[1],yindex[1],zindex[1],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[1].phi[0][ll] = val1->value_[xindex[1]+n*(yindex[1]+n*zindex[1])].phi[0][ll];
        }
      } else {
        interp3d(x[1],y[1],z[1],val1,work[1],par);
      }

      li =  1; lj =  1; lk = -1;
      z[2] = zt + lk*dx; y[2] = yt + lj*dx; x[2] = xt + li*dx;
      rc = findindex(x[2],y[2],z[2],val2,xindex[2],yindex[2],zindex[2],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[2].phi[0][ll] = val2->value_[xindex[2]+n*(yindex[2]+n*zindex[2])].phi[0][ll];
        }
      } else {
        interp3d(x[2],y[2],z[2],val2,work[2],par);
      }

      li = -1; lj =  1; lk = -1;
      z[3] = zt + lk*dx; y[3] = yt + lj*dx; x[3] = xt + li*dx;
      rc = findindex(x[3],y[3],z[3],val3,xindex[3],yindex[3],zindex[3],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[3].phi[0][ll] = val3->value_[xindex[3]+n*(yindex[3]+n*zindex[3])].phi[0][ll];
        }
      } else {
        interp3d(x[3],y[3],z[3],val3,work[3],par);
      }

      li = -1; lj = -1; lk =  1;
      z[4] = zt + lk*dx; y[4] = yt + lj*dx; x[4] = xt + li*dx;
      rc = findindex(x[4],y[4],z[4],val4,xindex[4],yindex[4],zindex[4],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[4].phi[0][ll] = val4->value_[xindex[4]+n*(yindex[4]+n*zindex[4])].phi[0][ll];
        }
      } else {
        interp3d(x[4],y[4],z[4],val4,work[4],par);
      }

      li =  1; lj = -1; lk =  1;
      z[5] = zt + lk*dx; y[5] = yt + lj*dx; x[5] = xt + li*dx;
      rc = findindex(x[5],y[5],z[5],val5,xindex[5],yindex[5],zindex[5],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[5].phi[0][ll] = val5->value_[xindex[5]+n*(yindex[5]+n*zindex[5])].phi[0][ll];
        }
      } else {
        interp3d(x[5],y[5],z[5],val5,work[5],par);
      }

      li =  1; lj =  1; lk =  1;
      z[6] = zt + lk*dx; y[6] = yt + lj*dx; x[6] = xt + li*dx;
      rc = findindex(x[6],y[6],z[6],val6,xindex[6],yindex[6],zindex[6],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[6].phi[0][ll] = val6->value_[xindex[6]+n*(yindex[6]+n*zindex[6])].phi[0][ll];
        }
      } else {
        interp3d(x[6],y[6],z[6],val6,work[6],par);
      }

      li = -1; lj =  1; lk =  1;
      z[7] = zt + lk*dx; y[7] = yt + lj*dx; x[7] = xt + li*dx;
      rc = findindex(x[7],y[7],z[7],val7,xindex[7],yindex[7],zindex[7],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[7].phi[0][ll] = val7->value_[xindex[7]+n*(yindex[7]+n*zindex[7])].phi[0][ll];
        }
      } else {
        interp3d(x[7],y[7],z[7],val7,work[7],par);
      }
      // }}}

      had_double_type tmp2[2][2][num_eqns];
      had_double_type tmp3[2][num_eqns];

      // interp x
      for (int ll=0;ll<num_eqns;ll++) {
        tmp2[0][0][ll] = interp_linear(work[0].phi[0][ll],
                                       work[1].phi[0][ll],
                                             xt,
                                       x[0],x[1]);

        tmp2[1][0][ll] = interp_linear(work[3].phi[0][ll],
                                       work[2].phi[0][ll],
                                             xt,
                                       x[3],x[2]);

        tmp2[0][1][ll] = interp_linear(work[4].phi[0][ll],
                                       work[5].phi[0][ll],
                                             xt,
                                       x[4],x[5]);

        tmp2[1][1][ll] = interp_linear(work[7].phi[0][ll],
                                       work[6].phi[0][ll],
                                             xt,
                                       x[7],x[6]);

      }
   
      // interp y
      for (int ll=0;ll<num_eqns;ll++) {
        tmp3[0][ll] = interp_linear(tmp2[0][0][ll],tmp2[1][0][ll],yt,
                                     y[0],y[2]);

        tmp3[1][ll] = interp_linear(tmp2[0][1][ll],tmp2[1][1][ll],yt,
                                     y[4],y[6]);
      }

      // interp z
      for (int ll=0;ll<num_eqns;ll++) {
        result.phi[0][ll] = interp_linear(tmp3[0][ll],tmp3[1][ll],
                                          zt,
                                          z[0],z[4]);
      } 

      return;
    }
    // }}}

    // special interp 2d xy {{{
    void stencil::special_interp2d_xy(had_double_type &xt,had_double_type &yt,had_double_type &zt,had_double_type &dx,
                                      access_memory_block<stencil_data> &val0, 
                                      access_memory_block<stencil_data> &val1, 
                                      access_memory_block<stencil_data> &val2, 
                                      access_memory_block<stencil_data> &val3, 
                                      nodedata &result, Parameter const& par) {

      // we know the patch that has our data; now we need to find the index of the anchor we need inside the patch
      had_double_type x[4],y[4],z[4];  
      int li,lj,lk;
      int i,j,k;

      int xindex[4] = {-1,-1,-1,-1};
      int yindex[4] = {-1,-1,-1,-1};
      int zindex[4] = {-1,-1,-1,-1};

      int n = par->granularity;
      int rc;

      nodedata work[4];

      li = -1; lj = -1; lk = 0;
      z[0] = zt + lk*dx; y[0] = yt + lj*dx; x[0] = xt + li*dx;
      rc = findindex(x[0],y[0],z[0],val0,xindex[0],yindex[0],zindex[0],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[0].phi[0][ll] = val0->value_[xindex[0]+n*(yindex[0]+n*zindex[0])].phi[0][ll];
        }
      } else {
        interp3d(x[0],y[0],z[0],val0,work[0],par);
      }

      li =  1; lj = -1; lk = 0;
      z[1] = zt + lk*dx; y[1] = yt + lj*dx; x[1] = xt + li*dx;
      rc = findindex(x[1],y[1],z[1],val1,xindex[1],yindex[1],zindex[1],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[1].phi[0][ll] = val1->value_[xindex[1]+n*(yindex[1]+n*zindex[1])].phi[0][ll];
        }
      } else {
        interp3d(x[1],y[1],z[1],val1,work[1],par);
      }

      li =  1; lj =  1; lk = 0;
      z[2] = zt + lk*dx; y[2] = yt + lj*dx; x[2] = xt + li*dx;
      rc = findindex(x[2],y[2],z[2],val2,xindex[2],yindex[2],zindex[2],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[2].phi[0][ll] = val2->value_[xindex[2]+n*(yindex[2]+n*zindex[2])].phi[0][ll];
        }
      } else {
        interp3d(x[2],y[2],z[2],val2,work[2],par);
      }

      li = -1; lj =  1; lk = 0;
      z[3] = zt + lk*dx; y[3] = yt + lj*dx; x[3] = xt + li*dx;
      rc = findindex(x[3],y[3],z[3],val3,xindex[3],yindex[3],zindex[3],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[3].phi[0][ll] = val3->value_[xindex[3]+n*(yindex[3]+n*zindex[3])].phi[0][ll];
        }
      } else {
        interp3d(x[3],y[3],z[3],val3,work[3],par);
      }

      had_double_type tmp3[2][num_eqns];

      for (int ll=0;ll<num_eqns;ll++) {
        // interpolate x
        tmp3[0][ll] = interp_linear(work[0].phi[0][ll],
                                    work[1].phi[0][ll], 
                                    xt,
                                    x[0],x[1]);

        tmp3[1][ll] = interp_linear(work[3].phi[3][ll],
                                    work[2].phi[2][ll], 
                                    xt,
                                    x[3],x[2]);

        // interpolate y
        result.phi[0][ll] = interp_linear(tmp3[0][ll],tmp3[1][ll],
                                          yt,
                                          y[0],y[2]);
      }

      return;
    } // }}}

    // special interp 2d xz {{{
    void stencil::special_interp2d_xz(had_double_type &xt,had_double_type &yt,had_double_type &zt,had_double_type &dx,
                                      access_memory_block<stencil_data> &val0, 
                                      access_memory_block<stencil_data> &val1, 
                                      access_memory_block<stencil_data> &val2, 
                                      access_memory_block<stencil_data> &val3, 
                                      nodedata &result, Parameter const& par) {

      // we know the patch that has our data; now we need to find the index of the anchor we need inside the patch
      had_double_type x[4],y[4],z[4];  
      int li,lj,lk;
      int i,j,k;

      int xindex[4] = {-1,-1,-1,-1};
      int yindex[4] = {-1,-1,-1,-1};
      int zindex[4] = {-1,-1,-1,-1};

      int n = par->granularity;
      int rc;

      nodedata work[4];

      li = -1; lj = 0; lk = -1;
      z[0] = zt + lk*dx; y[0] = yt + lj*dx; x[0] = xt + li*dx;
      rc = findindex(x[0],y[0],z[0],val0,xindex[0],yindex[0],zindex[0],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[0].phi[0][ll] = val0->value_[xindex[0]+n*(yindex[0]+n*zindex[0])].phi[0][ll];
        }
      } else {
        interp3d(x[0],y[0],z[0],val0,work[0],par);
      }

      li =  1; lj =  0; lk = -1;
      z[1] = zt + lk*dx; y[1] = yt + lj*dx; x[1] = xt + li*dx;
      rc = findindex(x[1],y[1],z[1],val1,xindex[1],yindex[1],zindex[1],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[1].phi[0][ll] = val1->value_[xindex[1]+n*(yindex[1]+n*zindex[1])].phi[0][ll];
        }
      } else {
        interp3d(x[1],y[1],z[1],val1,work[1],par);
      }

      li = -1; lj =  0; lk = 1;
      z[2] = zt + lk*dx; y[2] = yt + lj*dx; x[2] = xt + li*dx;
      rc = findindex(x[2],y[2],z[2],val2,xindex[2],yindex[2],zindex[2],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[2].phi[0][ll] = val2->value_[xindex[2]+n*(yindex[2]+n*zindex[2])].phi[0][ll];
        }
      } else {
        interp3d(x[2],y[2],z[2],val2,work[2],par);
      }

      li =  1; lj =  0; lk = 1;
      z[3] = zt + lk*dx; y[3] = yt + lj*dx; x[3] = xt + li*dx;
      rc = findindex(x[3],y[3],z[3],val3,xindex[3],yindex[3],zindex[3],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[3].phi[0][ll] = val3->value_[xindex[3]+n*(yindex[3]+n*zindex[3])].phi[0][ll];
        }
      } else {
        interp3d(x[3],y[3],z[3],val3,work[3],par);
      }

      had_double_type tmp3[2][num_eqns];

      for (int ll=0;ll<num_eqns;ll++) {
        // interpolate x
        tmp3[0][ll] = interp_linear(work[0].phi[0][ll],
                                    work[1].phi[0][ll], 
                                    xt,
                                    x[0],x[1]);

        tmp3[1][ll] = interp_linear(work[2].phi[3][ll],
                                    work[3].phi[2][ll], 
                                    xt,
                                    x[2],x[3]);

        // interpolate z
        result.phi[0][ll] = interp_linear(tmp3[0][ll],tmp3[1][ll],
                                          zt,
                                          z[0],z[2]);
      }

      return;
    } // }}}

    // special interp 2d yz {{{
    void stencil::special_interp2d_yz(had_double_type &xt,had_double_type &yt,had_double_type &zt,had_double_type &dx,
                                      access_memory_block<stencil_data> &val0, 
                                      access_memory_block<stencil_data> &val1, 
                                      access_memory_block<stencil_data> &val2, 
                                      access_memory_block<stencil_data> &val3, 
                                      nodedata &result, Parameter const& par) {

      // we know the patch that has our data; now we need to find the index of the anchor we need inside the patch
      had_double_type x[4],y[4],z[4];  
      int li,lj,lk;
      int i,j,k;

      int xindex[4] = {-1,-1,-1,-1};
      int yindex[4] = {-1,-1,-1,-1};
      int zindex[4] = {-1,-1,-1,-1};

      int n = par->granularity;
      int rc;

      nodedata work[4];

      li =  0; lj = -1; lk = -1;
      z[0] = zt + lk*dx; y[0] = yt + lj*dx; x[0] = xt + li*dx;
      rc = findindex(x[0],y[0],z[0],val0,xindex[0],yindex[0],zindex[0],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[0].phi[0][ll] = val0->value_[xindex[0]+n*(yindex[0]+n*zindex[0])].phi[0][ll];
        }
      } else {
        interp3d(x[0],y[0],z[0],val0,work[0],par);
      }

      li =  0; lj =  1; lk = -1;
      z[1] = zt + lk*dx; y[1] = yt + lj*dx; x[1] = xt + li*dx;
      rc = findindex(x[1],y[1],z[1],val1,xindex[1],yindex[1],zindex[1],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[1].phi[0][ll] = val1->value_[xindex[1]+n*(yindex[1]+n*zindex[1])].phi[0][ll];
        }
      } else {
        interp3d(x[1],y[1],z[1],val1,work[1],par);
      }

      li =  0; lj =  -1; lk = 1;
      z[2] = zt + lk*dx; y[2] = yt + lj*dx; x[2] = xt + li*dx;
      rc = findindex(x[2],y[2],z[2],val2,xindex[2],yindex[2],zindex[2],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[2].phi[0][ll] = val2->value_[xindex[2]+n*(yindex[2]+n*zindex[2])].phi[0][ll];
        }
      } else {
        interp3d(x[2],y[2],z[2],val2,work[2],par);
      }

      li =  0; lj =  1; lk = 1;
      z[3] = zt + lk*dx; y[3] = yt + lj*dx; x[3] = xt + li*dx;
      rc = findindex(x[3],y[3],z[3],val3,xindex[3],yindex[3],zindex[3],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[3].phi[0][ll] = val3->value_[xindex[3]+n*(yindex[3]+n*zindex[3])].phi[0][ll];
        }
      } else {
        interp3d(x[3],y[3],z[3],val3,work[3],par);
      }

      had_double_type tmp3[2][num_eqns];

      for (int ll=0;ll<num_eqns;ll++) {
        // interpolate y
        tmp3[0][ll] = interp_linear(work[0].phi[0][ll],
                                    work[1].phi[0][ll], 
                                    yt,
                                    y[0],y[1]);

        tmp3[1][ll] = interp_linear(work[2].phi[3][ll],
                                    work[3].phi[2][ll], 
                                    yt,
                                    y[2],y[3]);

        // interpolate z
        result.phi[0][ll] = interp_linear(tmp3[0][ll],tmp3[1][ll],
                                          zt,
                                          z[0],z[2]);
      }

      return;
    } // }}}

    // special interp 1d x {{{
    void stencil::special_interp1d_x(had_double_type &xt,had_double_type &yt,had_double_type &zt,had_double_type &dx,
                                      access_memory_block<stencil_data> &val0, 
                                      access_memory_block<stencil_data> &val1, 
                                      nodedata &result, Parameter const& par) {

      // we know the patch that has our data; now we need to find the index of the anchor we need inside the patch
      had_double_type x[2],y[2],z[2];  
      int li,lj,lk;
      int i,j,k;

      int xindex[2] = {-1,-1};
      int yindex[2] = {-1,-1};
      int zindex[2] = {-1,-1};

      int n = par->granularity;
      int rc;

      nodedata work[2];

      li = -1; lj = 0; lk = 0;
      z[0] = zt + lk*dx; y[0] = yt + lj*dx; x[0] = xt + li*dx;
      rc = findindex(x[0],y[0],z[0],val0,xindex[0],yindex[0],zindex[0],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[0].phi[0][ll] = val0->value_[xindex[0]+n*(yindex[0]+n*zindex[0])].phi[0][ll];
        }
      } else {
        interp3d(x[0],y[0],z[0],val0,work[0],par);
      }

      li =  1; lj =  0; lk = 0;
      z[1] = zt + lk*dx; y[1] = yt + lj*dx; x[1] = xt + li*dx;
      rc = findindex(x[1],y[1],z[1],val1,xindex[1],yindex[1],zindex[1],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[1].phi[0][ll] = val1->value_[xindex[1]+n*(yindex[1]+n*zindex[1])].phi[0][ll];
        }
      } else {
        interp3d(x[1],y[1],z[1],val1,work[1],par);
      }

      for (int ll=0;ll<num_eqns;ll++) {
        // interpolate x
        result.phi[0][ll] = interp_linear(work[0].phi[0][ll],work[1].phi[0][ll],
                                          xt,
                                          x[0],x[1]);
      }

      return;
    } // }}}

    // special interp 1d y {{{
    void stencil::special_interp1d_y(had_double_type &xt,had_double_type &yt,had_double_type &zt,had_double_type &dx,
                                      access_memory_block<stencil_data> &val0, 
                                      access_memory_block<stencil_data> &val1, 
                                      nodedata &result, Parameter const& par) {

      // we know the patch that has our data; now we need to find the index of the anchor we need inside the patch
      had_double_type x[2],y[2],z[2];  
      int li,lj,lk;
      int i,j,k;

      int xindex[2] = {-1,-1};
      int yindex[2] = {-1,-1};
      int zindex[2] = {-1,-1};

      int n = par->granularity;
      int rc;

      nodedata work[2];

      li =  0; lj = -1; lk = 0;
      z[0] = zt + lk*dx; y[0] = yt + lj*dx; x[0] = xt + li*dx;
      rc = findindex(x[0],y[0],z[0],val0,xindex[0],yindex[0],zindex[0],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[0].phi[0][ll] = val0->value_[xindex[0]+n*(yindex[0]+n*zindex[0])].phi[0][ll];
        }
      } else {
        interp3d(x[0],y[0],z[0],val0,work[0],par);
      }

      li =  0; lj =  1; lk = 0;
      z[1] = zt + lk*dx; y[1] = yt + lj*dx; x[1] = xt + li*dx;
      rc = findindex(x[1],y[1],z[1],val1,xindex[1],yindex[1],zindex[1],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[1].phi[0][ll] = val1->value_[xindex[1]+n*(yindex[1]+n*zindex[1])].phi[0][ll];
        }
      } else {
        interp3d(x[1],y[1],z[1],val1,work[1],par);
      }

      for (int ll=0;ll<num_eqns;ll++) {
        // interpolate y
        result.phi[0][ll] = interp_linear(work[0].phi[0][ll],work[1].phi[0][ll],
                                          yt,
                                          y[0],y[1]);
      }

      return;
    } // }}}

    // special interp 1d z {{{
    void stencil::special_interp1d_z(had_double_type &xt,had_double_type &yt,had_double_type &zt,had_double_type &dx,
                                      access_memory_block<stencil_data> &val0, 
                                      access_memory_block<stencil_data> &val1, 
                                      nodedata &result, Parameter const& par) {

      // we know the patch that has our data; now we need to find the index of the anchor we need inside the patch
      had_double_type x[2],y[2],z[2];  
      int li,lj,lk;
      int i,j,k;

      int xindex[2] = {-1,-1};
      int yindex[2] = {-1,-1};
      int zindex[2] = {-1,-1};

      int n = par->granularity;
      int rc;

      nodedata work[2];

      li = 0; lj = 0; lk = -1;
      z[0] = zt + lk*dx; y[0] = yt + lj*dx; x[0] = xt + li*dx;
      rc = findindex(x[0],y[0],z[0],val0,xindex[0],yindex[0],zindex[0],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[0].phi[0][ll] = val0->value_[xindex[0]+n*(yindex[0]+n*zindex[0])].phi[0][ll];
        }
      } else {
        interp3d(x[0],y[0],z[0],val0,work[0],par);
      }

      li = 0; lj = 0; lk = 1;
      z[1] = zt + lk*dx; y[1] = yt + lj*dx; x[1] = xt + li*dx;
      rc = findindex(x[1],y[1],z[1],val1,xindex[1],yindex[1],zindex[1],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<num_eqns;ll++) {
          work[1].phi[0][ll] = val1->value_[xindex[1]+n*(yindex[1]+n*zindex[1])].phi[0][ll];
        }
      } else {
        interp3d(x[1],y[1],z[1],val1,work[1],par);
      }

      for (int ll=0;ll<num_eqns;ll++) {
        // interpolate z
        result.phi[0][ll] = interp_linear(work[0].phi[0][ll],work[1].phi[0][ll],
                                          zt,
                                          z[0],z[1]);
      }

      return;
    } // }}}
        
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

            // TEST
            resultval.get() = val[compute_index].get();

            // copy over critical info
            resultval->x_ = val[compute_index]->x_;
            resultval->y_ = val[compute_index]->y_;
            resultval->z_ = val[compute_index]->z_;
            resultval->value_.resize(val[compute_index]->value_.size());
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
              int n = par->granularity;

#if 0
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
#endif

              for (int k=0;k<n;k++) {
                had_double_type zt = resultval->z_[k];
              for (int j=0;j<n;j++) {
                had_double_type yt = resultval->y_[j];
              for (int i=0;i<n;i++) {
                had_double_type xt = resultval->x_[i];

                // check if this is a prolongation point
                if ( !(par->min[level]+2*par->gw*dx < xt && par->max[level]-2*par->gw*dx > xt &&     
                       par->min[level]+2*par->gw*dx < yt && par->max[level]-2*par->gw*dx > yt &&     
                       par->min[level]+2*par->gw*dx < zt && par->max[level]-2*par->gw*dx > zt)  
                   ) {
                  // this is a prolongation point -- overwrite the value with an interpolated value from the coarse mesh
                  bool found = false;
                  for (int ii=0;ii<val.size();ii++) {
                    if ( ii != compute_index ) {
                      if ( floatcmp_ge(xt,val[ii]->x_[0])  && floatcmp_le(xt,val[ii]->x_[n-1]) &&
                           floatcmp_ge(yt,val[ii]->y_[0])  && floatcmp_le(yt,val[ii]->y_[n-1]) &&
                           floatcmp_ge(zt,val[ii]->z_[0])  && floatcmp_le(zt,val[ii]->z_[n-1]) ) {
                        found = true;
                        // interpolate
                        interp3d(xt,yt,zt,val[ii],resultval->value_[i+n*(j+n*k)],par);
                        break;
                      }
                    }
                  }

                  int anchor_index[27];
                  int has_corner[27] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
                  if ( !found ) {
                    // find the interpolating the anchors needed  
                    for (int lk=-1;lk<2;lk++) {
                      had_double_type zn = zt + lk*dx;
                    for (int lj=-1;lj<2;lj++) {
                      had_double_type yn = yt + lj*dx;
                    for (int li=-1;li<2;li++) {
                      had_double_type xn = xt + li*dx;
                      
                      for (int ii=0;ii<val.size();ii++) {
                        if ( floatcmp_ge(xn,val[ii]->x_[0])  && floatcmp_le(xn,val[ii]->x_[par->granularity-1]) &&
                             floatcmp_ge(yn,val[ii]->y_[0])  && floatcmp_le(yn,val[ii]->y_[par->granularity-1]) &&
                             floatcmp_ge(zn,val[ii]->z_[0])  && floatcmp_le(zn,val[ii]->z_[par->granularity-1]) &&
                             ii != compute_index ) {
                          if (li == -1 && lj == -1 && lk == -1 ) {
                            has_corner[0] = 1;
                            anchor_index[0] = ii;
                          }
                          if (li ==  1 && lj == -1 && lk == -1 ) {
                            has_corner[1] = 1;
                            anchor_index[1] = ii;
                          }
                          if (li ==  1 && lj ==  1 && lk == -1 ) {
                            has_corner[2] = 1;
                            anchor_index[2] = ii;
                          }
                          if (li == -1 && lj ==  1 && lk == -1 ) {
                            has_corner[3] = 1;
                            anchor_index[3] = ii;
                          }
                          if (li == -1 && lj == -1 && lk ==  1 ) {
                            has_corner[4] = 1;
                            anchor_index[4] = ii;
                          }
                          if (li ==  1 && lj == -1 && lk ==  1 ) {
                            has_corner[5] = 1;
                            anchor_index[5] = ii;
                          }
                          if (li ==  1 && lj ==  1 && lk ==  1 ) {
                            has_corner[6] = 1;
                            anchor_index[6] = ii;
                          }
                          if (li == -1 && lj ==  1 && lk ==  1 ) {
                            has_corner[7] = 1;
                            anchor_index[7] = ii;
                          }

                          if (li == -1 && lj == -1 && lk == 0 ) {
                            has_corner[8]   = 1;
                            anchor_index[8] = ii;
                          }
                          if (li ==  1 && lj == -1 && lk == 0 ) {
                            has_corner[9]  = 1;
                            anchor_index[9] = ii;
                          }
                          if (li ==  1 && lj ==  1 && lk == 0 ) {
                            has_corner[10] = 1;
                            anchor_index[10] = ii;
                          }
                          if (li == -1 && lj ==  1 && lk == 0 ) {
                            has_corner[11] = 1;
                            anchor_index[11] = ii;
                          }

                         if (li == 0 && lj == -1 && lk == -1 ) {
                            has_corner[12] = 1;
                            anchor_index[12] = ii;
                          }
                          if (li == 1 && lj == 0 && lk == -1 ) {
                            has_corner[13] = 1;
                            anchor_index[13] = ii;
                          }
                          if (li == 0 && lj == 1 && lk == -1 ) {
                            has_corner[14] = 1;
                            anchor_index[14] = ii;
                          }
                          if (li == -1 && lj == 0 && lk == -1 ) {
                            has_corner[15] = 1;
                            anchor_index[15] = ii;
                          }

                          if (li == 0 && lj == -1 && lk == 0 ) {
                            has_corner[16] = 1;
                            anchor_index[16] = ii;
                          }
                          if (li == 1 && lj == 0 && lk ==  0 ) {
                            has_corner[17] = 1;
                            anchor_index[17] = ii;
                          }
                          if (li == 0 && lj == 1 && lk ==  0 ) {
                            has_corner[18] = 1;
                            anchor_index[18] = ii;
                          }
                          if (li == -1 && lj == 0 && lk == 0 ) {
                            has_corner[19] = 1;
                            anchor_index[19] = ii;
                          }

                          if (li == 0 && lj == -1 && lk == 1 ) {
                            has_corner[20] = 1;
                            anchor_index[20] = ii;
                          }
                          if (li == 1 && lj == 0 && lk ==  1 ) {
                            has_corner[21] = 1;
                            anchor_index[21] = ii;
                          }
                          if (li == 0 && lj == 1 && lk ==  1 ) {
                            has_corner[22] = 1;
                            anchor_index[22] = ii;
                          }
                          if (li == -1 && lj == 0 && lk == 1 ) {
                            has_corner[23] = 1;
                            anchor_index[23] = ii;
                          }

                          if (li == 0 && lj == 0 && lk == 1 ) {
                            has_corner[24] = 1;
                            anchor_index[24] = ii;
                          }

                          if (li == 0 && lj == 0 && lk == -1 ) {
                            has_corner[25] = 1;
                            anchor_index[25] = ii;
                          }
                        }
                      }
                          
                    } } }
                  }

                  // Now we have the complete picture.  Determine what the interpolation options are and proceed. 
                  if ( has_corner[0] == 1 && has_corner[1] == 1 && has_corner[2] == 1 &&
                       has_corner[3] == 1 && has_corner[4] == 1 && has_corner[5] == 1 &&
                       has_corner[6] == 1 && has_corner[7] == 1 ) {
                    // 3D interpolation
                    found = true;

                    special_interp3d(xt,yt,zt,dx,
                                       val[anchor_index[0]],
                                       val[anchor_index[1]],
                                       val[anchor_index[2]],
                                       val[anchor_index[3]],
                                       val[anchor_index[4]],
                                       val[anchor_index[5]],
                                       val[anchor_index[6]],
                                       val[anchor_index[7]],
                                       resultval->value_[i+n*(j+n*k)],par);
                  } else if ( has_corner[8] == 1 && has_corner[9] == 1 && has_corner[10] == 1 && has_corner[11] == 1 ) {
                    // 2D interp
                    found = true;
                    special_interp2d_xy(xt,yt,zt,dx,
                                        val[anchor_index[8]],val[anchor_index[9]],
                                        val[anchor_index[10]],val[anchor_index[11]],resultval->value_[i+n*(j+n*k)],par);
                  } else if ( has_corner[12] == 1 && has_corner[14] == 1 && has_corner[20] ==1 && has_corner[22] == 1 ) {
                    // 2D interp
                    found = true;
                    special_interp2d_yz(xt,yt,zt,dx,
                                        val[anchor_index[12]],val[anchor_index[14]],
                                        val[anchor_index[20]],val[anchor_index[22]],resultval->value_[i+n*(j+n*k)],par);
                  } else if ( has_corner[15] == 1 && has_corner[13] == 1 && has_corner[23] == 1 && has_corner[21] == 1) {
                    // 2D interp
                    found = true;
                    special_interp2d_xz(xt,yt,zt,dx,
                                        val[anchor_index[15]],val[anchor_index[13]],
                                        val[anchor_index[23]],val[anchor_index[21]],resultval->value_[i+n*(j+n*k)],par);
                  } else if ( has_corner[16] == 1 && has_corner[18] == 1 ) {
                    // 1D interp
                    found = true;
                    special_interp1d_y(xt,yt,zt,dx,
                                       val[anchor_index[16]],val[anchor_index[18]],
                                       resultval->value_[i+n*(j+n*k)],par);
                  } else if ( has_corner[19] == 1 && has_corner[17] == 1 ) {
                    // 1D interp
                    found = true;
                    special_interp1d_x(xt,yt,zt,dx,
                                       val[anchor_index[19]],val[anchor_index[17]],
                                       resultval->value_[i+n*(j+n*k)],par);
                  } else if ( has_corner[24] == 1 && has_corner[25] == 1 ) {
                    // 1D interp
                    found = true;
                    special_interp1d_z(xt,yt,zt,dx,
                                       val[anchor_index[25]],val[anchor_index[24]],
                                       resultval->value_[i+n*(j+n*k)],par);
                  }

                  if ( !found ) {
                    std::cout << " PROBLEM: point " << xt << " " << yt << " " << zt << " BBOX : " <<  par->min[level] << " " << par->min[level]+2*par->gw*dx << " " <<  par->max[level] << " " << par->max[level]-2*par->gw*dx << std::endl;
                    std::cout << " Available data: " << std::endl;
                     for (int ii=0;ii<val.size();ii++) {
                       std::cout << val[ii]->x_[0] << " " << val[ii]->x_[par->granularity-1] << std::endl;
                       std::cout << val[ii]->y_[0] << " " << val[ii]->y_[par->granularity-1] << std::endl;
                       std::cout << val[ii]->z_[0] << " " << val[ii]->z_[par->granularity-1] << std::endl;
                       std::cout << " " << std::endl;
                     }      
                     for (int ii=0;ii<27;ii++) {
                       std::cout << " Has corner : " << ii << " " << has_corner[ii] << std::endl;
                     }      
                            
                    BOOST_ASSERT(false);
                  }
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

                // Check if this is a restriction point -- is it further than gw coarse dx points away from a fine mesh boundary?
                if ( par->min[level+1] < xx - par->gw*dx && xx+par->gw*dx > par->max[level+1] &&
                     par->min[level+1] < yy - par->gw*dx && yy+par->gw*dx > par->max[level+1] &&
                     par->min[level+1] < zz - par->gw*dx && zz+par->gw*dx > par->max[level+1] ) {

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
          resultval->level_ = val[compute_index]->level_;

          resultval->max_index_ = val[compute_index]->max_index_;
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

