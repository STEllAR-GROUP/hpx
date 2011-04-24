//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Matthew Anderson
//  Copyright (c)      2011 Bryce Lelbach
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
    stencil::floatcmp(double_type const& x1, double_type const& x2) {
      // compare two floating point numbers
      static double_type const epsilon = 1.e-8;
      if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
        // the numbers are close enough for coordinate comparison
        return true;
      } else {
        return false;
      }
    }

    inline bool 
    stencil::floatcmp_le(double_type const& x1, double_type const& x2) {
      // compare two floating point numbers
      static double_type const epsilon = 1.e-8;

      if ( x1 < x2 ) return true;

      if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
        // the numbers are close enough for coordinate comparison
        return true;
      } else {
        return false;
      }
    }

    inline bool 
    stencil::floatcmp_ge(double_type const& x1, double_type const& x2) {
      // compare two floating point numbers
      static double_type const epsilon = 1.e-8;

      if ( x1 > x2 ) return true;

      if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
        // the numbers are close enough for coordinate comparison
        return true;
      } else {
        return false;
      }
    }

    inline int 
    stencil::findindex(double_type &x,double_type &y, double_type &z,
                       access_memory_block<stencil_data> &val,
                       int &xindex,int &yindex,int &zindex,int n) {
      // find the index that has this point
      register bool foundx = false;
      register bool foundy = false;
      register bool foundz = false;
      for (int i=0;i<n;++i) {
        if ( !foundx && floatcmp(y,val->y_[i]) == 1 ) {
          yindex = i;
          if ( foundy && foundz ) break;
          else foundx = true;
        }
        if ( !foundy && floatcmp(x,val->x_[i]) == 1 ) {
          xindex = i;
          if ( foundx && foundz ) break;
          else foundy = true;
        }
        if ( !foundz && floatcmp(z,val->z_[i]) == 1 ) {
          zindex = i;
          if ( foundx && foundy ) break;
          else foundz = true;
        }
      }      

      if ( xindex < 0 || yindex < 0 || zindex < 0 ) return 1;
      else return 0;
    }

    inline std::size_t stencil::findlevel3D(std::size_t step, std::size_t item, 
                                            std::size_t &a, std::size_t &b, std::size_t &c, parameter const& par)
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

    double_type stencil::interp_linear(double_type y1, double_type y2,
                                           double_type x, double_type x1, double_type x2) {
      double_type xx1 = x - x1;
      double_type xx2 = x - x2;
      double_type result = xx2*y1/( (x1-x2) ) + xx1*y2/( (x2-x1) );
  
      return result;
    }

    // interp3d {{{
    void stencil::interp3d(double_type &x,double_type &y, double_type &z,
                                      access_memory_block<stencil_data> &val, 
                                      nodedata &result, parameter const& par) {
      static const int grain = par->granularity;

      int ii = -1;
      int jj = -1;
      int kk = -1;

      // FIXME: this might be more efficient if a separate boolean value was
      // used to track which variables were found (a la findindex), because
      // checking if a register is equal to zero should be faster than any
      // other comparison on some processors (or so I hear).

      // set up index bounds
      for (int i=0;i<grain;++i) {
        if ( ii == -1 && floatcmp_ge(val->x_[i],x) ) {
          ii = i;
          if ( jj != -1 && kk != -1 ) break;
        }         
        if ( jj == -1 && floatcmp_ge(val->y_[i],y) ) {
          jj = i;
          if ( ii != -1 && kk != -1 ) break;
        }         
        if ( kk == -1 && floatcmp_ge(val->z_[i],z) ) {
          kk = i;
          if ( ii != -1 && jj != -1 ) break;
        }         
      }
      BOOST_ASSERT(ii > -1 && jj > -1 && kk > -1);

      // Use the static const variable grain instead.
      //int nx = par->granularity;
      //int ny = par->granularity;
      //int nz = par->granularity;

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
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          result.phi[0][ll] = val->value_[ii+grain*(jj+grain*kk)].phi[0][ll];
        }
        return;
      }

      // Quick sanity check to be sure we have bracketed the point we wish to interpolate
      if ( !no_interp_x ) {
        BOOST_ASSERT(floatcmp_le(val->x_[ii-1],x) && floatcmp_ge(val->x_[ii],x) );
      }
      if ( !no_interp_y ) {
        BOOST_ASSERT(floatcmp_le(val->y_[jj-1],y) && floatcmp_ge(val->y_[jj],y) );
      }
      if ( !no_interp_z ) {
        BOOST_ASSERT(floatcmp_le(val->z_[kk-1],z) && floatcmp_ge(val->z_[kk],z) );
      }

      double_type tmp2[2][2][HPX_SMP_AMR3D_NUM_EQUATIONS];
      double_type tmp3[2][HPX_SMP_AMR3D_NUM_EQUATIONS];

      // interpolate in x {{{
      if ( !no_interp_x && !no_interp_y && !no_interp_z ) {
        for (int k=kk-1;k<kk+1;++k) {
          for (int j=jj-1;j<jj+1;++j) {
            for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
              tmp2[j-(jj-1)][k-(kk-1)][ll] = interp_linear(val->value_[ii-1+grain*(j+grain*k)].phi[0][ll],
                                                   val->value_[ii  +grain*(j+grain*k)].phi[0][ll],
                                                   x,
                                                   val->x_[ii-1],val->x_[ii]);
            }
          }
        }
      } else if ( no_interp_x && !no_interp_y && !no_interp_z ) {
        for (int k=kk-1;k<kk+1;++k) {
          for (int j=jj-1;j<jj+1;++j) {
            for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
              tmp2[j-(jj-1)][k-(kk-1)][ll] = val->value_[ii+grain*(j+grain*k)].phi[0][ll];
            }
          }
        }
      } else if ( !no_interp_x && no_interp_y && !no_interp_z ) {
        for (int k=kk-1;k<kk+1;++k) {
          for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
            tmp2[0][k-(kk-1)][ll] = interp_linear(val->value_[ii-1+grain*(jj+grain*k)].phi[0][ll],
                                              val->value_[ii  +grain*(jj+grain*k)].phi[0][ll],
                                              x,
                                              val->x_[ii-1],val->x_[ii]);
          }
        }
      } else if ( !no_interp_x && !no_interp_y && no_interp_z ) {
        for (int j=jj-1;j<jj+1;++j) {
          for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
            tmp2[j-(jj-1)][0][ll] = interp_linear(val->value_[ii-1+grain*(j+grain*kk)].phi[0][ll],
                                              val->value_[ii  +grain*(j+grain*kk)].phi[0][ll],
                                              x,
                                              val->x_[ii-1],val->x_[ii]);
          }
        }
      } else if ( no_interp_x && no_interp_y && !no_interp_z ) {
        for (int k=kk-1;k<kk+1;++k) {
          for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
            tmp2[0][k-(kk-1)][ll] = val->value_[ii+grain*(jj+grain*k)].phi[0][ll];
          }
        }
      } else if ( no_interp_x && !no_interp_y && no_interp_z ) {
        for (int j=jj-1;j<jj+1;++j) {
          for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
            tmp2[j-(jj-1)][0][ll] = val->value_[ii+grain*(j+grain*kk)].phi[0][ll];
          }
        }
      } else if ( !no_interp_x && no_interp_y && no_interp_z ) {
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          result.phi[0][ll] = interp_linear(val->value_[ii-1+grain*(jj+grain*kk)].phi[0][ll],
                                            val->value_[ii  +grain*(jj+grain*kk)].phi[0][ll],
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
        for (int k=0;k<2;++k) {
          for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
            tmp3[k][ll] = interp_linear(tmp2[0][k][ll],tmp2[1][k][ll],y,
                                         val->y_[jj-1],val->y_[jj]);
          }
        }
      } else if ( no_interp_y && !no_interp_z ) {
        for (int k=0;k<2;++k) {
          for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
            tmp3[k][ll] = tmp2[0][k][ll];
          }
        }
      } else if ( !no_interp_y && no_interp_z ) {
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
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
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
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
    void stencil::special_interp3d(double_type &xt,double_type &yt, double_type &zt,double_type &dx,
                                      access_memory_block<stencil_data> &val0, 
                                      access_memory_block<stencil_data> &val1, 
                                      access_memory_block<stencil_data> &val2, 
                                      access_memory_block<stencil_data> &val3, 
                                      access_memory_block<stencil_data> &val4, 
                                      access_memory_block<stencil_data> &val5, 
                                      access_memory_block<stencil_data> &val6, 
                                      access_memory_block<stencil_data> &val7, 
                                      nodedata &result, parameter const& par) {

      // we know the patch that has our data; now we need to find the index of the anchor we need inside the patch
      double_type x[8],y[8],z[8];  
      int li,lj,lk;
//       int i,j,k;
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
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[0].phi[0][ll] = val0->value_[xindex[0]+n*(yindex[0]+n*zindex[0])].phi[0][ll];
        }
      } else {
        interp3d(x[0],y[0],z[0],val0,work[0],par);
      }

      li =  1; lj = -1; lk = -1;
      z[1] = zt + lk*dx; y[1] = yt + lj*dx; x[1] = xt + li*dx;
      rc = findindex(x[1],y[1],z[1],val1,xindex[1],yindex[1],zindex[1],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[1].phi[0][ll] = val1->value_[xindex[1]+n*(yindex[1]+n*zindex[1])].phi[0][ll];
        }
      } else {
        interp3d(x[1],y[1],z[1],val1,work[1],par);
      }

      li =  1; lj =  1; lk = -1;
      z[2] = zt + lk*dx; y[2] = yt + lj*dx; x[2] = xt + li*dx;
      rc = findindex(x[2],y[2],z[2],val2,xindex[2],yindex[2],zindex[2],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[2].phi[0][ll] = val2->value_[xindex[2]+n*(yindex[2]+n*zindex[2])].phi[0][ll];
        }
      } else {
        interp3d(x[2],y[2],z[2],val2,work[2],par);
      }

      li = -1; lj =  1; lk = -1;
      z[3] = zt + lk*dx; y[3] = yt + lj*dx; x[3] = xt + li*dx;
      rc = findindex(x[3],y[3],z[3],val3,xindex[3],yindex[3],zindex[3],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[3].phi[0][ll] = val3->value_[xindex[3]+n*(yindex[3]+n*zindex[3])].phi[0][ll];
        }
      } else {
        interp3d(x[3],y[3],z[3],val3,work[3],par);
      }

      li = -1; lj = -1; lk =  1;
      z[4] = zt + lk*dx; y[4] = yt + lj*dx; x[4] = xt + li*dx;
      rc = findindex(x[4],y[4],z[4],val4,xindex[4],yindex[4],zindex[4],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[4].phi[0][ll] = val4->value_[xindex[4]+n*(yindex[4]+n*zindex[4])].phi[0][ll];
        }
      } else {
        interp3d(x[4],y[4],z[4],val4,work[4],par);
      }

      li =  1; lj = -1; lk =  1;
      z[5] = zt + lk*dx; y[5] = yt + lj*dx; x[5] = xt + li*dx;
      rc = findindex(x[5],y[5],z[5],val5,xindex[5],yindex[5],zindex[5],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[5].phi[0][ll] = val5->value_[xindex[5]+n*(yindex[5]+n*zindex[5])].phi[0][ll];
        }
      } else {
        interp3d(x[5],y[5],z[5],val5,work[5],par);
      }

      li =  1; lj =  1; lk =  1;
      z[6] = zt + lk*dx; y[6] = yt + lj*dx; x[6] = xt + li*dx;
      rc = findindex(x[6],y[6],z[6],val6,xindex[6],yindex[6],zindex[6],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[6].phi[0][ll] = val6->value_[xindex[6]+n*(yindex[6]+n*zindex[6])].phi[0][ll];
        }
      } else {
        interp3d(x[6],y[6],z[6],val6,work[6],par);
      }

      li = -1; lj =  1; lk =  1;
      z[7] = zt + lk*dx; y[7] = yt + lj*dx; x[7] = xt + li*dx;
      rc = findindex(x[7],y[7],z[7],val7,xindex[7],yindex[7],zindex[7],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[7].phi[0][ll] = val7->value_[xindex[7]+n*(yindex[7]+n*zindex[7])].phi[0][ll];
        }
      } else {
        interp3d(x[7],y[7],z[7],val7,work[7],par);
      }
      // }}}

      double_type tmp2[2][2][HPX_SMP_AMR3D_NUM_EQUATIONS];
      double_type tmp3[2][HPX_SMP_AMR3D_NUM_EQUATIONS];

      // interp x
      for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
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
      for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
        tmp3[0][ll] = interp_linear(tmp2[0][0][ll],tmp2[1][0][ll],yt,
                                     y[0],y[2]);

        tmp3[1][ll] = interp_linear(tmp2[0][1][ll],tmp2[1][1][ll],yt,
                                     y[4],y[6]);
      }

      // interp z
      for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
        result.phi[0][ll] = interp_linear(tmp3[0][ll],tmp3[1][ll],
                                          zt,
                                          z[0],z[4]);
      } 

      return;
    }
    // }}}

    // special interp 2d xy {{{
    void stencil::special_interp2d_xy(double_type &xt,double_type &yt,double_type &zt,double_type &dx,
                                      access_memory_block<stencil_data> &val0, 
                                      access_memory_block<stencil_data> &val1, 
                                      access_memory_block<stencil_data> &val2, 
                                      access_memory_block<stencil_data> &val3, 
                                      nodedata &result, parameter const& par) {

      // we know the patch that has our data; now we need to find the index of the anchor we need inside the patch
      double_type x[4],y[4],z[4];  
      int li,lj,lk;
//       int i,j,k;

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
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[0].phi[0][ll] = val0->value_[xindex[0]+n*(yindex[0]+n*zindex[0])].phi[0][ll];
        }
      } else {
        interp3d(x[0],y[0],z[0],val0,work[0],par);
      }

      li =  1; lj = -1; lk = 0;
      z[1] = zt + lk*dx; y[1] = yt + lj*dx; x[1] = xt + li*dx;
      rc = findindex(x[1],y[1],z[1],val1,xindex[1],yindex[1],zindex[1],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[1].phi[0][ll] = val1->value_[xindex[1]+n*(yindex[1]+n*zindex[1])].phi[0][ll];
        }
      } else {
        interp3d(x[1],y[1],z[1],val1,work[1],par);
      }

      li =  1; lj =  1; lk = 0;
      z[2] = zt + lk*dx; y[2] = yt + lj*dx; x[2] = xt + li*dx;
      rc = findindex(x[2],y[2],z[2],val2,xindex[2],yindex[2],zindex[2],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[2].phi[0][ll] = val2->value_[xindex[2]+n*(yindex[2]+n*zindex[2])].phi[0][ll];
        }
      } else {
        interp3d(x[2],y[2],z[2],val2,work[2],par);
      }

      li = -1; lj =  1; lk = 0;
      z[3] = zt + lk*dx; y[3] = yt + lj*dx; x[3] = xt + li*dx;
      rc = findindex(x[3],y[3],z[3],val3,xindex[3],yindex[3],zindex[3],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[3].phi[0][ll] = val3->value_[xindex[3]+n*(yindex[3]+n*zindex[3])].phi[0][ll];
        }
      } else {
        interp3d(x[3],y[3],z[3],val3,work[3],par);
      }

      double_type tmp3[2][HPX_SMP_AMR3D_NUM_EQUATIONS];

      // TEST
      //std::cout << " TEST BEFORE " << result.phi[0][3]  << std::endl;

      for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
        // interpolate x
        tmp3[0][ll] = interp_linear(work[0].phi[0][ll],
                                    work[1].phi[0][ll], 
                                    xt,
                                    x[0],x[1]);

        tmp3[1][ll] = interp_linear(work[3].phi[0][ll],
                                    work[2].phi[0][ll], 
                                    xt,
                                    x[3],x[2]);

        // interpolate y
        result.phi[0][ll] = interp_linear(tmp3[0][ll],tmp3[1][ll],
                                          yt,
                                          y[0],y[2]);
      }

      // TEST
      //std::cout << " TEST AFTER " << result.phi[0][3] << " input " << work[0].phi[0][3] << " " << work[1].phi[0][3] << " " << work[2].phi[0][3] << " " << work[3].phi[0][3] << " tmp " << tmp3[0][3] << " " << tmp3[1][3] << std::endl;

      return;
    } // }}}

    // special interp 2d xz {{{
    void stencil::special_interp2d_xz(double_type &xt,double_type &yt,double_type &zt,double_type &dx,
                                      access_memory_block<stencil_data> &val0, 
                                      access_memory_block<stencil_data> &val1, 
                                      access_memory_block<stencil_data> &val2, 
                                      access_memory_block<stencil_data> &val3, 
                                      nodedata &result, parameter const& par) {

      // we know the patch that has our data; now we need to find the index of the anchor we need inside the patch
      double_type x[4],y[4],z[4];  
      int li,lj,lk;
//       int i,j,k;

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
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[0].phi[0][ll] = val0->value_[xindex[0]+n*(yindex[0]+n*zindex[0])].phi[0][ll];
        }
      } else {
        interp3d(x[0],y[0],z[0],val0,work[0],par);
      }

      li =  1; lj =  0; lk = -1;
      z[1] = zt + lk*dx; y[1] = yt + lj*dx; x[1] = xt + li*dx;
      rc = findindex(x[1],y[1],z[1],val1,xindex[1],yindex[1],zindex[1],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[1].phi[0][ll] = val1->value_[xindex[1]+n*(yindex[1]+n*zindex[1])].phi[0][ll];
        }
      } else {
        interp3d(x[1],y[1],z[1],val1,work[1],par);
      }

      li = -1; lj =  0; lk = 1;
      z[2] = zt + lk*dx; y[2] = yt + lj*dx; x[2] = xt + li*dx;
      rc = findindex(x[2],y[2],z[2],val2,xindex[2],yindex[2],zindex[2],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[2].phi[0][ll] = val2->value_[xindex[2]+n*(yindex[2]+n*zindex[2])].phi[0][ll];
        }
      } else {
        interp3d(x[2],y[2],z[2],val2,work[2],par);
      }

      li =  1; lj =  0; lk = 1;
      z[3] = zt + lk*dx; y[3] = yt + lj*dx; x[3] = xt + li*dx;
      rc = findindex(x[3],y[3],z[3],val3,xindex[3],yindex[3],zindex[3],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[3].phi[0][ll] = val3->value_[xindex[3]+n*(yindex[3]+n*zindex[3])].phi[0][ll];
        }
      } else {
        interp3d(x[3],y[3],z[3],val3,work[3],par);
      }

      double_type tmp3[2][HPX_SMP_AMR3D_NUM_EQUATIONS];

      for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
        // interpolate x
        tmp3[0][ll] = interp_linear(work[0].phi[0][ll],
                                    work[1].phi[0][ll], 
                                    xt,
                                    x[0],x[1]);

        tmp3[1][ll] = interp_linear(work[2].phi[0][ll],
                                    work[3].phi[0][ll], 
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
    void stencil::special_interp2d_yz(double_type &xt,double_type &yt,double_type &zt,double_type &dx,
                                      access_memory_block<stencil_data> &val0, 
                                      access_memory_block<stencil_data> &val1, 
                                      access_memory_block<stencil_data> &val2, 
                                      access_memory_block<stencil_data> &val3, 
                                      nodedata &result, parameter const& par) {

      // we know the patch that has our data; now we need to find the index of the anchor we need inside the patch
      double_type x[4],y[4],z[4];  
      int li,lj,lk;
//       int i,j,k;

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
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[0].phi[0][ll] = val0->value_[xindex[0]+n*(yindex[0]+n*zindex[0])].phi[0][ll];
        }
      } else {
        interp3d(x[0],y[0],z[0],val0,work[0],par);
      }

      li =  0; lj =  1; lk = -1;
      z[1] = zt + lk*dx; y[1] = yt + lj*dx; x[1] = xt + li*dx;
      rc = findindex(x[1],y[1],z[1],val1,xindex[1],yindex[1],zindex[1],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[1].phi[0][ll] = val1->value_[xindex[1]+n*(yindex[1]+n*zindex[1])].phi[0][ll];
        }
      } else {
        interp3d(x[1],y[1],z[1],val1,work[1],par);
      }

      li =  0; lj =  -1; lk = 1;
      z[2] = zt + lk*dx; y[2] = yt + lj*dx; x[2] = xt + li*dx;
      rc = findindex(x[2],y[2],z[2],val2,xindex[2],yindex[2],zindex[2],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[2].phi[0][ll] = val2->value_[xindex[2]+n*(yindex[2]+n*zindex[2])].phi[0][ll];
        }
      } else {
        interp3d(x[2],y[2],z[2],val2,work[2],par);
      }

      li =  0; lj =  1; lk = 1;
      z[3] = zt + lk*dx; y[3] = yt + lj*dx; x[3] = xt + li*dx;
      rc = findindex(x[3],y[3],z[3],val3,xindex[3],yindex[3],zindex[3],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[3].phi[0][ll] = val3->value_[xindex[3]+n*(yindex[3]+n*zindex[3])].phi[0][ll];
        }
      } else {
        interp3d(x[3],y[3],z[3],val3,work[3],par);
      }

      double_type tmp3[2][HPX_SMP_AMR3D_NUM_EQUATIONS];

      for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
        // interpolate y
        tmp3[0][ll] = interp_linear(work[0].phi[0][ll],
                                    work[1].phi[0][ll], 
                                    yt,
                                    y[0],y[1]);

        tmp3[1][ll] = interp_linear(work[2].phi[0][ll],
                                    work[3].phi[0][ll], 
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
    void stencil::special_interp1d_x(double_type &xt,double_type &yt,double_type &zt,double_type &dx,
                                      access_memory_block<stencil_data> &val0, 
                                      access_memory_block<stencil_data> &val1, 
                                      nodedata &result, parameter const& par) {

      // we know the patch that has our data; now we need to find the index of the anchor we need inside the patch
      double_type x[2],y[2],z[2];  
      int li,lj,lk;
//       int i,j,k;

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
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[0].phi[0][ll] = val0->value_[xindex[0]+n*(yindex[0]+n*zindex[0])].phi[0][ll];
        }
      } else {
        interp3d(x[0],y[0],z[0],val0,work[0],par);
      }

      li =  1; lj =  0; lk = 0;
      z[1] = zt + lk*dx; y[1] = yt + lj*dx; x[1] = xt + li*dx;
      rc = findindex(x[1],y[1],z[1],val1,xindex[1],yindex[1],zindex[1],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[1].phi[0][ll] = val1->value_[xindex[1]+n*(yindex[1]+n*zindex[1])].phi[0][ll];
        }
      } else {
        interp3d(x[1],y[1],z[1],val1,work[1],par);
      }

      for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
        // interpolate x
        result.phi[0][ll] = interp_linear(work[0].phi[0][ll],work[1].phi[0][ll],
                                          xt,
                                          x[0],x[1]);
      }

      return;
    } // }}}

    // special interp 1d y {{{
    void stencil::special_interp1d_y(double_type &xt,double_type &yt,double_type &zt,double_type &dx,
                                      access_memory_block<stencil_data> &val0, 
                                      access_memory_block<stencil_data> &val1, 
                                      nodedata &result, parameter const& par) {

      // we know the patch that has our data; now we need to find the index of the anchor we need inside the patch
      double_type x[2],y[2],z[2];  
      int li,lj,lk;
//       int i,j,k;

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
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[0].phi[0][ll] = val0->value_[xindex[0]+n*(yindex[0]+n*zindex[0])].phi[0][ll];
        }
      } else {
        interp3d(x[0],y[0],z[0],val0,work[0],par);
      }

      li =  0; lj =  1; lk = 0;
      z[1] = zt + lk*dx; y[1] = yt + lj*dx; x[1] = xt + li*dx;
      rc = findindex(x[1],y[1],z[1],val1,xindex[1],yindex[1],zindex[1],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[1].phi[0][ll] = val1->value_[xindex[1]+n*(yindex[1]+n*zindex[1])].phi[0][ll];
        }
      } else {
        interp3d(x[1],y[1],z[1],val1,work[1],par);
      }

      for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
        // interpolate y
        result.phi[0][ll] = interp_linear(work[0].phi[0][ll],work[1].phi[0][ll],
                                          yt,
                                          y[0],y[1]);
      }

      return;
    } // }}}

    // special interp 1d z {{{
    void stencil::special_interp1d_z(double_type &xt,double_type &yt,double_type &zt,double_type &dx,
                                      access_memory_block<stencil_data> &val0, 
                                      access_memory_block<stencil_data> &val1, 
                                      nodedata &result, parameter const& par) {

      // we know the patch that has our data; now we need to find the index of the anchor we need inside the patch
      double_type x[2],y[2],z[2];  
      int li,lj,lk;
//       int i,j,k;

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
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[0].phi[0][ll] = val0->value_[xindex[0]+n*(yindex[0]+n*zindex[0])].phi[0][ll];
        }
      } else {
        interp3d(x[0],y[0],z[0],val0,work[0],par);
      }

      li = 0; lj = 0; lk = 1;
      z[1] = zt + lk*dx; y[1] = yt + lj*dx; x[1] = xt + li*dx;
      rc = findindex(x[1],y[1],z[1],val1,xindex[1],yindex[1],zindex[1],n);
      if ( rc == 0 ) {
        for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
          work[1].phi[0][ll] = val1->value_[xindex[1]+n*(yindex[1]+n*zindex[1])].phi[0][ll];
        }
      } else {
        interp3d(x[1],y[1],z[1],val1,work[1],par);
      }

      for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
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
        parameter const& par)
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
            double_type dx = par->dx0/pow(2.0,level);
            double_type x = par->min[level] + a*dx*par->granularity;
            double_type y = par->min[level] + b*dx*par->granularity;
            double_type z = par->min[level] + c*dx*par->granularity;
            compute_index = -1;
            for (int i=0;i<val.size();++i) {
              // LIFT MATH HERE
              if ( floatcmp(x,val[i]->x_[0]) == 1 && 
                   floatcmp(y,val[i]->y_[0]) == 1 && 
                   floatcmp(z,val[i]->z_[0]) == 1 &&
                   floatcmp(x+dx*(par->granularity-1),val[i]->x_[par->granularity-1]) == 1 && 
                   floatcmp(y+dx*(par->granularity-1),val[i]->y_[par->granularity-1]) == 1 && 
                   floatcmp(z+dx*(par->granularity-1),val[i]->z_[par->granularity-1]) == 1 ) {
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
            resultval->timestep_ = val[compute_index]->timestep_;
            resultval->value_.resize(val[compute_index]->value_.size());
            resultval->level_ = val[compute_index]->level_;
            resultval->max_index_ = val[compute_index]->max_index_;
            resultval->index_ = val[compute_index]->index_;

            // We may be dealing with either restriction or prolongation (both are performed at the same time)
            bool restriction = false;
            bool prolongation = false;
            for (int i=0;i<val.size();++i) {
              if ( resultval->level_ < val[i]->level_ ) restriction = true;
              if ( resultval->level_ > val[i]->level_ ) prolongation = true;
              if ( restriction && prolongation ) break;
            }

            int n = par->granularity;

            if ( prolongation && restriction ) {
              // prolongation and restriction {{{
              // interpolation
//              double_type xmin = val[compute_index]->x_[0];
//              double_type xmax = val[compute_index]->x_[n-1];
//              double_type ymin = val[compute_index]->y_[0];
//              double_type ymax = val[compute_index]->y_[n-1];
//              double_type zmin = val[compute_index]->z_[0];
//              double_type zmax = val[compute_index]->z_[n-1];

              for (int k=0;k<n;++k) {
                double_type zt = resultval->z_[k];
              for (int j=0;j<n;++j) {
                double_type yt = resultval->y_[j];
              for (int i=0;i<n;++i) {
                double_type xt = resultval->x_[i];

                // check if this is a prolongation point
                if ( ( floatcmp_le(xt,par->min[level]+par->gw*dx) && floatcmp_ge(xt,par->min[level]) ) ||
                     ( floatcmp_le(xt,par->max[level])            && floatcmp_ge(xt,par->max[level]-par->gw*dx) ) ||
                     ( floatcmp_le(yt,par->min[level]+par->gw*dx) && floatcmp_ge(yt,par->min[level]) ) ||
                     ( floatcmp_le(yt,par->max[level])            && floatcmp_ge(yt,par->max[level]-par->gw*dx) ) ||
                     ( floatcmp_le(zt,par->min[level]+par->gw*dx) && floatcmp_ge(zt,par->min[level]) ) ||
                     ( floatcmp_le(zt,par->max[level])            && floatcmp_ge(zt,par->max[level]-par->gw*dx) ) 
                   ) {
                  // this is a prolongation point -- overwrite the value with an interpolated value from the coarse mesh
                  bool found = false;
                  for (int ii=0;ii<val.size();++ii) {
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

                  // this is pretty dangerous, given that the anchor_indexs might not be
                  // initialized when they're used as indices for another array
                  int anchor_index[27] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
                  int has_corner[27] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
                  if ( !found ) {
                    // find the interpolating the anchors needed  
                    for (int lk=-1;lk<2;++lk) {
                      double_type zn = zt + lk*dx;
                    for (int lj=-1;lj<2;++lj) {
                      double_type yn = yt + lj*dx;
                    for (int li=-1;li<2;++li) {
                      double_type xn = xt + li*dx;
                      
                      for (int ii=0;ii<val.size();++ii) {
                        if ( floatcmp_ge(xn,val[ii]->x_[0])  && floatcmp_le(xn,val[ii]->x_[n-1]) &&
                             floatcmp_ge(yn,val[ii]->y_[0])  && floatcmp_le(yn,val[ii]->y_[n-1]) &&
                             floatcmp_ge(zn,val[ii]->z_[0])  && floatcmp_le(zn,val[ii]->z_[n-1]) &&
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
                          if (li == 0 && lj == 0 && lk == 0 ) {
                            has_corner[26] = 1;
                            anchor_index[26] = ii;
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
                  }
//#if 0
                  if ( !found ) {
                    std::cout << " PROBLEM: point " << xt << " " << yt << " " << zt << " BBOX : " <<  par->min[level] << " " << par->min[level]+2*par->gw*dx << " " <<  par->max[level] << " " << par->max[level]-2*par->gw*dx << std::endl;
                    std::cout << " Available data: " << std::endl;
                     for (int ii=0;ii<val.size();++ii) {
                       if ( ii != compute_index ) {
                         std::cout << val[ii]->x_[0] << " " << val[ii]->x_[n-1] << std::endl;
                         std::cout << val[ii]->y_[0] << " " << val[ii]->y_[n-1] << std::endl;
                         std::cout << val[ii]->z_[0] << " " << val[ii]->z_[n-1] << std::endl;
                       }
                     }      
                     for (int ii=0;ii<27;++ii) {
                       std::cout << " Has corner : " << ii << " " << has_corner[ii] << std::endl;
                     }      
                            
                    BOOST_ASSERT(false);
                  }
//#endif
                      // Check if this is a restriction point
                } else if ( par->min[level+1]+par->gw*dx < xt && xt < par->max[level+1]-par->gw*dx &&
                     par->min[level+1]+par->gw*dx < yt && yt < par->max[level+1]-par->gw*dx &&
                     par->min[level+1]+par->gw*dx < zt && zt < par->max[level+1]-par->gw*dx ) {
                  int last_time = -1;
                  bool found = false;
                  if ( last_time != -1 ) {
                    // check the bounding box of the finer mesh
                    double_type xmin = val[last_time]->x_[0];                      
                    double_type xmax = val[last_time]->x_[n-1];                      
                    double_type ymin = val[last_time]->y_[0];                      
                    double_type ymax = val[last_time]->y_[n-1];                      
                    double_type zmin = val[last_time]->z_[0];                      
                    double_type zmax = val[last_time]->z_[n-1];                      

                    if ( floatcmp_ge(xt,xmin) && floatcmp_le(xt,xmax) &&
                         floatcmp_ge(yt,ymin) && floatcmp_le(yt,ymax) &&
                         floatcmp_ge(zt,zmin) && floatcmp_le(zt,zmax) ) {
                      found = true;
                    } else {
                      last_time = -1;
                    }
                  }

                  if ( !found ) {
                    for (int ii=0;ii<val.size();++ii) {
                      if ( ii != compute_index ) {
                        // check the bounding box of the finer mesh
                        double_type xmin = val[ii]->x_[0];                      
                        double_type xmax = val[ii]->x_[n-1];                      
                        double_type ymin = val[ii]->y_[0];                      
                        double_type ymax = val[ii]->y_[n-1];                      
                        double_type zmin = val[ii]->z_[0];                      
                        double_type zmax = val[ii]->z_[n-1];                      

                        if ( floatcmp_ge(xt,xmin) && floatcmp_le(xt,xmax) &&
                             floatcmp_ge(yt,ymin) && floatcmp_le(yt,ymax) &&
                             floatcmp_ge(zt,zmin) && floatcmp_le(zt,zmax) ) {
                          found = true;
                          last_time = ii;
                          break;
                        }
                      }
                    }
                  }

                  if ( !found ) {
                    std::cout << " DEBUG coords " << xt << " " << yt << " " << zt <<  std::endl;
                    for (int ii=0;ii<val.size();++ii) {
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
                  for (int ii=0;ii<par->granularity;++ii) {
                    if ( floatcmp(xt,val[last_time]->x_[ii]) == 1 ) aa = ii;
                    if ( floatcmp(yt,val[last_time]->y_[ii]) == 1 ) bb = ii;
                    if ( floatcmp(zt,val[last_time]->z_[ii]) == 1 ) cc = ii;
                    if ( aa != -1 && bb != -1 && cc != -1 ) break;
                  }
                  BOOST_ASSERT(aa != -1); 
                  BOOST_ASSERT(bb != -1); 
                  BOOST_ASSERT(cc != -1); 
                  
                  for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
                    resultval->value_[i+n*(j+n*k)].phi[0][ll] = val[last_time]->value_[aa+n*(bb+n*cc)].phi[0][ll]; 
                  }
                } else {
                  // neither a prolongation nor restriction point -- copy the value
                  for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
                    resultval->value_[i+n*(j+n*k)].phi[0][ll] =  
                          val[compute_index]->value_[i+n*(j+n*k)].phi[0][ll];
                  } 
                } 
              } } }

              // }}}
            } else if ( prolongation && !restriction ) {
              // prolongation {{{
              // interpolation
//              double_type xmin = val[compute_index]->x_[0];
//              double_type xmax = val[compute_index]->x_[n-1];
//              double_type ymin = val[compute_index]->y_[0];
//              double_type ymax = val[compute_index]->y_[n-1];
//              double_type zmin = val[compute_index]->z_[0];
//              double_type zmax = val[compute_index]->z_[n-1];

              for (int k=0;k<n;++k) {
                double_type zt = resultval->z_[k];
              for (int j=0;j<n;++j) {
                double_type yt = resultval->y_[j];
              for (int i=0;i<n;++i) {
                double_type xt = resultval->x_[i];

                // check if this is a prolongation point
                if ( ( floatcmp_le(xt,par->min[level]+par->gw*dx) && floatcmp_ge(xt,par->min[level]) ) ||
                     ( floatcmp_le(xt,par->max[level])            && floatcmp_ge(xt,par->max[level]-par->gw*dx) ) ||
                     ( floatcmp_le(yt,par->min[level]+par->gw*dx) && floatcmp_ge(yt,par->min[level]) ) ||
                     ( floatcmp_le(yt,par->max[level])            && floatcmp_ge(yt,par->max[level]-par->gw*dx) ) ||
                     ( floatcmp_le(zt,par->min[level]+par->gw*dx) && floatcmp_ge(zt,par->min[level]) ) ||
                     ( floatcmp_le(zt,par->max[level])            && floatcmp_ge(zt,par->max[level]-par->gw*dx) ) 
                   ) {
                  // this is a prolongation point -- overwrite the value with an interpolated value from the coarse mesh
                  bool found = false;
                  for (int ii=0;ii<val.size();++ii) {
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

                  // this is pretty dangerous, given that the anchor_indexs might not be
                  // initialized when they're used as indices for another array
                  int anchor_index[27] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
                  int has_corner[27] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
                  if ( !found ) {
                    // find the interpolating the anchors needed  
                    for (int lk=-1;lk<2;++lk) {
                      double_type zn = zt + lk*dx;
                    for (int lj=-1;lj<2;++lj) {
                      double_type yn = yt + lj*dx;
                    for (int li=-1;li<2;++li) {
                      double_type xn = xt + li*dx;
                      
                      for (int ii=0;ii<val.size();++ii) {
                        if ( floatcmp_ge(xn,val[ii]->x_[0])  && floatcmp_le(xn,val[ii]->x_[n-1]) &&
                             floatcmp_ge(yn,val[ii]->y_[0])  && floatcmp_le(yn,val[ii]->y_[n-1]) &&
                             floatcmp_ge(zn,val[ii]->z_[0])  && floatcmp_le(zn,val[ii]->z_[n-1]) &&
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
                          if (li == 0 && lj == 0 && lk == 0 ) {
                            has_corner[26] = 1;
                            anchor_index[26] = ii;
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
                  }
//#if 0
                  if ( !found ) {
                    std::cout << " PROBLEM: point " << xt << " " << yt << " " << zt << " BBOX : " <<  par->min[level] << " " << par->min[level]+2*par->gw*dx << " " <<  par->max[level] << " " << par->max[level]-2*par->gw*dx << std::endl;
                    std::cout << " Available data: " << std::endl;
                     for (int ii=0;ii<val.size();++ii) {
                       if ( ii != compute_index ) {
                         std::cout << val[ii]->x_[0] << " " << val[ii]->x_[n-1] << std::endl;
                         std::cout << val[ii]->y_[0] << " " << val[ii]->y_[n-1] << std::endl;
                         std::cout << val[ii]->z_[0] << " " << val[ii]->z_[n-1] << std::endl;
                       }
                     }      
                     for (int ii=0;ii<27;++ii) {
                       std::cout << " Has corner : " << ii << " " << has_corner[ii] << std::endl;
                     }      
                            
                    BOOST_ASSERT(false);
                  }
//#endif
                } else {
                  // not a prolongation point -- copy values
                  for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
                    resultval->value_[i+n*(j+n*k)].phi[0][ll] =  
                          val[compute_index]->value_[i+n*(j+n*k)].phi[0][ll];
                  } 
                } 

              } } }

              // }}}
            } else if ( restriction && !prolongation ) {
              // restriction {{{
              int last_time = -1;
              bool found = false;
              double_type xt,yt,zt;
              for (int k=0;k<n;++k) {
                zt = resultval->z_[k];
              for (int j=0;j<n;++j) {
                yt = resultval->y_[j];
              for (int i=0;i<n;++i) {
                xt = resultval->x_[i];

                // Check if this is a restriction point -- is it further than gw coarse dx points away from a fine mesh boundary?
                if ( par->min[level+1]+par->gw*dx < xt && xt < par->max[level+1]-par->gw*dx &&
                     par->min[level+1]+par->gw*dx < yt && yt < par->max[level+1]-par->gw*dx &&
                     par->min[level+1]+par->gw*dx < zt && zt < par->max[level+1]-par->gw*dx ) {

                  found = false;
                  if ( last_time != -1 ) {
                    // check the bounding box of the finer mesh
                    double_type xmin = val[last_time]->x_[0];                      
                    double_type xmax = val[last_time]->x_[n-1];                      
                    double_type ymin = val[last_time]->y_[0];                      
                    double_type ymax = val[last_time]->y_[n-1];                      
                    double_type zmin = val[last_time]->z_[0];                      
                    double_type zmax = val[last_time]->z_[n-1];                      

                    if ( floatcmp_ge(xt,xmin) && floatcmp_le(xt,xmax) &&
                         floatcmp_ge(yt,ymin) && floatcmp_le(yt,ymax) &&
                         floatcmp_ge(zt,zmin) && floatcmp_le(zt,zmax) ) {
                      found = true;
                    } else {
                      last_time = -1;
                    }
                  }

                  if ( !found ) {
                    for (int ii=0;ii<val.size();++ii) {
                      if ( ii != compute_index ) {
                        // check the bounding box of the finer mesh
                        double_type xmin = val[ii]->x_[0];                      
                        double_type xmax = val[ii]->x_[n-1];                      
                        double_type ymin = val[ii]->y_[0];                      
                        double_type ymax = val[ii]->y_[n-1];                      
                        double_type zmin = val[ii]->z_[0];                      
                        double_type zmax = val[ii]->z_[n-1];                      

                        if ( floatcmp_ge(xt,xmin) && floatcmp_le(xt,xmax) &&
                             floatcmp_ge(yt,ymin) && floatcmp_le(yt,ymax) &&
                             floatcmp_ge(zt,zmin) && floatcmp_le(zt,zmax) ) {
                          found = true;
                          last_time = ii;
                          break;
                        }
                      }
                    }
                  }

                  if ( !found ) {
                    std::cout << " DEBUG coords " << xt << " " << yt << " " << zt <<  std::endl;
                    for (int ii=0;ii<val.size();++ii) {
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
                  for (int ii=0;ii<par->granularity;++ii) {
                    if ( floatcmp(xt,val[last_time]->x_[ii]) == 1 ) aa = ii;
                    if ( floatcmp(yt,val[last_time]->y_[ii]) == 1 ) bb = ii;
                    if ( floatcmp(zt,val[last_time]->z_[ii]) == 1 ) cc = ii;
                    if ( aa != -1 && bb != -1 && cc != -1 ) break;
                  }
                  BOOST_ASSERT(aa != -1); 
                  BOOST_ASSERT(bb != -1); 
                  BOOST_ASSERT(cc != -1); 
                  
                  for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
                    resultval->value_[i+n*(j+n*k)].phi[0][ll] = val[last_time]->value_[aa+n*(bb+n*cc)].phi[0][ll]; 
                  }
                } else {
                  // This case shouldn't happen ( I would think...)
                  BOOST_ASSERT(false);
                  // not a restriction point -- copy values
                  for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
                    resultval->value_[i+n*(j+n*k)].phi[0][ll] =  
                          val[compute_index]->value_[i+n*(j+n*k)].phi[0][ll];
                  } 
                }
              } } }
              // }}}
            } else {
              // copy over previous values {{{
              for (int k=0;k<n;++k) {
              for (int j=0;j<n;++j) {
              for (int i=0;i<n;++i) {
                for (int ll=0;ll<HPX_SMP_AMR3D_NUM_EQUATIONS;++ll) {
                  resultval->value_[i+n*(j+n*k)].phi[0][ll] =  
                        val[compute_index]->value_[i+n*(j+n*k)].phi[0][ll];
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
            double_type dx = par->dx0/pow(2.0,level);
            double_type x = par->min[level] + a*dx*par->granularity;
            double_type y = par->min[level] + b*dx*par->granularity;
            double_type z = par->min[level] + c*dx*par->granularity;
            compute_index = -1;
            for (int i=0;i<val.size();++i) {
              if ( floatcmp(x,val[i]->x_[0]) == 1 && 
                   floatcmp(y,val[i]->y_[0]) == 1 && 
                   floatcmp(z,val[i]->z_[0]) == 1 ) {
                compute_index = i;
                break;
              }
            }
            if ( compute_index == -1 ) {
              for (int i=0;i<val.size();++i) {
                std::cout << " DEBUG " << val[i]->x_[0] << " " << val[i]->y_[0] << " "<< val[i]->z_[0] << std::endl;
              }
              std::cout << " PROBLEM LOCATING x " << x << " y " << y << " z " << z << " val size " << val.size() << " level " << level << std::endl;
              BOOST_ASSERT(false);
            }
            boundary = true;
          } 

          if ( vecval.size() == 0 ) {
            vecval.resize(par->num_rows);
          }
          if ( vecval[row].size() == 0 ) {
            vecval[row].resize(3*par->granularity * 3*par->granularity * 3*par->granularity);
          }
          std::vector<nodedata>::iterator niter;

          int count_i = 0;
          int count_j = 0;
          if ( boundary ) {
            bbox[0] = 1; bbox[1] = 1; bbox[2] = 1;
            bbox[3] = 1; bbox[4] = 1; bbox[5] = 1;
          }
          for (int i=0;i<val.size();++i) {
            int ii(0),jj(0),kk(0);
            if ( val.size() == 27 ) {
              kk = i/9 - 1;
              jj = count_j - 1;
              ii = count_i - 1;
              ++count_i;
              if ( count_i%3 == 0 ) {
                count_i = 0; 
                ++count_j;
              }
              if ( count_j%3 == 0 ) count_j = 0;
            } else {
              double_type x = val[compute_index]->x_[0];
              double_type y = val[compute_index]->y_[0];
              double_type z = val[compute_index]->z_[0];

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

            static const int grain = par->granularity;
            static const int grainx3 = 3 * grain;
            int count = 0;
            for (niter=val[i]->value_.begin();niter!=val[i]->value_.end();++niter) {
              int tmp_index = count/grain;
              int c = tmp_index/grain;
              int b = tmp_index%grain;
              int a = count - grain*(b+c*grain);

              vecval[row][a+(ii+1)*grain 
                        + grainx3*( 
                             (b+(jj+1)*grain)
                                +grainx3*(c+(kk+1)*grain) )] = &(*niter); 
              ++count;
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

              double_type dt = par->dt0/pow(2.0,level);
              double_type dx = par->dx0/pow(2.0,level); 

              // call rk update 
              int adj_index = 0;
              /*int gft =*/rkupdate(vecval[row],resultval.get_ptr(),
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
                           parameter const& par)
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

