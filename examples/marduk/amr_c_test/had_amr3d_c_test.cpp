//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2009-2011 Matt Anderson
//  Copyright (c)      2011 Bryce Lelbaach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <cstdio>

#include <boost/scoped_array.hpp>

#include <hpx/hpx.hpp>

#include "../amr_c/stencil_data.hpp"
#include "../amr_c/stencil_functions.hpp"
#include <examples/marduk/parameter.hpp>

namespace hpx { namespace components { namespace amr 
{

///////////////////////////////////////////////////////////////////////////////
// local functions
inline int floatcmp(double_type const& x1, double_type const& x2) 
{
  // compare to floating point numbers
  static double_type const epsilon = 1.e-8;
  if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
    // the numbers are close enough for coordinate comparison
    return 1;
  } else {
    return 0;
  }
}

double_type phi_analytic(double_type x,double_type y,double_type z,double_type t,double_type d)
{
  double_type A = -(x-0.5*d*cos(t))*(x-0.5*d*cos(t)) - (y+0.5*d*sin(t))*(y+0.5*d*sin(t)) - z*z;
  double_type B = -(x+0.5*d*cos(t))*(x+0.5*d*cos(t)) - (y-0.5*d*sin(t))*(y-0.5*d*sin(t)) - z*z;
  double_type Phi = exp(A) + exp(B);
  return Phi;
}

inline void calcrhsA(struct nodedata * rhs,std::vector<access_memory_block<stencil_data> > const&val,
                     std::vector<int> &src, std::vector<int> &vsrc,
                     int lnx,int lny,int lnz,
                     double_type const& dx,double_type const t,
                     double_type const x, double_type const y, double_type const z,
                     int const i, int const j, int const k)
{
  // initial separation
  double_type d = 11.0;
  double eps = 1.e-10;
  double uxx = (phi_analytic(x+eps,y,z,t,d) -2*phi_analytic(x,y,z,t,d) + phi_analytic(x-eps,y,z,t,d) )/(eps*eps); 
  double uyy = (phi_analytic(x,y+eps,z,t,d) -2*phi_analytic(x,y,z,t,d) + phi_analytic(x,y-eps,z,t,d) )/(eps*eps); 
  double uzz = (phi_analytic(x,y,z+eps,t,d) -2*phi_analytic(x,y,z,t,d) + phi_analytic(x,y,z-eps,t,d) )/(eps*eps); 
  double utt = (phi_analytic(x,y,z,t+eps,d) -2*phi_analytic(x,y,z,t,d) + phi_analytic(x,y,z,t-eps,d) )/(eps*eps); 
  double f = utt - uxx - uyy - uzz;

  BOOST_ASSERT( i + lnx*(j+lny*k) < int(vsrc.size()) && i + lnx*(j+lny*k) < int(src.size()) );
  if ( vsrc[i + lnx*(j+lny*k)] != -1 && src[i + lnx*(j+lny*k)] != -1 &&
       vsrc[i-1 + lnx*(j+lny*k)] != -1 && src[i-1 + lnx*(j+lny*k)] != -1 &&
       vsrc[i+1 + lnx*(j+lny*k)] != -1 && src[i+1 + lnx*(j+lny*k)] != -1 &&
       vsrc[i + lnx*(j-1+lny*k)] != -1 && src[i + lnx*(j-1+lny*k)] != -1 &&
       vsrc[i + lnx*(j+1+lny*k)] != -1 && src[i + lnx*(j+1+lny*k)] != -1 &&
       vsrc[i + lnx*(j+lny*(k-1))] != -1 && src[i + lnx*(j+lny*(k-1))] != -1 &&
       vsrc[i + lnx*(j+lny*(k+1))] != -1 && src[i + lnx*(j+lny*(k+1))] != -1 
     ) {
    BOOST_ASSERT( vsrc[i + lnx*(j+lny*k)] < int(val.size()) && 
                  src[i + lnx*(j+lny*k)] < int(val[vsrc[i + lnx*(j+lny*k)] ]->value_.size()) );
    rhs->phi[0][0] = val[vsrc[i + lnx*(j+lny*k)] ]->value_[ src[i + lnx*(j+lny*k)] ].phi[0][4];

    BOOST_ASSERT( vsrc[i+1 + lnx*(j+lny*k)] < int(val.size()) && 
                  src[i+1 + lnx*(j+lny*k)] < int(val[vsrc[i+1 + lnx*(j+lny*k)] ]->value_.size()) );
    BOOST_ASSERT( vsrc[i-1 + lnx*(j+lny*k)] < int(val.size()) && 
                  src[i-1 + lnx*(j+lny*k)] < int(val[vsrc[i-1 + lnx*(j+lny*k)] ]->value_.size()) );
    rhs->phi[0][1] = - 0.5*(
                      val[vsrc[i+1 + lnx*(j+lny*k)] ]->value_[ src[i+1 + lnx*(j+lny*k)] ].phi[0][4]
                    - val[vsrc[i-1 + lnx*(j+lny*k)] ]->value_[ src[i-1 + lnx*(j+lny*k)] ].phi[0][4]
                           )/dx;

    BOOST_ASSERT( vsrc[i + lnx*(j+1+lny*k)] < int(val.size()) && 
                   src[i + lnx*(j+1+lny*k)] < int(val[vsrc[i + lnx*(j+1+lny*k)] ]->value_.size()) );
    BOOST_ASSERT( vsrc[i + lnx*(j-1+lny*k)] < int(val.size()) && 
                   src[i + lnx*(j-1+lny*k)] < int(val[vsrc[i + lnx*(j-1+lny*k)] ]->value_.size()) );
    rhs->phi[0][2] = - 0.5*(
                       val[vsrc[i + lnx*(j+1+lny*k)] ]->value_[ src[i + lnx*(j+1+lny*k)] ].phi[0][4]
                     - val[vsrc[i + lnx*(j-1+lny*k)] ]->value_[ src[i + lnx*(j-1+lny*k)] ].phi[0][4]
                           )/dx;
  
    BOOST_ASSERT( vsrc[i + lnx*(j+lny*(k+1))] < int(val.size()) && 
                   src[i + lnx*(j+lny*(k+1))] < int(val[vsrc[i + lnx*(j+lny*(k+1))] ]->value_.size()) );
    BOOST_ASSERT( vsrc[i + lnx*(j+lny*(k-1))] < int(val.size()) && 
                   src[i + lnx*(j+lny*(k-1))] < int(val[vsrc[i + lnx*(j+lny*(k-1))] ]->value_.size()) );
    rhs->phi[0][3] = - 0.5*(
                       val[vsrc[i + lnx*(j+lny*(k+1))] ]->value_[ src[i + lnx*(j+lny*(k+1))] ].phi[0][4]
                     - val[vsrc[i + lnx*(j+lny*(k-1))] ]->value_[ src[i + lnx*(j+lny*(k-1))] ].phi[0][4]
                           )/dx;

    rhs->phi[0][4] = - 0.5*(
                      val[vsrc[i+1 + lnx*(j+lny*k)] ]->value_[ src[i+1 + lnx*(j+lny*k)] ].phi[0][1]
                    - val[vsrc[i-1 + lnx*(j+lny*k)] ]->value_[ src[i-1 + lnx*(j+lny*k)] ].phi[0][1]
                            )/dx
                      - 0.5*(
                       val[vsrc[i + lnx*(j+1+lny*k)] ]->value_[ src[i + lnx*(j+1+lny*k)] ].phi[0][2]
                     - val[vsrc[i + lnx*(j-1+lny*k)] ]->value_[ src[i + lnx*(j-1+lny*k)] ].phi[0][2]
                            )/dx
                      - 0.5*(
                       val[vsrc[i + lnx*(j+lny*(k+1))] ]->value_[ src[i + lnx*(j+lny*(k+1))] ].phi[0][3]
                     - val[vsrc[i + lnx*(j+lny*(k-1))] ]->value_[ src[i + lnx*(j+lny*(k-1))] ].phi[0][3]
                            )/dx
                     + f;
 
  } else {
    rhs->phi[0][0] = 0.0;
    rhs->phi[0][1] = 0.0;
    rhs->phi[0][2] = 0.0;
    rhs->phi[0][3] = 0.0;
    rhs->phi[0][4] = 0.0;
  }
  return;
}

inline void calcrhsB(struct nodedata * rhs, boost::scoped_array<nodedata> const& work,
                     int lnx,int lny,int lnz,
                     double_type const& dx,double_type const t,
                     double_type const x, double_type const y, double_type const z,
                     int const i, int const j, int const k)
{
  // initial separation
  double_type d = 11.0;
  double eps = 1.e-10;
  double uxx = (phi_analytic(x+eps,y,z,t,d) -2*phi_analytic(x,y,z,t,d) + phi_analytic(x-eps,y,z,t,d) )/(eps*eps); 
  double uyy = (phi_analytic(x,y+eps,z,t,d) -2*phi_analytic(x,y,z,t,d) + phi_analytic(x,y-eps,z,t,d) )/(eps*eps); 
  double uzz = (phi_analytic(x,y,z+eps,t,d) -2*phi_analytic(x,y,z,t,d) + phi_analytic(x,y,z-eps,t,d) )/(eps*eps); 
  double utt = (phi_analytic(x,y,z,t+eps,d) -2*phi_analytic(x,y,z,t,d) + phi_analytic(x,y,z,t-eps,d) )/(eps*eps); 
  double f = utt - uxx - uyy - uzz;

  rhs->phi[0][0] = work[i+lnx*(j+lny*k)].phi[1][4];

  rhs->phi[0][1] = - 0.5*( work[i+1 + lnx*(j+lny*k)].phi[1][4]   - work[i-1 + lnx*(j+lny*k)].phi[1][4])/dx;

  rhs->phi[0][2] = - 0.5*( work[i + lnx*(j+1+lny*k)].phi[1][4]   - work[i + lnx*(j-1+lny*k)].phi[1][4])/dx;

  rhs->phi[0][3] = - 0.5*( work[i + lnx*(j+lny*(k+1))].phi[1][4] - work[i + lnx*(j+lny*(k-1))].phi[1][4])/dx;

  rhs->phi[0][4] = - 0.5*( work[i+1 + lnx*(j+lny*k)].phi[1][1]   - work[i-1 + lnx*(j+lny*k)].phi[1][1])/dx
                   - 0.5*( work[i + lnx*(j+1+lny*k)].phi[1][2]   - work[i + lnx*(j-1+lny*k)].phi[1][2])/dx
                   - 0.5*( work[i + lnx*(j+lny*(k+1))].phi[1][3] - work[i + lnx*(j+lny*(k-1))].phi[1][3])/dx
                     + f;
 
  return;
}

///////////////////////////////////////////////////////////////////////////
int generate_initial_data(stencil_data* val, int item, int maxitems, int row,
    detail::parameter const& par)
{
    // provide initial data for the given data value 
    val->max_index_ = maxitems;
    val->index_ = item;
    val->timestep_ = 0;

    nodedata node;

    int gi = par.item2gi[item];  
    double_type minx = par.gr_minx[gi];
    double_type miny = par.gr_miny[gi];
    double_type minz = par.gr_minz[gi];
    double_type h = par.gr_h[gi];

    int nx = par.gr_nx[gi];
    int ny = par.gr_ny[gi];
    int nz = par.gr_nz[gi];

    bool found = false;
    int level = 0;
    for (int ii=par.allowedl;ii>=0;ii--) {
      if ( item < int(par.rowsize[ii])) {
        level = ii;
        found = true;
        break;
      }
    }
    if ( !found ) {
      HPX_THROW_IN_CURRENT_FUNC(bad_parameter, "marduk: Problem in prep_ports");
    }

    val->level_ = level;

    val->value_.resize(nx*ny*nz);

    // initial separation
    double_type d = 11.0;
    double_type t = 0.0;
    double_type dx = 1.e-10;

    for (int k=0;k<nz;k++) {
      double_type z = minz + k*h;
    for (int j=0;j<ny;j++) {
      double_type y = miny + j*h;
    for (int i=0;i<nx;i++) {
      double_type x = minx + i*h;

      node.phi[0][0] = phi_analytic(x,y,z,t,d);
      node.phi[0][1] = -0.5*( phi_analytic(x+dx,y,z,t,d) - phi_analytic(x-dx,y,z,t,d) );
      node.phi[0][2] = -0.5*( phi_analytic(x,y+dx,z,t,d) - phi_analytic(x,y-dx,z,t,d) );
      node.phi[0][3] = -0.5*( phi_analytic(x,y,z+dx,t,d) - phi_analytic(x,y,z-dx,t,d) );
      node.phi[0][4] = 0.5*( phi_analytic(x,y,z,t-dx,d) - phi_analytic(x,y,z,t+dx,d) );

      // initial error estimate (for driving refinement)
      node.error = phi_analytic(x,y,z,t,d);
   
      val->value_[i + nx*(j+k*ny)] = node;
    } } }

    return 1;
}

int rkupdate(std::vector<access_memory_block<stencil_data> > const&val, 
             stencil_data* result, 
             std::vector<int> &src, std::vector<int> &vsrc,double dt,double dx,double t,
             int nx0, int ny0, int nz0,
             double minx0, double miny0, double minz0,
             detail::parameter const& par)
{
    // allocate some temporary arrays for calculating the rhs
    nodedata rhs;
    int lnx = nx0 + 6;
    int lny = ny0 + 6;
    int lnz = nz0 + 6;
    boost::scoped_array<nodedata> work(new nodedata[lnx*lny*lnz]);
    boost::scoped_array<nodedata> work2(new nodedata[lnx*lny*lnz]);

    int num_eqns = HPX_SMP_AMR3D_NUM_EQUATIONS;

    static double_type const c_0_75 = 0.75;
    static double_type const c_0_25 = 0.25;
    static double_type const c_2_3 = double_type(2.)/double_type(3.);
    static double_type const c_1_3 = double_type(1.)/double_type(3.);

    double x,y,z;

    // -------------------------------------------------------------------------
    // iter 0
    for (int k=1; k<lnz-1;k++) {
      z = minz0 + (k-3)*dx;
    for (int j=1; j<lny-1;j++) {
      y = miny0 + (j-3)*dx;
    for (int i=1; i<lnx-1;i++) {
      x = minx0 + (i-3)*dx;
      calcrhsA(&rhs,val,src,vsrc,lnx,lny,lnz,dx,t,x,y,z,i,j,k);
  
      nodedata& nd = work[i+lnx*(j+lny*k)]; 
      if ( vsrc[i + lnx*(j+lny*k)] != -1 && src[i + lnx*(j+lny*k)] != -1 ) {
        nodedata const & ndvecval = val[ vsrc[i + lnx*(j+lny*k)] ]->value_[ src[i + lnx*(j+lny*k)] ];
        for (int ll=0;ll<num_eqns;ll++) { 
          nd.phi[0][ll] = ndvecval.phi[0][ll];
          nd.phi[1][ll] = ndvecval.phi[0][ll] + rhs.phi[0][ll]*dt;
        }
      } else {
        for (int ll=0;ll<num_eqns;ll++) { 
          nd.phi[0][ll] = 0.0;
          nd.phi[1][ll] = 0.0;
        }
      }
    }}}

    // -------------------------------------------------------------------------
    // iter 1
    for (int k=2; k<lnz-2;k++) {
      z = minz0 + (k-3)*dx;
    for (int j=2; j<lny-2;j++) {
      y = miny0 + (j-3)*dx;
    for (int i=2; i<lnx-2;i++) {
      x = minx0 + (i-3)*dx;
      calcrhsB(&rhs,work,lnx,lny,lnz,dx,t+dt,x,y,z,i,j,k);
      nodedata& nd = work[i+lnx*(j+lny*k)];
      nodedata& nd2 = work2[i+lnx*(j+lny*k)];
      for (int ll=0;ll<num_eqns;ll++) {
        nd2.phi[1][ll] = c_0_75*nd.phi[0][ll] +
                         c_0_25*nd.phi[1][ll] + c_0_25*rhs.phi[0][ll]*dt;
      }
    }}}

    // -------------------------------------------------------------------------
    // iter 2
    for (int k=3; k<lnz-3;k++) {
      z = minz0 + (k-3)*dx;
    for (int j=3; j<lny-3;j++) {
      y = miny0 + (j-3)*dx;
    for (int i=3; i<lnx-3;i++) {
      x = minx0 + (i-3)*dx;
      calcrhsB(&rhs,work2,lnx,lny,lnz,dx,t+0.5*dt,x,y,z,i,j,k);

      nodedata& nd = work[i+lnx*(j+lny*k)];
      nodedata& nd2 = work2[i+lnx*(j+lny*k)];
      nodedata& ndresult = result->value_[i-3+nx0*(j-3+ny0*(k-3))];

      for (int ll=0;ll<num_eqns;ll++) {
        ndresult.phi[0][ll] = c_1_3*nd.phi[0][ll] + c_2_3*(nd2.phi[1][ll] + rhs.phi[0][ll]*dt);
      }
    }}}

    return 1;
}

}}}

