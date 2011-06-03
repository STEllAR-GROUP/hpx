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
    int level;
    for (int ii=par.allowedl;ii>=0;ii--) {
      if ( item < par.rowsize[ii]) {
        level = ii;
        found = true;
        break;
      }
    }
    if ( !found ) {
      HPX_THROW_IN_CURRENT_FUNC(bad_parameter, "Problem in prep_ports");
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
      node.phi[0][1] = 0.5*( phi_analytic(x+dx,y,z,t,d) - phi_analytic(x-dx,y,z,t,d) );
      node.phi[0][2] = 0.5*( phi_analytic(x,y+dx,z,t,d) - phi_analytic(x,y-dx,z,t,d) );
      node.phi[0][3] = 0.5*( phi_analytic(x,y,z+dx,t,d) - phi_analytic(x,y,z-dx,t,d) );
      node.phi[0][4] = 0.5*( phi_analytic(x,y,z,t-dx,d) - phi_analytic(x,y,z,t+dx,d) );

      // initial error estimate (for driving refinement)
      node.error = phi_analytic(x,y,z,t,d);
   
      val->value_[i + nx*(j+k*ny)] = node;
    } } }

    return 1;
}

int rkupdate(std::vector<nodedata*> const& vecval, stencil_data* result, 
  bool boundary,
  int *bbox, int compute_index, 
  double_type const& dt, double_type const& dx, double_type const& tstep,
  int level, detail::parameter const& par)
{
    // allocate some temporary arrays for calculating the rhs
//     nodedata rhs;
    boost::scoped_array<nodedata> work(new nodedata[vecval.size()]);
    boost::scoped_array<nodedata> work2(new nodedata[vecval.size()]);
    boost::scoped_array<nodedata> work3(new nodedata[vecval.size()]);

    static double_type const c_0_75 = 0.75;
    static double_type const c_0_25 = 0.25;
    static double_type const c_2_3 = double_type(2.)/double_type(3.);
    static double_type const c_1_3 = double_type(1.)/double_type(3.);

    return 1;
}

}}}

