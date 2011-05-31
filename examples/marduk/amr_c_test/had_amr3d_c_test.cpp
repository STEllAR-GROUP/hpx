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

    double_type delta   = 0.5;
    double_type amp     = 0.0;
    double_type amp_dot = 1.0;
    double_type R0      = 1.0;

    static double_type const c_0 = 0.0;
    static double_type const c_7 = 7.0;
    static double_type const c_6 = 6.0;
    static double_type const c_8 = 8.0;
    
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

    for (int k=0;k<nz;k++) {
      double z = minz + k*h;
    for (int j=0;j<ny;j++) {
      double y = miny + j*h;
    for (int i=0;i<nx;i++) {
      double x = minx + i*h;

      double_type r = sqrt(x*x+y*y+z*z);

      if ( pow(r-R0,2) <= delta*delta && r > 0 ) {
        double_type Phi = amp*pow((r-R0)*(r-R0)
                             -delta*delta,4)/pow(delta,8)/r;
        double_type D1Phi = amp*pow((r-R0)*(r-R0)-delta*delta,3)*
                                 (c_7*r*r-c_6*r*R0+R0*R0+delta*delta)*
                                 x/(r*r)/pow(delta,8);
        double_type D2Phi  = amp*pow((r-R0)*(r-R0)-delta*delta,3)*
                                 (c_7*r*r-c_6*r*R0+R0*R0+delta*delta)*
                                 y/(r*r)/pow(delta,8);
        double_type D3Phi = amp*pow((r-R0)*(r-R0)-delta*delta,3)*
                                 (c_7*r*r-c_6*r*R0+R0*R0+delta*delta)*
                                 z/(r*r)/pow(delta,8);
        double_type D4Phi = amp_dot*pow((r-R0)*(r-R0)-delta*delta,3)*
                                c_8*(R0-r)/pow(delta,8)/r;

        node.phi[0][0] = Phi;
        node.phi[0][1] = D1Phi;
        node.phi[0][2] = D2Phi;
        node.phi[0][3] = D3Phi;
        node.phi[0][4] = D4Phi;
      } else {
        node.phi[0][0] = c_0;
        node.phi[0][1] = c_0;
        node.phi[0][2] = c_0;
        node.phi[0][3] = c_0;
        node.phi[0][4] = c_0;
      }
   
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

