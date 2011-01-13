//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <cmath>

//#include "../amr_c/stencil.hpp"
#include "../amr_c/stencil_data.hpp"
#include "../amr_c/stencil_functions.hpp"
#include "../had_config.hpp"
#include <stdio.h>

#define UGLIFY 1

///////////////////////////////////////////////////////////////////////////////
// windows needs to initialize MPFR in each shared library
#if defined(BOOST_WINDOWS) 

#include "../init_mpfr.hpp"

namespace hpx { namespace components { namespace amr 
{
    // initialize mpreal default precision
    init_mpfr init_;
}}}
#endif

///////////////////////////////////////////////////////////////////////////////
// local functions
inline int floatcmp(had_double_type const& x1, had_double_type const& x2) 
{
  // compare to floating point numbers
  static had_double_type const epsilon = 1.e-8;
  if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
    // the numbers are close enough for coordinate comparison
    return 1;
  } else {
    return 0;
  }
}

void calcrhs(struct nodedata * rhs,
               std::vector< nodedata* > const& vecval,
               std::vector< had_double_type* > const& vecx,
                int flag, had_double_type const& dx, int size,
                bool boundary, int *bbox,int compute_index, Par const& par);

inline had_double_type initial_chi(had_double_type const& r,Par const& par) 
{
  return par.amp*exp( -(r-par.R0)*(r-par.R0)/(par.delta*par.delta) );   
}

inline had_double_type initial_Phi(had_double_type const& r,Par const& par) 
{
  // Phi is the r derivative of chi
  static had_double_type const c_m2 = -2.;
  return par.amp*exp( -(r-par.R0)*(r-par.R0)/(par.delta*par.delta) ) * ( c_m2*(r-par.R0)/(par.delta*par.delta) );
}

///////////////////////////////////////////////////////////////////////////
int generate_initial_data(stencil_data* val, int item, int maxitems, int row,
    Par const& par)
{
    // provide initial data for the given data value 
    val->max_index_ = maxitems;
    val->index_ = item;
    val->timestep_ = 0;
    val->cycle_ = 0;

    val->granularity = par.granularity;
    val->x_.resize(par.granularity);
    val->y_.resize(par.granularity);
    val->z_.resize(par.granularity);
    val->value_.resize(par.granularity*par.granularity*par.granularity);

    //number of values per stencil_data
    nodedata node;

    val->level_= 0;
    had_double_type dx = par.dx0;

    int tmp_index = item/par.nx0;
    int c = tmp_index/par.nx0;
    int b = tmp_index%par.nx0;
    int a = item - par.nx0*(b+c*par.nx0);
    BOOST_ASSERT(item == a + par.nx0*(b+c*par.nx0));

    static had_double_type const c_0 = 0.0;
    static had_double_type const c_7 = 7.0;
    static had_double_type const c_6 = 6.0;
    static had_double_type const c_8 = 8.0;

    for (int i=0;i<par.granularity;i++) {
      had_double_type x = par.minx0 + a*dx*par.granularity + i*dx;
      had_double_type y = par.minx0 + b*dx*par.granularity + i*dx;
      had_double_type z = par.minx0 + c*dx*par.granularity + i*dx;

      val->x_[i] = x;
      val->y_[i] = y;
      val->z_[i] = z;
    }

    for (int k=0;k<par.granularity;k++) {
    for (int j=0;j<par.granularity;j++) {
    for (int i=0;i<par.granularity;i++) {
      had_double_type x = val->x_[i];
      had_double_type y = val->y_[j];
      had_double_type z = val->z_[k];

      had_double_type r = sqrt(x*x+y*y+z*z);

      if ( pow(r-par.R0,2) <= par.delta*par.delta && r > 0 ) {
        had_double_type Phi = par.amp*pow((r-par.R0)*(r-par.R0)
                             -par.delta*par.delta,4)/pow(par.delta,8)/r;
        had_double_type D1Phi = par.amp*pow((r-par.R0)*(r-par.R0)-par.delta*par.delta,3)*
                                 (c_7*r*r-c_6*r*par.R0+par.R0*par.R0+par.delta*par.delta)*
                                 x/(r*r)/pow(par.delta,8);
        had_double_type D2Phi  = par.amp*pow((r-par.R0)*(r-par.R0)-par.delta*par.delta,3)*
                                 (c_7*r*r-c_6*r*par.R0+par.R0*par.R0+par.delta*par.delta)*
                                 y/(r*r)/pow(par.delta,8); 
        had_double_type D3Phi = par.amp*pow((r-par.R0)*(r-par.R0)-par.delta*par.delta,3)*
                                 (c_7*r*r-c_6*r*par.R0+par.R0*par.R0+par.delta*par.delta)*
                                 z/(r*r)/pow(par.delta,8);
        had_double_type D4Phi = par.amp_dot*pow((r-par.R0)*(r-par.R0)-par.delta*par.delta,3)*
                                c_8*(par.R0-r)/pow(par.delta,8)/r; 
  
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

      val->value_[i+par.granularity*(j+k*par.granularity)] = node;
    }}}

    return 1;
}

int rkupdate(nodedata3D const& vecval, stencil_data* result, 
  bool boundary,
  int *bbox, int compute_index, 
  had_double_type const& dt, had_double_type const& dx, had_double_type const& timestep,
  int level, Par const& par)
{
  // allocate some temporary arrays for calculating the rhs
  nodedata rhs;
  std::vector<nodedata> work;
  std::vector<nodedata* > pwork;
  work.resize(vecval.size());

  // -------------------------------------------------------------------------
  // iter 0
    std::size_t start,end;
    start = par.granularity;
    end = 2*par.granularity;

    //Euler for starters
    for (int k=start; k<end;k++) {
    for (int j=start; j<end;j++) {
    for (int i=start; i<end;i++) {
      result->value_[i+par.granularity*(j+k*par.granularity)].phi[0][0] = 
               vecval[i][j][k]->phi[0][0] + vecval[i][j][k]->phi[0][4]*dt;

      result->value_[i+par.granularity*(j+k*par.granularity)].phi[0][1] = 
               vecval[i][j][k]->phi[0][1] 
         - dt*0.5*(vecval[i+1][j][k]->phi[0][4] - vecval[i-1][j][k]->phi[0][4])/dx;

      result->value_[i+par.granularity*(j+k*par.granularity)].phi[0][2] = 
               vecval[i][j][k]->phi[0][2] 
         - dt*0.5*(vecval[i][j+1][k]->phi[0][4] - vecval[i][j+1][k]->phi[0][4])/dx;

      result->value_[i+par.granularity*(j+k*par.granularity)].phi[0][3] = 
               vecval[i][j][k]->phi[0][3] 
         - dt*0.5*(vecval[i][j][k+1]->phi[0][4] - vecval[i][j][k-1]->phi[0][4])/dx;

      result->value_[i+par.granularity*(j+k*par.granularity)].phi[0][4] = 
               vecval[i][j][k]->phi[0][4] 
         - dt*0.5*(vecval[i+1][j][k]->phi[0][1] - vecval[i-1][j][k]->phi[0][1])/dx
         - dt*0.5*(vecval[i][j+1][k]->phi[0][2] - vecval[i][j-1][k]->phi[0][2])/dx
         - dt*0.5*(vecval[i][j][k+1]->phi[0][3] - vecval[i][j][k-1]->phi[0][3])/dx;
    }}}

    result->timestep_ = timestep + 1.0/pow(2.0,level);

  return 1;
}

// This is a pointwise calculation: compute the rhs for point result given input values in array phi
void calcrhs(struct nodedata * rhs,
               std::vector< nodedata* > const& vecval,
               std::vector< had_double_type* > const& vecx,
                int flag, had_double_type const& dx, int size,
                bool boundary, int *bbox,int compute_index, Par const& par)
{
}
