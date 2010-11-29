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
inline had_double_type derivs(std::vector< nodedata* > const& vecval,
                             int compute_index,int flag, int field,
                             had_double_type dx)
{
  had_double_type result;
  static had_double_type const c_8 = 8.0;
  static had_double_type const c_12 = 12.0;
  static had_double_type const c_1 = 1.0;
  static had_double_type const c_1_12 = c_1/c_12;

  result = (-vecval[compute_index+2]->phi[flag][field]
        +c_8*vecval[compute_index+1]->phi[flag][field]
        -c_8*vecval[compute_index-1]->phi[flag][field]
            +vecval[compute_index-2]->phi[flag][field])*c_1_12/dx;

  return result;
}

void calcrhs(struct nodedata * rhs,
               std::vector< nodedata* > const& vecval,
               std::vector< had_double_type* > const& vecx,
                int flag, had_double_type const& dx, int size,
                int compute_index, Par const& par);

inline had_double_type initial_phi(had_double_type const& x1,had_double_type const& r,Par const& par) 
{
  if ( -x1 <= r && r <= x1 ) {
     return par.id_amp*tanh( r/(par.id_sigma*par.id_sigma));
  } else if ( r >= x1 && r <= par.id_x0+x1 ) {
     return -par.id_amp*tanh( (r-par.id_x0)/(par.id_sigma*par.id_sigma));
  } else if ( r <= -x1 ) {
     return -par.id_amp*tanh( (r-2.*par.id_x0)/(par.id_sigma*par.id_sigma));
  } else if ( r >= par.id_x0 + x1 ) {
     return par.id_amp*tanh( (r-2.*par.id_x0)/(par.id_sigma*par.id_sigma));
  } else {
     BOOST_ASSERT(false);
  }
}

inline had_double_type initial_dphi(had_double_type const& x1,had_double_type const& r,Par const& par) 
{
  static had_double_type const c_0_5 = 0.5;
  static had_double_type const dx = 1.e-10; 
  static had_double_type const c_0_5_inv_dx = c_0_5/dx;
  had_double_type chi_p1 = initial_phi(x1,r+dx,par);
  had_double_type chi_m1 = initial_phi(x1,r-dx,par);
  return c_0_5_inv_dx*(chi_p1 - chi_m1); 
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
    val->value_.resize(par.granularity);

    //number of values per stencil_data
    nodedata node;

    // find out what level we are at
    std::size_t level = -1;
    for (int j=0;j<=par.allowedl;j++) {
      if (item >= par.level_begin[j] && item < par.level_end[j] ) {
        level = j; 
        break;
      }    
    }
    BOOST_ASSERT(level >= 0);

    val->level_= level;
    had_double_type dx = par.dx0/pow(2.0,(int) level);

    had_double_type r_start = par.minx0;
    for (int j=par.allowedl;j>level;j--) {
      r_start += (par.level_end[j]-par.level_begin[j])*par.granularity*par.dx0/pow(2.0,j);
    }
    for (int j=par.level_begin[level];j<item;j++) {
      r_start += dx*par.granularity;
    }

    static had_double_type const c_0 = 0.0;
    static had_double_type const c_1 = 1.0;
    static had_double_type const c_0_5 = 0.5;
    static had_double_type const c_0_25 = 0.25;
    static had_double_type const c_12 = 12.0;
    static had_double_type const c_1_12 = c_1/c_12;

    had_double_type H = sqrt(par.lambda*c_1_12)*par.v*par.v;
    had_double_type invH = c_1/H;

    static had_double_type const x1 = c_0_5*par.id_x0;

    for (int i=0;i<par.granularity;i++) {
      had_double_type r = r_start + i*dx;

      val->x_[i] = r;
   
      node.phi[0][0] = initial_phi(x1,r,par);

      // pi = phi,t
      node.phi[0][1] = c_0;

      node.phi[0][2] = initial_dchi(x1,r,par); // this is the derivative of variable 0

      // a
      node.phi[0][3] = invH;

      // f = a,t
      node.phi[0][4] = invH;

      node.phi[0][5] = c_0; //this is the derivative of variable 4 (which is constant)

      // b
      node.phi[0][6] = invH;

      node.phi[0][7] = invH*(-c_0_5+c_0_25*node.phi[0][2]*node.phi[0][2] + c_0_5*invH*invH*
                        (c_0_25*par.lambda*pow(node.phi[0][0]*node.phi[0][0]-par.v*par.v,2.0) ) );

      node.phi[0][8] = c_0; // this is the derviatvive of variable 6 (which is constant)

      val->value_[i] = node;
    }

    return 1;
}

int rkupdate(std::vector< nodedata* > const& vecval, stencil_data* result, 
  std::vector< had_double_type* > const& vecx, int size,
  int compute_index, 
  had_double_type const& dt, had_double_type const& dx, had_double_type const& timestep,
  int level, Par const& par)
{
  // allocate some temporary arrays for calculating the rhs
  nodedata rhs;
  std::vector<nodedata> work;
  std::vector<nodedata* > pwork;
  work.resize(vecval.size());

  static had_double_type const c_1 = 1.;
  static had_double_type const c_2 = 2.;
  static had_double_type const c_0_75 = 0.75;
  static had_double_type const c_0_5 = 0.5;
  static had_double_type const c_0_25 = 0.25;
  static had_double_type const c_0 = 0.0;

  static had_double_type const c_4_3 = had_double_type(4.)/had_double_type(3.);
  static had_double_type const c_2_3 = had_double_type(2.)/had_double_type(3.);
  static had_double_type const c_1_3 = had_double_type(1.)/had_double_type(3.);

  #ifdef UGLIFY
  had_double_type tmp,tmp2;
  #endif

  // TEST
  //for (int j=0; j<result->granularity; j++) {
  //  for (int i=0; i<num_eqns; i++) {
  //    result->value_[j].phi[0][i] = vecval[j+compute_index]->phi[0][i];
  //  }
  //}

    // -------------------------------------------------------------------------
    // iter 0
    std::size_t start,end;
    if ( compute_index-7 > 0 ) start = compute_index-7;
    else {
      BOOST_ASSERT(false);
    } 

    if ( compute_index+result->granularity+7 < vecval.size() ) end = compute_index+result->granularity+7;
    else {
      BOOST_ASSERT(false);
    }

    for (int j=start;  j<end;j++) {
      calcrhs(&rhs,vecval,vecx,0,dx,size,j,par);
      for (int i=0; i<num_eqns; i++) {
        work[j].phi[0][i] = vecval[j]->phi[0][i];
#ifndef UGLIFY
        work[j].phi[1][i] = vecval[j]->phi[0][i] + rhs.phi[0][i]*dt;
#else
        // uglify
        work[j].phi[1][i] = dt;
        work[j].phi[1][i] *= rhs.phi[0][i];
        work[j].phi[1][i] += vecval[j]->phi[0][i];
#endif
      }
    }

    std::vector<nodedata>::iterator n_iter;
    for (n_iter=work.begin();n_iter!=work.end();++n_iter) pwork.push_back( &(*n_iter) );

  //----------------------------------------------------------------------
  // iter 1
    for (int j=start; j<end; j++) {
      calcrhs(&rhs,pwork,vecx,1,dx,size,j,par);
      for (int i=0; i<num_eqns; i++) {
#ifndef UGLIFY
        work[j].phi[1][i] = c_0_75*work[j].phi[0][i]
                        +c_0_25*work[j].phi[1][i] + c_0_25*rhs.phi[0][i]*dt;
#else
        // uglify
        tmp = dt;
        tmp *= c_0_25;
        tmp *= rhs.phi[0][i];
        work[j].phi[1][i] *= c_0_25;
        work[j].phi[1][i] += tmp;
        tmp = c_0_75;
        tmp *= work[j].phi[0][i];
        work[j].phi[1][i] += tmp;
#endif
      }
    }

  //----------------------------------------------------------------------
  // iter 2
    for (int j=0; j<result->granularity; j++) {
      calcrhs(&rhs,pwork,vecx,1,dx,size,j+compute_index,par);
      for (int i=0; i<num_eqns; i++) {
#ifndef UGLIFY
        result->value_[j].phi[0][i] = c_1_3*work[j+compute_index].phi[0][i]
                        +c_2_3*(work[j+compute_index].phi[1][i] + rhs.phi[0][i]*dt);
#else
        // uglify
        tmp = c_1_3;
        tmp *= work[j+compute_index].phi[0][i];
        result->value_[j].phi[0][i] = dt;
        result->value_[j].phi[0][i] *= rhs.phi[0][i];
        result->value_[j].phi[0][i] += work[j+compute_index].phi[1][i];
        result->value_[j].phi[0][i] *= c_2_3;
        result->value_[j].phi[0][i] += tmp;
#endif
      }
    }

  // timestep update
#ifndef UGLIFY
  result->timestep_ = timestep + 1.0/pow(2.0,level);
#else
  // uglify
  tmp = pow(c_2,level);
  result->timestep_ = c_1;
  result->timestep_ /= tmp;
  result->timestep_ += timestep;
#endif

  return 1;
}

// This is a pointwise calculation: compute the rhs for point result given input values in array phi
void calcrhs(struct nodedata * rhs,
               std::vector< nodedata* > const& vecval,
               std::vector< had_double_type* > const& vecx,
                int flag, had_double_type const& dx, int size,
                int compute_index, Par const& par)
{
  static had_double_type const c_0_25 = 0.25;
  static had_double_type const c_3 = 3.0;
  static had_double_type const c_m3 = -3.0;
  static had_double_type const c_2 = 2.0;

  had_double_type const x = *vecx[compute_index];
  had_double_type const phi = vecval[compute_index]->phi[flag][0];
  had_double_type const Pi  = vecval[compute_index]->phi[flag][1];
  had_double_type const chi = vecval[compute_index]->phi[flag][2];
  had_double_type const a   = vecval[compute_index]->phi[flag][3];
  had_double_type const f   = vecval[compute_index]->phi[flag][4];
  had_double_type const g   = vecval[compute_index]->phi[flag][5];
  had_double_type const b   = vecval[compute_index]->phi[flag][6];
  had_double_type const q   = vecval[compute_index]->phi[flag][7];
  had_double_type const r   = vecval[compute_index]->phi[flag][8];
  had_double_type const VV  = c_0_25*par.lambda*pow(phi*phi-par.v*par.v,2);
  had_double_type const dphiVV  = par.lambda*phi*(phi*phi-par.v*par.v,2);

  had_double_type const dzPi  = derivs(vecval,compute_index,flag,1,dx);
  had_double_type const dzchi = derivs(vecval,compute_index,flag,2,dx);
  had_double_type const dzf   = derivs(vecval,compute_index,flag,4,dx);
  had_double_type const dzg   = derivs(vecval,compute_index,flag,5,dx);
  had_double_type const dzq   = derivs(vecval,compute_index,flag,7,dx);

  rhs->phi[0][0] = Pi;
  rhs->phi[0][1] = -Pi*(f/a +q/b) 
                   +( (c_3*a*g/(b*b) -a*a*r/(b*b*b) )*chi
                   +  (a/b)*(a/b)*dzchi - a*a*dphiVV );
  rhs->phi[0][2] = dzPi;
  rhs->phi[0][3] = f;
  rhs->phi[0][4] = a*( -(f*g)/(a*b)
                  + (c_2*g*g/(b*b) - a*g*r/(b*b*b)
                  +  a*dzg/(b*b) + a*a*VV));
  rhs->phi[0][5] = dzf;
  rhs->phi[0][6] = q;
  rhs->phi[0][7] = b*( -(f*q)/(a*b)
                  +(c_m3*a*g*r/(b*b*b)
                   +c_3*a*dzg/(b*b)
                   +(a*chi/b)*(a*chi/b) + a*a*VV));
  rhs->phi[0][8] = dzq;

}
