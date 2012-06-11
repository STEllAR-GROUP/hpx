//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2009-2011 Matt Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <cstdio>

#include <boost/scoped_array.hpp>

#include <hpx/hpx.hpp>

#include "../stencil/stencil_data.hpp"
#include "../stencil/stencil_functions.hpp"
#include <examples/adaptive1d/parameter.hpp>

#include <iostream>
#include <fstream>

#if defined(RNPL_FOUND)
#include <sdf.h>
#endif

namespace hpx { namespace components { namespace adaptive1d
{

///////////////////////////////////////////////////////////////////////////
int generate_initial_data(stencil_data* val, int item, int maxitems, int row,
    detail::parameter const& par)
{
    // provide initial data for the given data value
    val->max_index_ = maxitems;
    val->index_ = item;
    val->timestep_ = 0;

    int gi = par.item2gi[item];
    double minx = par.gr_minx[gi];
    double dx = par.gr_h[gi];
    int nx = par.gr_nx[gi];

    val->value_.resize(nx);
    for (std::size_t i=0;i<val->value_.size();i++) {
      double x = minx + i*dx;
      val->value_[i].x = x;

      double x1 = 0.5*par.x0;

      double H = sqrt(par.lambda/12.0)*par.v*par.v;
      double invH = 1.0/H;
      double u1;
      double dx_u1;
      if ( -x1 <= x && x <= x1 ) {
        u1 = par.amp*tanh( x/(par.id_sigma*par.id_sigma) );
        dx_u1 = par.amp*(1.0-pow(tanh(x/(par.id_sigma*par.id_sigma)),2))/
                                  (par.id_sigma*par.id_sigma);
      } else if ( x >= x1 && x <= par.x0 + x1 ) {
        u1 = -par.amp*tanh( (x-par.x0)/(par.id_sigma*par.id_sigma) );
        dx_u1 = -par.amp*(1.0-pow(tanh( (x-par.x0)/(par.id_sigma*par.id_sigma)),2))/
                                  (par.id_sigma*par.id_sigma);
      } else if ( x <= -x1 ) {
        u1 = -par.amp*tanh( (x-2*par.x0)/(par.id_sigma*par.id_sigma) );
        dx_u1 = -par.amp*(1.0-pow(tanh( (x-2*par.x0)/(par.id_sigma*par.id_sigma)),2))/
                                  (par.id_sigma*par.id_sigma);
      } else if ( x >= par.x0 + x1 ) {
        u1 = par.amp*tanh( (x-2*par.x0)/(par.id_sigma*par.id_sigma) );
        dx_u1 = par.amp*(1.0-pow(tanh( (x-2*par.x0)/(par.id_sigma*par.id_sigma)),2))/
                                  (par.id_sigma*par.id_sigma);
      } else {
        // shouldn't happen -- throw an assertion
        u1 = -99.0;
        dx_u1 = -99.0;
        BOOST_ASSERT(false);
      }

      val->value_[i].phi[0][0] = u1;
      val->value_[i].phi[0][1] = 0.0;

      val->value_[i].phi[0][2] = dx_u1;

      val->value_[i].phi[0][3] = invH;
      val->value_[i].phi[0][4] = invH;

      val->value_[i].phi[0][5] = 0.0;

      val->value_[i].phi[0][6] = invH;

      val->value_[i].phi[0][7] = invH*(-0.5 + 0.25*dx_u1*dx_u1 +
                                   0.5*invH*invH*(0.25*par.lambda*pow(u1*u1-par.v*par.v,2)) );

      val->value_[i].phi[0][8] = 0.0;
    }

#if defined(RNPL_FOUND)
    if ( par.out_every > 0 ) {
      // output initial data
      double datatime = 0.0;
      int shape[3];
      shape[0] = val->value_.size();
      char cnames[80] = { "x" };
      char fname[80];
      applier::applier& appl = applier::get_applier();
      naming::id_type this_prefix = appl.get_runtime_support_gid();
      int locality = get_locality_id_from_id( this_prefix );
      std::vector<double> xcoord,value;
      xcoord.resize(val->value_.size());
      value.resize(val->value_.size());
      for (std::size_t j=0;j<NUM_EQUATIONS;j++) {
        sprintf(fname,"%s/%dfield%d",par.outdir.c_str(),locality,(int) j);
        for (std::size_t i=0;i<val->value_.size();i++) {
          xcoord[i] = val->value_[i].x;
          value[i] = val->value_[i].phi[0][j];
        }
        gft_out_full(fname,datatime,shape,cnames,1,&*xcoord.begin(),&*value.begin());
      }
    }
#endif

    return 1;
}

inline void calcrhs(struct nodedata &rhs,
                   double phi,double Pi,
                   double chi,double a,double f,
                   double g, double b, double q,
                   double r, double VV, double dphiVV,
                   double dzphi,double dzPi,double dzchi,
                   double dza,double dzf,
                   double dzg,double dzb,double dzq,double dzr) {

      rhs.phi[0][0] = Pi;
      rhs.phi[0][1] = -Pi*(f/a + q/b)
                      + ((3.0*a*g/(b*b)-(a*a)*r/(b*b*b))*chi
                      +  pow(a/b,2)*dzchi - (a*a)*dphiVV);
      rhs.phi[0][2] = dzPi;
      rhs.phi[0][3] = f;
      rhs.phi[0][4] = a*(-(f*q)/(a*b)
                   + (2*(g*g)/(b*b) - a*g*r/(b*b*b)
                   + a*dzg/(b*b) + (a*a)*VV));
      rhs.phi[0][5] = dzf;
      rhs.phi[0][6] = q;
      rhs.phi[0][7] = b*(-(f*q)/(a*b)
                      + (-3.0*a*g*r/(b*b*b)
                      + 3.0*a*dzg/(b*b)
                      + pow(a*chi/b,2) + a*a*VV));
      rhs.phi[0][8] = dzq;
}

// rkupdate3 {{{
int rkupdate3(std::vector<access_memory_block<stencil_data> > &val,
             double t, detail::parameter const& par)
{
    nodedata rhs;

    int size0 = val[0]->value_.size();
    int size1 = val[1]->value_.size();

    int num_neighbors = par.num_neighbors;

    double phi,Pi,chi,a,f,g,b,q,r,VV,dphiVV;
    double dzphi,dzPi,dzchi,dza,dzf,dzg,dzb,dzq,dzr;

    int num_eqns = NUM_EQUATIONS;

    int worksize = size1 + 4*num_neighbors;
    boost::scoped_array<nodedata> work(new nodedata[worksize]);
    boost::scoped_array<nodedata> work2(new nodedata[worksize]);

    double dt = par.cfl*par.h;
    double odx = 1.0/par.h;

    int input,index;
    int linput,lindex;
    int rinput,rindex;

    static double const c_0_75 = 0.75;
    static double const c_0_25 = 0.25;
    static double const c_2_3 = double(2.)/double(3.);
    static double const c_1_3 = double(1.)/double(3.);

    // ----------------------------------------------------------------------
    // iter 0
    for (int i=-2*num_neighbors;i<size1 + 2*num_neighbors;i++) {
      // Computer derivatives {{{
      if ( i < 0 ) { index = size0+i; input = 0; }
      else if ( i >= size1 ) { index = i-size1; input = 2; }
      else { index = i; input = 1; }

      if ( i-1 < 0 ) { lindex = size0+i-1; linput = 0; }
      else if ( i-1 >= size1 ) { lindex = i-1-size1; linput = 2; }
      else { lindex = i-1; linput = 1; }

      if ( i+1 < 0 ) { rindex = size0+i+1; rinput = 0; }
      else if ( i+1 >= size1 ) { rindex = i+1-size1; rinput = 2; }
      else { rindex = i+1; rinput = 1; }

      phi = val[input]->value_[index].phi[0][0];
      Pi  = val[input]->value_[index].phi[0][1];
      chi = val[input]->value_[index].phi[0][2];
      a   = val[input]->value_[index].phi[0][3];
      f   = val[input]->value_[index].phi[0][4];
      g   = val[input]->value_[index].phi[0][5];
      b   = val[input]->value_[index].phi[0][6];
      q   = val[input]->value_[index].phi[0][7];
      r   = val[input]->value_[index].phi[0][8];
      VV  = 0.25*par.lambda*pow(phi*phi-par.v*par.v,2);
      dphiVV = par.lambda*phi*(phi*phi-par.v*par.v);

      dzphi = 0.5*odx*( val[rinput]->value_[rindex].phi[0][0] -
                    val[linput]->value_[lindex].phi[0][0]);
      dzPi   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][1] -
                    val[linput]->value_[lindex].phi[0][1]);
      dzchi = 0.5*odx*( val[rinput]->value_[rindex].phi[0][2] -
                    val[linput]->value_[lindex].phi[0][2]);
      dza   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][3] -
                    val[linput]->value_[lindex].phi[0][3]);
      dzf   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][4] -
                    val[linput]->value_[lindex].phi[0][4]);
      dzg   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][5] -
                    val[linput]->value_[lindex].phi[0][5]);
      dzb   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][6] -
                    val[linput]->value_[lindex].phi[0][6]);
      dzq   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][7] -
                    val[linput]->value_[lindex].phi[0][7]);
      dzr   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][8] -
                    val[linput]->value_[lindex].phi[0][8]);
      // }}}

      calcrhs(rhs,phi,Pi,chi,a,f,g,b,q,r,VV,dphiVV,dzphi,dzPi,dzchi,dza,dzf,dzg,dzb,dzq,dzr);

      nodedata& nd = work[i+2*num_neighbors];

      nd.phi[0][0] = phi;
      nd.phi[0][1] = Pi;
      nd.phi[0][2] = chi;
      nd.phi[0][3] = a;
      nd.phi[0][4] = f;
      nd.phi[0][5] = g;
      nd.phi[0][6] = b;
      nd.phi[0][7] = q;
      nd.phi[0][8] = r;

      nd.phi[1][0] = phi + rhs.phi[0][0]*dt;
      nd.phi[1][1] = Pi  + rhs.phi[0][1]*dt;
      nd.phi[1][2] = chi + rhs.phi[0][2]*dt;
      nd.phi[1][3] = a   + rhs.phi[0][3]*dt;
      nd.phi[1][4] = f   + rhs.phi[0][4]*dt;
      nd.phi[1][5] = g   + rhs.phi[0][5]*dt;
      nd.phi[1][6] = b   + rhs.phi[0][6]*dt;
      nd.phi[1][7] = q   + rhs.phi[0][7]*dt;
      nd.phi[1][8] = r   + rhs.phi[0][8]*dt;
    }

    // ----------------------------------------------------------------------
    // iter 1
    for (int i=num_neighbors;i<worksize-num_neighbors;i++) {
      // Calculate derivatives {{{
      nodedata& nd = work[i];
      nodedata& lnd = work[i-1];
      nodedata& rnd = work[i+1];
      nodedata& nd2 = work2[i];

      phi = nd.phi[1][0];
      Pi  = nd.phi[1][1];
      chi = nd.phi[1][2];
      a   = nd.phi[1][3];
      f   = nd.phi[1][4];
      g   = nd.phi[1][5];
      b   = nd.phi[1][6];
      q   = nd.phi[1][7];
      r   = nd.phi[1][8];
      VV  = 0.25*par.lambda*pow(phi*phi-par.v*par.v,2);
      dphiVV = par.lambda*phi*(phi*phi-par.v*par.v);

      dzphi = 0.5*odx*( rnd.phi[0][0] - lnd.phi[0][0]);
      dzPi   = 0.5*odx*( rnd.phi[0][1] - lnd.phi[0][1]);
      dzchi = 0.5*odx*( rnd.phi[0][2] - lnd.phi[0][2]);
      dza   = 0.5*odx*( rnd.phi[0][3] - lnd.phi[0][3]);
      dzf   = 0.5*odx*( rnd.phi[0][4] - lnd.phi[0][4]);
      dzg   = 0.5*odx*( rnd.phi[0][5] - lnd.phi[0][5]);
      dzb   = 0.5*odx*( rnd.phi[0][6] - lnd.phi[0][6]);
      dzq   = 0.5*odx*( rnd.phi[0][7] - lnd.phi[0][7]);
      dzr   = 0.5*odx*( rnd.phi[0][8] - lnd.phi[0][8]);
      // }}}

      calcrhs(rhs,phi,Pi,chi,a,f,g,b,q,r,VV,dphiVV,dzphi,dzPi,dzchi,dza,dzf,dzg,dzb,dzq,dzr);

      for (int ll=0;ll<num_eqns;ll++) {
        nd2.phi[1][ll] = c_0_75*nd.phi[0][ll] + c_0_25*nd.phi[1][ll]
                                              + c_0_25*rhs.phi[0][ll]*dt;
      }
    }

    // ----------------------------------------------------------------------
    // iter 2
    for (int i=2*num_neighbors;i<worksize-2*num_neighbors;i++) {
      // Calculate derivatives {{{
      nodedata& nd = work[i];
      nodedata& nd2 = work2[i];
      nodedata& lnd2 = work2[i-1];
      nodedata& rnd2 = work2[i+1];

      phi = nd2.phi[1][0];
      Pi  = nd2.phi[1][1];
      chi = nd2.phi[1][2];
      a   = nd2.phi[1][3];
      f   = nd2.phi[1][4];
      g   = nd2.phi[1][5];
      b   = nd2.phi[1][6];
      q   = nd2.phi[1][7];
      r   = nd2.phi[1][8];
      VV  = 0.25*par.lambda*pow(phi*phi-par.v*par.v,2);
      dphiVV = par.lambda*phi*(phi*phi-par.v*par.v);

      dzphi = 0.5*odx*( rnd2.phi[0][0] - lnd2.phi[0][0]);
      dzPi   = 0.5*odx*( rnd2.phi[0][1] - lnd2.phi[0][1]);
      dzchi = 0.5*odx*( rnd2.phi[0][2] - lnd2.phi[0][2]);
      dza   = 0.5*odx*( rnd2.phi[0][3] - lnd2.phi[0][3]);
      dzf   = 0.5*odx*( rnd2.phi[0][4] - lnd2.phi[0][4]);
      dzg   = 0.5*odx*( rnd2.phi[0][5] - lnd2.phi[0][5]);
      dzb   = 0.5*odx*( rnd2.phi[0][6] - lnd2.phi[0][6]);
      dzq   = 0.5*odx*( rnd2.phi[0][7] - lnd2.phi[0][7]);
      dzr   = 0.5*odx*( rnd2.phi[0][8] - lnd2.phi[0][8]);
      // }}}

      calcrhs(rhs,phi,Pi,chi,a,f,g,b,q,r,VV,dphiVV,dzphi,dzPi,dzchi,dza,dzf,dzg,dzb,dzq,dzr);

      for (int ll=0;ll<num_eqns;ll++) {
        val[3]->value_[i-2*num_neighbors].phi[0][ll] =
                         c_1_3*nd.phi[0][ll]
                       + c_2_3*(nd2.phi[1][ll] + rhs.phi[0][ll]*dt);
      }
    }

#if defined(RNPL_FOUND)
    if ( fmod(val[1]->timestep_,par.out_every) < 1.e-6 ) {
      double datatime = t + dt;
      int shape[3];
      shape[0] = val[3]->value_.size();
      char cnames[80] = { "x" };
      char fname[80];
      applier::applier& appl = applier::get_applier();
      naming::id_type this_prefix = appl.get_runtime_support_gid();
      int locality = get_locality_id_from_id( this_prefix );
      std::vector<double> xcoord,value;
      xcoord.resize(val[3]->value_.size());
      value.resize(val[3]->value_.size());
      for (std::size_t j=0;j<NUM_EQUATIONS;j++) {
        sprintf(fname,"%s/%dfield%d",par.outdir.c_str(),locality,(int) j);
        for (std::size_t i=0;i<val[3]->value_.size();i++) {
          xcoord[i] = val[3]->value_[i].x;
          value[i] = val[3]->value_[i].phi[0][j];
        }
        gft_out_full(fname,datatime,shape,cnames,1,&*xcoord.begin(),&*value.begin());
      }
    }
#endif

    return 1;
}
// }}}

// rkupdate2a {{{
// boundary is on left, right neighbor is val[1]
int rkupdate2a(std::vector<access_memory_block<stencil_data> > &val,
             double t, detail::parameter const& par)
{
    nodedata rhs;

    int size0 = val[0]->value_.size();

    int num_neighbors = par.num_neighbors;

    double phi,Pi,chi,a,f,g,b,q,r,VV,dphiVV;
    double dzphi,dzPi,dzchi,dza,dzf,dzg,dzb,dzq,dzr;

    int num_eqns = NUM_EQUATIONS;

    int worksize = size0 + 2*num_neighbors;
    boost::scoped_array<nodedata> work(new nodedata[worksize]);
    boost::scoped_array<nodedata> work2(new nodedata[worksize]);

    double dt = par.cfl*par.h;
    double odx = 1.0/par.h;

    int input,index;
    int linput,lindex;
    int rinput,rindex;

    static double const c_0_75 = 0.75;
    static double const c_0_25 = 0.25;
    static double const c_2_3 = double(2.)/double(3.);
    static double const c_1_3 = double(1.)/double(3.);

    // ----------------------------------------------------------------------
    // iter 0
    for (int i=num_neighbors;i<size0 + 2*num_neighbors;i++) {
      // Computer derivatives {{{
      if ( i >= size0 ) { index = i-size0; input = 1; }
      else { index = i; input = 0; }

      if ( i-1 >= size0 ) { lindex = i-1-size0; linput = 1; }
      else { lindex = i-1; linput = 0; }

      if ( i+1 >= size0 ) { rindex = i+1-size0; rinput = 1; }
      else { rindex = i+1; rinput = 0; }

      phi = val[input]->value_[index].phi[0][0];
      Pi  = val[input]->value_[index].phi[0][1];
      chi = val[input]->value_[index].phi[0][2];
      a   = val[input]->value_[index].phi[0][3];
      f   = val[input]->value_[index].phi[0][4];
      g   = val[input]->value_[index].phi[0][5];
      b   = val[input]->value_[index].phi[0][6];
      q   = val[input]->value_[index].phi[0][7];
      r   = val[input]->value_[index].phi[0][8];
      VV  = 0.25*par.lambda*pow(phi*phi-par.v*par.v,2);
      dphiVV = par.lambda*phi*(phi*phi-par.v*par.v);

      dzphi = 0.5*odx*( val[rinput]->value_[rindex].phi[0][0] -
                    val[linput]->value_[lindex].phi[0][0]);
      dzPi   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][1] -
                    val[linput]->value_[lindex].phi[0][1]);
      dzchi = 0.5*odx*( val[rinput]->value_[rindex].phi[0][2] -
                    val[linput]->value_[lindex].phi[0][2]);
      dza   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][3] -
                    val[linput]->value_[lindex].phi[0][3]);
      dzf   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][4] -
                    val[linput]->value_[lindex].phi[0][4]);
      dzg   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][5] -
                    val[linput]->value_[lindex].phi[0][5]);
      dzb   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][6] -
                    val[linput]->value_[lindex].phi[0][6]);
      dzq   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][7] -
                    val[linput]->value_[lindex].phi[0][7]);
      dzr   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][8] -
                    val[linput]->value_[lindex].phi[0][8]);
      // }}}

      calcrhs(rhs,phi,Pi,chi,a,f,g,b,q,r,VV,dphiVV,dzphi,dzPi,dzchi,dza,dzf,dzg,dzb,dzq,dzr);

      nodedata& nd = work[i];

      nd.phi[0][0] = phi;
      nd.phi[0][1] = Pi;
      nd.phi[0][2] = chi;
      nd.phi[0][3] = a;
      nd.phi[0][4] = f;
      nd.phi[0][5] = g;
      nd.phi[0][6] = b;
      nd.phi[0][7] = q;
      nd.phi[0][8] = r;

      nd.phi[1][0] = phi + rhs.phi[0][0]*dt;
      nd.phi[1][1] = Pi  + rhs.phi[0][1]*dt;
      nd.phi[1][2] = chi + rhs.phi[0][2]*dt;
      nd.phi[1][3] = a   + rhs.phi[0][3]*dt;
      nd.phi[1][4] = f   + rhs.phi[0][4]*dt;
      nd.phi[1][5] = g   + rhs.phi[0][5]*dt;
      nd.phi[1][6] = b   + rhs.phi[0][6]*dt;
      nd.phi[1][7] = q   + rhs.phi[0][7]*dt;
      nd.phi[1][8] = r   + rhs.phi[0][8]*dt;
    }

    // ----------------------------------------------------------------------
    // iter 1
    for (int i=2*num_neighbors;i<worksize-num_neighbors;i++) {
      // Calculate derivatives {{{
      nodedata& nd = work[i];
      nodedata& lnd = work[i-1];
      nodedata& rnd = work[i+1];
      nodedata& nd2 = work2[i];

      phi = nd.phi[1][0];
      Pi  = nd.phi[1][1];
      chi = nd.phi[1][2];
      a   = nd.phi[1][3];
      f   = nd.phi[1][4];
      g   = nd.phi[1][5];
      b   = nd.phi[1][6];
      q   = nd.phi[1][7];
      r   = nd.phi[1][8];
      VV  = 0.25*par.lambda*pow(phi*phi-par.v*par.v,2);
      dphiVV = par.lambda*phi*(phi*phi-par.v*par.v);

      dzphi = 0.5*odx*( rnd.phi[0][0] - lnd.phi[0][0]);
      dzPi   = 0.5*odx*( rnd.phi[0][1] - lnd.phi[0][1]);
      dzchi = 0.5*odx*( rnd.phi[0][2] - lnd.phi[0][2]);
      dza   = 0.5*odx*( rnd.phi[0][3] - lnd.phi[0][3]);
      dzf   = 0.5*odx*( rnd.phi[0][4] - lnd.phi[0][4]);
      dzg   = 0.5*odx*( rnd.phi[0][5] - lnd.phi[0][5]);
      dzb   = 0.5*odx*( rnd.phi[0][6] - lnd.phi[0][6]);
      dzq   = 0.5*odx*( rnd.phi[0][7] - lnd.phi[0][7]);
      dzr   = 0.5*odx*( rnd.phi[0][8] - lnd.phi[0][8]);
      // }}}

      calcrhs(rhs,phi,Pi,chi,a,f,g,b,q,r,VV,dphiVV,dzphi,dzPi,dzchi,dza,dzf,dzg,dzb,dzq,dzr);

      for (int ll=0;ll<num_eqns;ll++) {
        nd2.phi[1][ll] = c_0_75*nd.phi[0][ll] + c_0_25*nd.phi[1][ll]
                                              + c_0_25*rhs.phi[0][ll]*dt;
      }
    }

    // ----------------------------------------------------------------------
    // iter 2
    for (int i=3*num_neighbors;i<worksize-2*num_neighbors;i++) {
      // Calculate derivatives {{{
      nodedata& nd = work[i];
      nodedata& nd2 = work2[i];
      nodedata& lnd2 = work2[i-1];
      nodedata& rnd2 = work2[i+1];

      phi = nd2.phi[1][0];
      Pi  = nd2.phi[1][1];
      chi = nd2.phi[1][2];
      a   = nd2.phi[1][3];
      f   = nd2.phi[1][4];
      g   = nd2.phi[1][5];
      b   = nd2.phi[1][6];
      q   = nd2.phi[1][7];
      r   = nd2.phi[1][8];
      VV  = 0.25*par.lambda*pow(phi*phi-par.v*par.v,2);
      dphiVV = par.lambda*phi*(phi*phi-par.v*par.v);

      dzphi = 0.5*odx*( rnd2.phi[0][0] - lnd2.phi[0][0]);
      dzPi   = 0.5*odx*( rnd2.phi[0][1] - lnd2.phi[0][1]);
      dzchi = 0.5*odx*( rnd2.phi[0][2] - lnd2.phi[0][2]);
      dza   = 0.5*odx*( rnd2.phi[0][3] - lnd2.phi[0][3]);
      dzf   = 0.5*odx*( rnd2.phi[0][4] - lnd2.phi[0][4]);
      dzg   = 0.5*odx*( rnd2.phi[0][5] - lnd2.phi[0][5]);
      dzb   = 0.5*odx*( rnd2.phi[0][6] - lnd2.phi[0][6]);
      dzq   = 0.5*odx*( rnd2.phi[0][7] - lnd2.phi[0][7]);
      dzr   = 0.5*odx*( rnd2.phi[0][8] - lnd2.phi[0][8]);
      // }}}

      calcrhs(rhs,phi,Pi,chi,a,f,g,b,q,r,VV,dphiVV,dzphi,dzPi,dzchi,dza,dzf,dzg,dzb,dzq,dzr);

      for (int ll=0;ll<num_eqns;ll++) {
        val[2]->value_[i].phi[0][ll] =
                         c_1_3*nd.phi[0][ll]
                       + c_2_3*(nd2.phi[1][ll] + rhs.phi[0][ll]*dt);
      }
    }

#if defined(RNPL_FOUND)
    if ( fmod(val[1]->timestep_,par.out_every) < 1.e-6 ) {
      double datatime = t + dt;
      int shape[3];
      shape[0] = val[2]->value_.size();
      char cnames[80] = { "x" };
      char fname[80];
      applier::applier& appl = applier::get_applier();
      naming::id_type this_prefix = appl.get_runtime_support_gid();
      int locality = get_locality_id_from_id( this_prefix );
      std::vector<double> xcoord,value;
      xcoord.resize(val[2]->value_.size());
      value.resize(val[2]->value_.size());
      for (std::size_t j=0;j<NUM_EQUATIONS;j++) {
        sprintf(fname,"%s/%dfield%d",par.outdir.c_str(),locality,(int) j);
        for (std::size_t i=0;i<val[3]->value_.size();i++) {
          xcoord[i] = val[2]->value_[i].x;
          value[i] = val[2]->value_[i].phi[0][j];
        }
        gft_out_full(fname,datatime,shape,cnames,1,&*xcoord.begin(),&*value.begin());
      }
    }
#endif

    return 1;
}
// }}}

// rkupdate2b {{{
// boundary is on right, left neighbor is val[0]
int rkupdate2b(std::vector<access_memory_block<stencil_data> > &val,
             double t, detail::parameter const& par)
{
    nodedata rhs;

    int size0 = val[0]->value_.size();
    int size1 = val[1]->value_.size();

    int num_neighbors = par.num_neighbors;

    double phi,Pi,chi,a,f,g,b,q,r,VV,dphiVV;
    double dzphi,dzPi,dzchi,dza,dzf,dzg,dzb,dzq,dzr;

    int num_eqns = NUM_EQUATIONS;

    int worksize = size1 + 2*num_neighbors;
    boost::scoped_array<nodedata> work(new nodedata[worksize]);
    boost::scoped_array<nodedata> work2(new nodedata[worksize]);

    double dt = par.cfl*par.h;
    double odx = 1.0/par.h;

    int input,index;
    int linput,lindex;
    int rinput,rindex;

    static double const c_0_75 = 0.75;
    static double const c_0_25 = 0.25;
    static double const c_2_3 = double(2.)/double(3.);
    static double const c_1_3 = double(1.)/double(3.);

    // ----------------------------------------------------------------------
    // iter 0
    for (int i=-2*num_neighbors;i<size1-num_neighbors;i++) {
      // Computer derivatives {{{
      if ( i < 0 ) { index = size0+i; input = 0; }
      else { index = i; input = 1; }

      if ( i-1 < 0 ) { lindex = size0+i-1; linput = 0; }
      else { lindex = i-1; linput = 1; }

      if ( i+1 < 0 ) { rindex = size0+i+1; rinput = 0; }
      else { rindex = i+1; rinput = 1; }

      phi = val[input]->value_[index].phi[0][0];
      Pi  = val[input]->value_[index].phi[0][1];
      chi = val[input]->value_[index].phi[0][2];
      a   = val[input]->value_[index].phi[0][3];
      f   = val[input]->value_[index].phi[0][4];
      g   = val[input]->value_[index].phi[0][5];
      b   = val[input]->value_[index].phi[0][6];
      q   = val[input]->value_[index].phi[0][7];
      r   = val[input]->value_[index].phi[0][8];
      VV  = 0.25*par.lambda*pow(phi*phi-par.v*par.v,2);
      dphiVV = par.lambda*phi*(phi*phi-par.v*par.v);

      dzphi = 0.5*odx*( val[rinput]->value_[rindex].phi[0][0] -
                    val[linput]->value_[lindex].phi[0][0]);
      dzPi   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][1] -
                    val[linput]->value_[lindex].phi[0][1]);
      dzchi = 0.5*odx*( val[rinput]->value_[rindex].phi[0][2] -
                    val[linput]->value_[lindex].phi[0][2]);
      dza   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][3] -
                    val[linput]->value_[lindex].phi[0][3]);
      dzf   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][4] -
                    val[linput]->value_[lindex].phi[0][4]);
      dzg   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][5] -
                    val[linput]->value_[lindex].phi[0][5]);
      dzb   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][6] -
                    val[linput]->value_[lindex].phi[0][6]);
      dzq   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][7] -
                    val[linput]->value_[lindex].phi[0][7]);
      dzr   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][8] -
                    val[linput]->value_[lindex].phi[0][8]);
      // }}}

      calcrhs(rhs,phi,Pi,chi,a,f,g,b,q,r,VV,dphiVV,dzphi,dzPi,dzchi,dza,dzf,dzg,dzb,dzq,dzr);

      nodedata& nd = work[i+2*num_neighbors];

      nd.phi[0][0] = phi;
      nd.phi[0][1] = Pi;
      nd.phi[0][2] = chi;
      nd.phi[0][3] = a;
      nd.phi[0][4] = f;
      nd.phi[0][5] = g;
      nd.phi[0][6] = b;
      nd.phi[0][7] = q;
      nd.phi[0][8] = r;

      nd.phi[1][0] = phi + rhs.phi[0][0]*dt;
      nd.phi[1][1] = Pi  + rhs.phi[0][1]*dt;
      nd.phi[1][2] = chi + rhs.phi[0][2]*dt;
      nd.phi[1][3] = a   + rhs.phi[0][3]*dt;
      nd.phi[1][4] = f   + rhs.phi[0][4]*dt;
      nd.phi[1][5] = g   + rhs.phi[0][5]*dt;
      nd.phi[1][6] = b   + rhs.phi[0][6]*dt;
      nd.phi[1][7] = q   + rhs.phi[0][7]*dt;
      nd.phi[1][8] = r   + rhs.phi[0][8]*dt;
    }

    // ----------------------------------------------------------------------
    // iter 1
    for (int i=num_neighbors;i<worksize-num_neighbors;i++) {
      // Calculate derivatives {{{
      nodedata& nd = work[i];
      nodedata& lnd = work[i-1];
      nodedata& rnd = work[i+1];
      nodedata& nd2 = work2[i];

      phi = nd.phi[1][0];
      Pi  = nd.phi[1][1];
      chi = nd.phi[1][2];
      a   = nd.phi[1][3];
      f   = nd.phi[1][4];
      g   = nd.phi[1][5];
      b   = nd.phi[1][6];
      q   = nd.phi[1][7];
      r   = nd.phi[1][8];
      VV  = 0.25*par.lambda*pow(phi*phi-par.v*par.v,2);
      dphiVV = par.lambda*phi*(phi*phi-par.v*par.v);

      dzphi = 0.5*odx*( rnd.phi[0][0] - lnd.phi[0][0]);
      dzPi   = 0.5*odx*( rnd.phi[0][1] - lnd.phi[0][1]);
      dzchi = 0.5*odx*( rnd.phi[0][2] - lnd.phi[0][2]);
      dza   = 0.5*odx*( rnd.phi[0][3] - lnd.phi[0][3]);
      dzf   = 0.5*odx*( rnd.phi[0][4] - lnd.phi[0][4]);
      dzg   = 0.5*odx*( rnd.phi[0][5] - lnd.phi[0][5]);
      dzb   = 0.5*odx*( rnd.phi[0][6] - lnd.phi[0][6]);
      dzq   = 0.5*odx*( rnd.phi[0][7] - lnd.phi[0][7]);
      dzr   = 0.5*odx*( rnd.phi[0][8] - lnd.phi[0][8]);
      // }}}

      calcrhs(rhs,phi,Pi,chi,a,f,g,b,q,r,VV,dphiVV,dzphi,dzPi,dzchi,dza,dzf,dzg,dzb,dzq,dzr);

      for (int ll=0;ll<num_eqns;ll++) {
        nd2.phi[1][ll] = c_0_75*nd.phi[0][ll] + c_0_25*nd.phi[1][ll]
                                              + c_0_25*rhs.phi[0][ll]*dt;
      }
    }

    // ----------------------------------------------------------------------
    // iter 2
    for (int i=2*num_neighbors;i<worksize-2*num_neighbors;i++) {
      // Calculate derivatives {{{
      nodedata& nd = work[i];
      nodedata& nd2 = work2[i];
      nodedata& lnd2 = work2[i-1];
      nodedata& rnd2 = work2[i+1];

      phi = nd2.phi[1][0];
      Pi  = nd2.phi[1][1];
      chi = nd2.phi[1][2];
      a   = nd2.phi[1][3];
      f   = nd2.phi[1][4];
      g   = nd2.phi[1][5];
      b   = nd2.phi[1][6];
      q   = nd2.phi[1][7];
      r   = nd2.phi[1][8];
      VV  = 0.25*par.lambda*pow(phi*phi-par.v*par.v,2);
      dphiVV = par.lambda*phi*(phi*phi-par.v*par.v);

      dzphi = 0.5*odx*( rnd2.phi[0][0] - lnd2.phi[0][0]);
      dzPi   = 0.5*odx*( rnd2.phi[0][1] - lnd2.phi[0][1]);
      dzchi = 0.5*odx*( rnd2.phi[0][2] - lnd2.phi[0][2]);
      dza   = 0.5*odx*( rnd2.phi[0][3] - lnd2.phi[0][3]);
      dzf   = 0.5*odx*( rnd2.phi[0][4] - lnd2.phi[0][4]);
      dzg   = 0.5*odx*( rnd2.phi[0][5] - lnd2.phi[0][5]);
      dzb   = 0.5*odx*( rnd2.phi[0][6] - lnd2.phi[0][6]);
      dzq   = 0.5*odx*( rnd2.phi[0][7] - lnd2.phi[0][7]);
      dzr   = 0.5*odx*( rnd2.phi[0][8] - lnd2.phi[0][8]);
      // }}}

      calcrhs(rhs,phi,Pi,chi,a,f,g,b,q,r,VV,dphiVV,dzphi,dzPi,dzchi,dza,dzf,dzg,dzb,dzq,dzr);

      for (int ll=0;ll<num_eqns;ll++) {
        val[2]->value_[i-2*num_neighbors].phi[0][ll] =
                         c_1_3*nd.phi[0][ll]
                       + c_2_3*(nd2.phi[1][ll] + rhs.phi[0][ll]*dt);
      }
    }

#if defined(RNPL_FOUND)
    if ( fmod(val[1]->timestep_,par.out_every) < 1.e-6 ) {
      double datatime = t + dt;
      int shape[3];
      shape[0] = val[2]->value_.size();
      char cnames[80] = { "x" };
      char fname[80];
      applier::applier& appl = applier::get_applier();
      naming::id_type this_prefix = appl.get_runtime_support_gid();
      int locality = get_locality_id_from_id( this_prefix );
      std::vector<double> xcoord,value;
      xcoord.resize(val[2]->value_.size());
      value.resize(val[2]->value_.size());
      for (std::size_t j=0;j<NUM_EQUATIONS;j++) {
        sprintf(fname,"%s/%dfield%d",par.outdir.c_str(),locality,(int) j);
        for (std::size_t i=0;i<val[2]->value_.size();i++) {
          xcoord[i] = val[2]->value_[i].x;
          value[i] = val[2]->value_[i].phi[0][j];
        }
        gft_out_full(fname,datatime,shape,cnames,1,&*xcoord.begin(),&*value.begin());
      }
    }
#endif

    return 1;
}
// }}}

// rkupdate1 {{{
int rkupdate1(std::vector<access_memory_block<stencil_data> > &val,
             double t, detail::parameter const& par)
{
    nodedata rhs;

    int size0 = val[0]->value_.size();

    int num_neighbors = par.num_neighbors;

    double phi,Pi,chi,a,f,g,b,q,r,VV,dphiVV;
    double dzphi,dzPi,dzchi,dza,dzf,dzg,dzb,dzq,dzr;

    int num_eqns = NUM_EQUATIONS;

    boost::scoped_array<nodedata> work(new nodedata[size0]);
    boost::scoped_array<nodedata> work2(new nodedata[size0]);

    double dt = par.cfl*par.h;
    double odx = 1.0/par.h;

    int input,index;
    int linput,lindex;
    int rinput,rindex;

    static double const c_0_75 = 0.75;
    static double const c_0_25 = 0.25;
    static double const c_2_3 = double(2.)/double(3.);
    static double const c_1_3 = double(1.)/double(3.);

    // ----------------------------------------------------------------------
    // iter 0
    for (int i=num_neighbors;i<size0-num_neighbors;i++) {
      // Computer derivatives {{{
      input = 0; linput = 0; rinput = 0;
      index = i; lindex = i-1; rindex = i+1;

      phi = val[input]->value_[index].phi[0][0];
      Pi  = val[input]->value_[index].phi[0][1];
      chi = val[input]->value_[index].phi[0][2];
      a   = val[input]->value_[index].phi[0][3];
      f   = val[input]->value_[index].phi[0][4];
      g   = val[input]->value_[index].phi[0][5];
      b   = val[input]->value_[index].phi[0][6];
      q   = val[input]->value_[index].phi[0][7];
      r   = val[input]->value_[index].phi[0][8];
      VV  = 0.25*par.lambda*pow(phi*phi-par.v*par.v,2);
      dphiVV = par.lambda*phi*(phi*phi-par.v*par.v);

      dzphi = 0.5*odx*( val[rinput]->value_[rindex].phi[0][0] -
                    val[linput]->value_[lindex].phi[0][0]);
      dzPi   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][1] -
                    val[linput]->value_[lindex].phi[0][1]);
      dzchi = 0.5*odx*( val[rinput]->value_[rindex].phi[0][2] -
                    val[linput]->value_[lindex].phi[0][2]);
      dza   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][3] -
                    val[linput]->value_[lindex].phi[0][3]);
      dzf   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][4] -
                    val[linput]->value_[lindex].phi[0][4]);
      dzg   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][5] -
                    val[linput]->value_[lindex].phi[0][5]);
      dzb   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][6] -
                    val[linput]->value_[lindex].phi[0][6]);
      dzq   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][7] -
                    val[linput]->value_[lindex].phi[0][7]);
      dzr   = 0.5*odx*( val[rinput]->value_[rindex].phi[0][8] -
                    val[linput]->value_[lindex].phi[0][8]);
      // }}}

      calcrhs(rhs,phi,Pi,chi,a,f,g,b,q,r,VV,dphiVV,dzphi,dzPi,dzchi,dza,dzf,dzg,dzb,dzq,dzr);

      nodedata& nd = work[i];

      nd.phi[0][0] = phi;
      nd.phi[0][1] = Pi;
      nd.phi[0][2] = chi;
      nd.phi[0][3] = a;
      nd.phi[0][4] = f;
      nd.phi[0][5] = g;
      nd.phi[0][6] = b;
      nd.phi[0][7] = q;
      nd.phi[0][8] = r;

      nd.phi[1][0] = phi + rhs.phi[0][0]*dt;
      nd.phi[1][1] = Pi  + rhs.phi[0][1]*dt;
      nd.phi[1][2] = chi + rhs.phi[0][2]*dt;
      nd.phi[1][3] = a   + rhs.phi[0][3]*dt;
      nd.phi[1][4] = f   + rhs.phi[0][4]*dt;
      nd.phi[1][5] = g   + rhs.phi[0][5]*dt;
      nd.phi[1][6] = b   + rhs.phi[0][6]*dt;
      nd.phi[1][7] = q   + rhs.phi[0][7]*dt;
      nd.phi[1][8] = r   + rhs.phi[0][8]*dt;
    }

    // ----------------------------------------------------------------------
    // iter 1
    for (int i=2*num_neighbors;i<size0-2*num_neighbors;i++) {
      // Calculate derivatives {{{
      nodedata& nd = work[i];
      nodedata& lnd = work[i-1];
      nodedata& rnd = work[i+1];
      nodedata& nd2 = work2[i];

      phi = nd.phi[1][0];
      Pi  = nd.phi[1][1];
      chi = nd.phi[1][2];
      a   = nd.phi[1][3];
      f   = nd.phi[1][4];
      g   = nd.phi[1][5];
      b   = nd.phi[1][6];
      q   = nd.phi[1][7];
      r   = nd.phi[1][8];
      VV  = 0.25*par.lambda*pow(phi*phi-par.v*par.v,2);
      dphiVV = par.lambda*phi*(phi*phi-par.v*par.v);

      dzphi = 0.5*odx*( rnd.phi[0][0] - lnd.phi[0][0]);
      dzPi   = 0.5*odx*( rnd.phi[0][1] - lnd.phi[0][1]);
      dzchi = 0.5*odx*( rnd.phi[0][2] - lnd.phi[0][2]);
      dza   = 0.5*odx*( rnd.phi[0][3] - lnd.phi[0][3]);
      dzf   = 0.5*odx*( rnd.phi[0][4] - lnd.phi[0][4]);
      dzg   = 0.5*odx*( rnd.phi[0][5] - lnd.phi[0][5]);
      dzb   = 0.5*odx*( rnd.phi[0][6] - lnd.phi[0][6]);
      dzq   = 0.5*odx*( rnd.phi[0][7] - lnd.phi[0][7]);
      dzr   = 0.5*odx*( rnd.phi[0][8] - lnd.phi[0][8]);
      // }}}

      calcrhs(rhs,phi,Pi,chi,a,f,g,b,q,r,VV,dphiVV,dzphi,dzPi,dzchi,dza,dzf,dzg,dzb,dzq,dzr);

      for (int ll=0;ll<num_eqns;ll++) {
        nd2.phi[1][ll] = c_0_75*nd.phi[0][ll] + c_0_25*nd.phi[1][ll]
                                              + c_0_25*rhs.phi[0][ll]*dt;
      }
    }

    // ----------------------------------------------------------------------
    // iter 2
    for (int i=3*num_neighbors;i<size0-3*num_neighbors;i++) {
      // Calculate derivatives {{{
      nodedata& nd = work[i];
      nodedata& nd2 = work2[i];
      nodedata& lnd2 = work2[i-1];
      nodedata& rnd2 = work2[i+1];

      phi = nd2.phi[1][0];
      Pi  = nd2.phi[1][1];
      chi = nd2.phi[1][2];
      a   = nd2.phi[1][3];
      f   = nd2.phi[1][4];
      g   = nd2.phi[1][5];
      b   = nd2.phi[1][6];
      q   = nd2.phi[1][7];
      r   = nd2.phi[1][8];
      VV  = 0.25*par.lambda*pow(phi*phi-par.v*par.v,2);
      dphiVV = par.lambda*phi*(phi*phi-par.v*par.v);

      dzphi = 0.5*odx*( rnd2.phi[0][0] - lnd2.phi[0][0]);
      dzPi   = 0.5*odx*( rnd2.phi[0][1] - lnd2.phi[0][1]);
      dzchi = 0.5*odx*( rnd2.phi[0][2] - lnd2.phi[0][2]);
      dza   = 0.5*odx*( rnd2.phi[0][3] - lnd2.phi[0][3]);
      dzf   = 0.5*odx*( rnd2.phi[0][4] - lnd2.phi[0][4]);
      dzg   = 0.5*odx*( rnd2.phi[0][5] - lnd2.phi[0][5]);
      dzb   = 0.5*odx*( rnd2.phi[0][6] - lnd2.phi[0][6]);
      dzq   = 0.5*odx*( rnd2.phi[0][7] - lnd2.phi[0][7]);
      dzr   = 0.5*odx*( rnd2.phi[0][8] - lnd2.phi[0][8]);
      // }}}

      calcrhs(rhs,phi,Pi,chi,a,f,g,b,q,r,VV,dphiVV,dzphi,dzPi,dzchi,dza,dzf,dzg,dzb,dzq,dzr);

      for (int ll=0;ll<num_eqns;ll++) {
        val[1]->value_[i].phi[0][ll] =
                         c_1_3*nd.phi[0][ll]
                       + c_2_3*(nd2.phi[1][ll] + rhs.phi[0][ll]*dt);
      }
    }

#if defined(RNPL_FOUND)
    if ( fmod(val[1]->timestep_,par.out_every) < 1.e-6 ) {
      double datatime = t + dt;
      int shape[3];
      shape[0] = val[1]->value_.size();
      char cnames[80] = { "x" };
      char fname[80];
      applier::applier& appl = applier::get_applier();
      naming::id_type this_prefix = appl.get_runtime_support_gid();
      int locality = get_locality_id_from_id( this_prefix );
      std::vector<double> xcoord,value;
      xcoord.resize(val[1]->value_.size());
      value.resize(val[1]->value_.size());
      for (std::size_t j=0;j<NUM_EQUATIONS;j++) {
        sprintf(fname,"%s/%dfield%d",par.outdir.c_str(),locality,(int) j);
        for (std::size_t i=0;i<val[1]->value_.size();i++) {
          xcoord[i] = val[1]->value_[i].x;
          value[i] = val[1]->value_[i].phi[0][j];
        }
        gft_out_full(fname,datatime,shape,cnames,1,&*xcoord.begin(),&*value.begin());
      }
    }
#endif

    return 1;
}
// }}}

}}}

