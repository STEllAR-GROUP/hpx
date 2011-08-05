//  Copyright (c) 2007-2011 Hartmut Kaiser
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
    std::size_t grain_size;
    if ( item != maxitems-1 ) {
      grain_size = par.grain_size;
    } else {
      grain_size = par.nx0 - (maxitems-1)*par.grain_size;
    }
    // make sure the local grain size is always at least as big as the
    // user suggested grain size
    BOOST_ASSERT(grain_size >= par.grain_size);

    double dx = par.h;

    val->value_.resize(grain_size);
    for (std::size_t i=0;i<val->value_.size();i++) {
      double x = item*par.grain_size*dx + par.Rmin + i*dx;
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
    // output initial data
    double datatime = 0.0;
    int shape[3];
    shape[0] = val->value_.size();
    char cnames[80] = { "x|y|z" };
    char fname[80];
    applier::applier& appl = applier::get_applier();
    naming::id_type this_prefix = appl.get_runtime_support_gid();
    int locality = get_prefix_from_id( this_prefix );
    std::vector<double> xcoord,value;
    xcoord.resize(val->value_.size());
    value.resize(val->value_.size());
    for (std::size_t j=0;j<NUM_EQUATIONS;j++) {
      sprintf(fname,"%dfield%d",locality,(int) j);
      for (std::size_t i=0;i<val->value_.size();i++) {
        xcoord[i] = val->value_[i].x;
        value[i] = val->value_[i].phi[0][j];
      }
      gft_out_full(fname,datatime,shape,cnames,1,&*xcoord.begin(),&*value.begin()); 
    }
#endif
#if 0
    // Output data as plain text
    char fname[80];
    std::ofstream myfile;
    myfile.precision(10);
    applier::applier& appl = applier::get_applier();
    naming::id_type this_prefix = appl.get_runtime_support_gid();
    int locality = get_prefix_from_id( this_prefix );
    for (std::size_t j=0;j<NUM_EQUATIONS;j++) {
      sprintf(fname,"%dfield%d.txt",locality,(int) j);
      myfile.open(fname,std::fstream::app);
      for (std::size_t i=0;i<val->value_.size();i++) {
        myfile << val->value_[i].x << " " << val->value_[i].phi[0][j] << std::endl;
      }
      myfile.close();
    }

#endif

    return 1;
}

int rkupdate(std::vector<access_memory_block<stencil_data> > &val, 
             double t, detail::parameter const& par)
{

    //boost::scoped_array<nodedata> work(new nodedata[lnx]);
    //boost::scoped_array<nodedata> work2(new nodedata[lnx]);

    return 1;
}

}}}

