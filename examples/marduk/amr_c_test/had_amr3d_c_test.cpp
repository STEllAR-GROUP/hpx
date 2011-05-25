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

    return 1;
}

int rkupdate(std::vector<nodedata*> const& vecval, stencil_data* result, 
  bool boundary,
  int *bbox, int compute_index, 
  double_type const& dt, double_type const& dx, double_type const& tstep,
  int level, detail::parameter const& par)
{
    // allocate some temporary arrays for calculating the rhs
    nodedata rhs;
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

