//  Copyright (c) 2009 Matt Anderson
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_PARAMETER_OCT_23_2009_1249PM)
#define HPX_COMPONENTS_PARAMETER_OCT_23_2009_1249PM

#include "had_config.hpp"

#if defined(__cplusplus)
extern "C" {
#endif

struct Par {
      had_double_type lambda;
      int allowedl;
      int loglevel;
      had_double_type output;
      int output_stdout;
      int stencilsize;
      int linearbounds;
      int coarsestencilsize;
      int integrator;
      int nx0;
      int nt0;
      had_double_type minx0;
      had_double_type maxx0;
      had_double_type dx0;
      had_double_type dt0;
};

#if defined(__cplusplus)
}
#endif

#endif

