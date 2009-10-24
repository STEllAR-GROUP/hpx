//  Copyright (c) 2009 Matt Anderson
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_PARAMETER_OCT_23_2009_1249PM)
#define HPX_COMPONENTS_PARAMETER_OCT_23_2009_1249PM

#if defined(__cplusplus)
extern "C" {
#endif

struct Par {
      double lambda;
      int allowedl;
      int loglevel;
      int stencilsize;
      int nx0;
      int nt0;
      double minx0;
      double maxx0;
      double dx0;
      double dt0;
};

#if defined(__cplusplus)
}
#endif

#endif

