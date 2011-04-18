//  Copyright (c) 2007-2010 Hartmut Kaiser
//                          Matt Anderson 
//  Copyright (c)      2011 Bryce Lelbach
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#if !defined(HPX_COMPONENTS_HAD_CONFIG_FEB_08_2010_0226PM)
#define HPX_COMPONENTS_HAD_CONFIG_FEB_08_2010_0226PM

#if MPFR_FOUND != 0
#include "mpreal.h"
#include "serialize_mpreal.hpp"

//typedef mpfr::mpreal had_double_type;
typedef double had_double_type;
#else
typedef double had_double_type;
#endif
const int num_eqns = 5;
const int maxlevels = 20;

#include <hpx/util/default_vector.hpp>

#endif
