//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This examples illustrates the usage of the generic_component template and 
// related facilities.

#if !defined(HPX_COMPONENTS_GENERIC_NORESULT_OCT_13_2008_0853AM)
#define HPX_COMPONENTS_GENERIC_NORESULT_OCT_13_2008_0853AM

#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/plain_action.hpp>

// This is the function to wrap into the component. Its purpose is to print
// the floating point number it receives as its argument
HPX_COMPONENT_EXPORT void print_number(double arg);

typedef hpx::actions::plain_action1<double, print_number> print_number_action;

#endif

