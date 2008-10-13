//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This examples illustrates the usage of the generic_component template and 
// related facilities.

#if !defined(HPX_COMPONENTS_GENERIC_NORESULT_OCT_13_2008_0853AM)
#define HPX_COMPONENTS_GENERIC_NORESULT_OCT_13_2008_0853AM

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/server/generic_component.hpp>

// This is the function to wrap into the component. It's purpose is to print
// the floating point number it receives as its argument
HPX_COMPONENT_EXPORT void 
print_number (hpx::threads::thread_self&, hpx::applier::applier&, double arg);

// This has to be placed into a source file (needs to be compiled 
// once). We use generic_component1 here because the function 
// print_number() takes one additional argument. The number of additional
// arguments N needs to be reflected in the name of the generic_componentN.
typedef 
    hpx::components::server::generic_component1<void, double, print_number> 
print_number_wrapper;

#endif

