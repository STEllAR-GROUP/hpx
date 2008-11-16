//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This examples illustrates the usage of the generic_component template and 
// related facilities.

#include <iostream>
#include <hpx/hpx.hpp>
#include <boost/serialization/export.hpp>

#include "generic_component_noresult.hpp"

///////////////////////////////////////////////////////////////////////////////
// This is the function to wrap into the component. It's purpose is to print
// the floating point number it receives as its argument
hpx::threads::thread_state print_number(
    hpx::threads::thread_self&, hpx::applier::applier&, double arg)
{
    std::cout << arg << std::endl;
    return hpx::threads::terminated;
}

HPX_REGISTER_ACTION(print_number_action);

