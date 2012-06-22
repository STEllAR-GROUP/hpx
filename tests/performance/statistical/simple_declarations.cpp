//  Copyright (c)      2012 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//this file simply enumerates a multitude of functions used to define the action
//used in the benchmark.  Which function is used is decided based on input args

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/components.hpp>
#include <hpx/util/high_resolution_timer.hpp>


///////////////////////////////////////////////////////////////////////////////
//forward declarations
template <typename Vector, typename Package>
void run_void(uint64_t number);

///////////////////////////////////////////////////////////////////////////////
void void_thread(){
}

typedef hpx::actions::plain_action0<void_thread> void_action0;
typedef hpx::lcos::packaged_action<void_action0> void_package0;
HPX_REGISTER_PLAIN_ACTION(void_action0);

