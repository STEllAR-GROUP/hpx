//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <boost/program_options/variables_map.hpp>

int hpx_main();

// forwarding of mpx_main, if necessary
int hpx_main(boost::program_options::variables_map&)
{
    return hpx_main();
}
