////////////////////////////////////////////////////////////////////////////////
//  Copyright (c)      2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime.hpp>

namespace hpx { namespace threads
{

std::size_t hardware_concurrency()
{
    static std::size_t num_of_cores = boost::thread::hardware_concurrency();
    if (0 == num_of_cores)
        return 1; // Assume one core.
    else
        return num_of_cores;
}

topology const& get_topology()
{
    return get_runtime().get_topology();
}

}}

