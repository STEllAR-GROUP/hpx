////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime.hpp>

#if defined(__ANDROID__) && defined(ANDROID)
#include <cpu-features.h>
#endif

namespace hpx { namespace threads
{

std::size_t hardware_concurrency()
{
#if defined(__ANDROID__) && defined(ANDROID)
    static std::size_t num_of_cores = ::android_getCpuCount();
#else
    static std::size_t num_of_cores = boost::thread::hardware_concurrency();
#endif

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

