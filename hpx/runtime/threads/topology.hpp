////////////////////////////////////////////////////////////////////////////////
//  Copyright (c)      2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c)      2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_E43E0AF0_8A9D_4870_8CC7_E5AD53EF4798)
#define HPX_E43E0AF0_8A9D_4870_8CC7_E5AD53EF4798

#include <hpx/runtime/naming/name.hpp>
#include <hpx/exception.hpp>

#include <boost/thread.hpp>
#include <boost/cstdint.hpp>

namespace hpx { namespace threads
{

struct topology;

inline std::size_t least_significant_bit(boost::uint64_t mask)
{
    if (mask) {
        int c = 0;    // Will count mask's trailing zero bits.

        // Set mask's trailing 0s to 1s and zero rest.
        mask = (mask ^ (mask - 1)) >> 1;
        for (/**/; mask; ++c)
            mask >>= 1;

        return std::size_t(1) << c;
    }
    return std::size_t(1);
}

inline std::size_t least_significant_bit_set(boost::uint64_t mask)
{
    if (mask) {
        std::size_t c = 0;    // Will count mask's trailing zero bits.

        // Set mask's trailing 0s to 1s and zero rest.
        mask = (mask ^ (mask - 1)) >> 1;
        for (/**/; mask; ++c)
            mask >>= 1;

        return c;
    }
    return std::size_t(-1);
}

HPX_EXPORT std::size_t hardware_concurrency();

HPX_EXPORT topology const& get_topology();

}}

#if defined(HPX_HAVE_HWLOC)
    #include <hpx/runtime/threads/policies/hwloc_topology.hpp>
#elif defined(BOOST_WINDOWS)
    #include <hpx/runtime/threads/policies/windows_topology.hpp>
#elif defined(__APPLE__)
    #include <hpx/runtime/threads/policies/macosx_topology.hpp>
#elif defined(__linux__) 
    #include <hpx/runtime/threads/policies/linux_topology.hpp>
#else
    #include <hpx/runtime/threads/policies/noop_topology.hpp>
#endif

#endif // HPX_E43E0AF0_8A9D_4870_8CC7_E5AD53EF4798

