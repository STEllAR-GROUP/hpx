////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c)      2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_E43E0AF0_8A9D_4870_8CC7_E5AD53EF4798)
#define HPX_E43E0AF0_8A9D_4870_8CC7_E5AD53EF4798

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/address.hpp>

#include <boost/thread.hpp>
#include <boost/cstdint.hpp>

namespace hpx { namespace threads
{

struct topology
{
    virtual ~topology();

    virtual std::size_t get_numa_node_number(std::size_t num_thread, error_code& ec = throws) const = 0;
    virtual std::size_t get_numa_node_affinity_mask(std::size_t num_thread, bool numa_sensitive, error_code& ec = throws) const = 0;
    virtual std::size_t get_thread_affinity_mask(std::size_t num_thread, bool numa_sensitive, error_code& ec = throws) const = 0;
    virtual void set_thread_affinity(boost::thread& t, std::size_t num_thread, bool numa_sensitive, error_code& ec = throws) const = 0;
    virtual void set_thread_affinity(std::size_t num_thread, bool numa_sensitive, error_code& ec = throws) const = 0;
    virtual std::size_t get_thread_affinity_mask_from_lva(naming::address::address_type, error_code& ec = throws) const = 0;
};

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

}

using threads::hardware_concurrency;

}

#endif // HPX_E43E0AF0_8A9D_4870_8CC7_E5AD53EF4798

