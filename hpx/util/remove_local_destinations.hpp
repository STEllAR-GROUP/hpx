//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_REMOVE_LOCAL_DESTINATIONS_JUL_16_2012_1119AM)
#define HPX_UTIL_REMOVE_LOCAL_DESTINATIONS_JUL_16_2012_1119AM

#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/assert.hpp>

#include <boost/dynamic_bitset.hpp>

#include <cstddef>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    // Returns true if at least one destination has been retained, i.e. one
    // destination is remote. It returns false if all destinations have been
    // local.
    inline std::vector<naming::gid_type>::iterator
    remove_local_destinations(std::vector<naming::gid_type>& gids,
        std::vector<naming::address>& addrs,
        boost::dynamic_bitset<> const& locals)
    {
        HPX_ASSERT(gids.size() == addrs.size());

        std::vector<naming::gid_type>::iterator gids_it = gids.begin();
        std::vector<naming::gid_type>::iterator gids_end = gids.end();
        std::vector<naming::address>::iterator addrs_it = addrs.begin();

        // gids_it = find_if(gids_it, gids_end, pred)
        std::size_t i = 0; //-V707
        for (/**/; gids_it != gids_end; ++gids_it, ++addrs_it)
        {
            if (locals.test(i++))
                break;
        }
        if (gids_it == gids_end)
            return gids_it;

        // gids_next = remove_if(gids_it, gids_end, pred)
        std::vector<naming::gid_type>::iterator gids_next = gids_it;
        std::vector<naming::address>::iterator addrs_next = addrs_it;

        for (++gids_it, ++addrs_it; gids_it != gids_end; ++gids_it, ++addrs_it)
        {
            if (!locals.test(i++))
            {
                *gids_next++ = std::move(*gids_it);
                *addrs_next++ = std::move(*addrs_it);
            }
        }

        return gids_next;
    }
}}

#endif
