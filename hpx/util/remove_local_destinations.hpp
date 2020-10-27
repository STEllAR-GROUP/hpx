//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/assert.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/gid_type.hpp>

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

