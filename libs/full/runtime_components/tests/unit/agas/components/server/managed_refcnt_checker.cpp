////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/iostream.hpp>
#include <hpx/modules/format.hpp>

#include <cstdint>
#include <sstream>
#include <string>

#include "managed_refcnt_checker.hpp"

namespace hpx { namespace test { namespace server {

    managed_refcnt_checker::~managed_refcnt_checker()
    {
        const std::uint32_t prefix_ = get_locality_id();
        const naming::gid_type this_ = get_base_gid();

        std::ostringstream strm;

        if (!references_.empty())
        {
            hpx::util::format_to(
                strm, "[{1}/{2}]: held references\n", prefix_, this_);

            for (hpx::id_type const& ref : references_)
            {
                strm << "  " << ref << " "
                     << naming::get_management_type_name(
                            ref.get_management_type())
                     << "\n";
            }

            // Flush garbage collection.
            references_.clear();
            agas::garbage_collect();
        }

        if (hpx::invalid_id != target_)
        {
            hpx::util::format_to(
                strm, "[{1}/{2}]: destroying object\n", prefix_, this_);

            hpx::util::format_to(strm, "[{1}/{2}]: triggering flag {3}\n",
                prefix_, this_, target_);

            hpx::trigger_lco_event(target_);
        }

        std::string const str = strm.str();

        if (!str.empty())
            cout << str << std::flush;
    }
}}}    // namespace hpx::test::server

#endif
