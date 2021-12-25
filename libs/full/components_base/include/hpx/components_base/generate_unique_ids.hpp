//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concurrency/spinlock.hpp>
#include <hpx/naming_base/id_type.hpp>

#include <cstddef>
#include <mutex>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace util {

    /// The unique_id_ranges class is a type responsible for generating
    /// unique ids for components, parcels, threads etc.
    class HPX_EXPORT unique_id_ranges
    {
        using mutex_type = hpx::util::spinlock;

        mutex_type mtx_;

        /// size of the id range returned by command_getidrange
        /// FIXME: is this a policy?
        enum
        {
            range_delta = 0x100000
        };

    public:
        unique_id_ranges()
          : mtx_()
          , lower_(nullptr)
          , upper_(nullptr)
        {
        }

        /// Generate next unique component id
        naming::gid_type get_id(std::size_t count = 1);

        void set_range(
            naming::gid_type const& lower, naming::gid_type const& upper)
        {
            std::lock_guard l(mtx_);
            lower_ = lower;
            upper_ = upper;
        }

    private:
        /// The range of available ids for components
        naming::gid_type lower_;
        naming::gid_type upper_;
    };
}}    // namespace hpx::util

#include <hpx/config/warnings_suffix.hpp>
