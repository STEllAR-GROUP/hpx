//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/naming_base/gid_type.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    /// Create arrays of components using their default constructor
    template <typename Component, typename...Ts>
    naming::gid_type create(Ts&&...ts);

    template <typename Component, typename...Ts>
    naming::gid_type create_migrated(naming::gid_type const& gid, void** p,
        Ts&&...ts);

    template <typename Component, typename...Ts>
    std::vector<naming::gid_type> bulk_create(std::size_t count, Ts&&...ts);

    template <typename Component, typename...Ts>
    inline naming::gid_type construct(Ts&&...ts)
    {
        return create<Component>(std::forward<Ts>(ts)...);
    }
}}}


