//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/runtime/parcelset_fwd.hpp>

#include <system_error>

namespace hpx { namespace parcelset { namespace detail
{
    // The original parcel-sent handler is wrapped to keep the parcel alive
    // until after the data has been reliably sent (which is needed for zero
    // copy serialization).
    void HPX_EXPORT parcel_route_handler(
        std::error_code const& ec,
        parcelset::parcel const& p);
}}}

#endif
