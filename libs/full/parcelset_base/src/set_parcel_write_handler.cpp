//  Copyright (c) 2015-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/parcelset_base/detail/locality_interface_functions.hpp>
#include <hpx/parcelset_base/set_parcel_write_handler.hpp>

namespace hpx {

    parcel_write_handler_type set_parcel_write_handler(
        parcel_write_handler_type const& f)
    {
        return parcelset::detail::set_parcel_write_handler(f);
    }
}    // namespace hpx

#endif
