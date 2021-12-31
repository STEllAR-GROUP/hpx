//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/functional.hpp>

#include <system_error>

namespace hpx::parcelset {

    class HPX_EXPORT locality;
    class HPX_EXPORT parcel;

    /// The type of a function that can be registered as a parcel write handler
    /// using the function \a hpx::set_parcel_write_handler.
    ///
    /// \note A parcel write handler is a function which is called by the
    ///       parcel layer whenever a parcel has been sent by the underlying
    ///       networking library and if no explicit parcel handler function was
    ///       specified for the parcel.
    using parcel_write_handler_type = util::function_nonser<void(
        std::error_code const&, parcelset::parcel const&)>;
}    // namespace hpx::parcelset
