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

    class HPX_EXPORT parcelport;
    class HPX_EXPORT locality;
    class HPX_EXPORT parcel;

    /// The type of a function that can be registered as a parcel write handler
    /// using the function \a hpx::set_parcel_write_handler.
    ///
    /// \note A parcel write handler is a function which is called by the
    ///       parcel layer whenever a parcel has been sent by the underlying
    ///       networking library and if no explicit parcel handler function was
    ///       specified for the parcel.
    using parcel_write_handler_type =
        hpx::function<void(std::error_code const&, parcelset::parcel const&)>;

    ////////////////////////////////////////////////////////////////////////
    /// Type of background work to perform
    enum parcelport_background_mode
    {
        /// perform buffer flush operations
        parcelport_background_mode_flush_buffers = 0x01,
        /// perform send operations (includes buffer flush)
        parcelport_background_mode_send = 0x03,
        /// perform receive operations
        parcelport_background_mode_receive = 0x04,
        /// perform all operations
        parcelport_background_mode_all = 0x07
    };
}    // namespace hpx::parcelset
