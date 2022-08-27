//  Copyright (c) 2015-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/parcelset_base/parcelset_base_fwd.hpp>

namespace hpx {

    /// \cond NOINTERN
    using parcel_write_handler_type = parcelset::parcel_write_handler_type;
    /// \endcond

    /// Set the default parcel write handler which is invoked once a parcel has
    /// been sent if no explicit write handler was specified.
    ///
    /// \param f    The new parcel write handler to use from this point on
    ///
    /// \returns The function returns the parcel write handler which was
    ///          installed before this function was called.
    ///
    /// \note If no parcel handler function is registered by the user the
    ///       system will call a default parcel handler function which is not
    ///       performing any actions. However, this default function will
    ///       terminate the application in case of any errors detected during
    ///       preparing or sending the parcel.
    ///
    HPX_EXPORT parcel_write_handler_type set_parcel_write_handler(
        parcel_write_handler_type const& f);
}    // namespace hpx

#endif
