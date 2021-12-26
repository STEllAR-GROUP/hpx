//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/modules/datastructures.hpp>

#include <hpx/parcelset_base/parcel_interface.hpp>

namespace hpx::traits {

    ///////////////////////////////////////////////////////////////////////////
    // Customization point for retrieving embedded parcel from action
    template <typename Action, typename Enable = void>
    struct action_get_embedded_parcel
    {
        // return a new instance of a serialization filter
        template <typename TransferAction>
        static hpx::optional<parcelset::parcel> call(
            TransferAction const&) noexcept
        {
            return {};    // by default actions don't have an embedded parcel
        }
    };
}    // namespace hpx::traits

#endif
