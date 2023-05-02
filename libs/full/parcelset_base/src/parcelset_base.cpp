//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/parcelset_base/parcelset_base_fwd.hpp>

#include <cstddef>

namespace hpx::parcelset {

    ///////////////////////////////////////////////////////////////////////////
    namespace strings {

        // clang-format off
        inline constexpr char const* const parcelport_background_mode_names[] = {
            "unknown",
            "parcelport_background_mode_flush_buffers",
            "unknown",
            "parcelport_background_mode_send",
            "parcelport_background_mode_receive",
            "unknown",
            "unknown",
            "parcelport_background_mode_all",
        };
        // clang-format on
    }    // namespace strings

    char const* get_parcelport_background_mode_name(
        parcelport_background_mode mode)
    {
        if (mode < parcelport_background_mode::
                       parcelport_background_mode_flush_buffers ||
            mode > parcelport_background_mode::parcelport_background_mode_all)
        {
            return "unknown";
        }

        return strings::parcelport_background_mode_names
            [static_cast<std::size_t>(mode)];
    }
}    // namespace hpx::parcelset
