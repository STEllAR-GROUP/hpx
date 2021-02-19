//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstdint>

namespace hpx {
    namespace naming {

        namespace detail {

            struct HPX_EXPORT id_type_impl;
        }    // namespace detail

        using component_type = std::int32_t;
        using address_type = std::uint64_t;

        HPX_INLINE_CONSTEXPR_VARIABLE std::uint32_t invalid_locality_id =
            ~static_cast<std::uint32_t>(0);

        HPX_INLINE_CONSTEXPR_VARIABLE std::int32_t component_invalid = -1;

        struct HPX_EXPORT address;
        struct HPX_EXPORT gid_type;
        struct HPX_EXPORT id_type;
    }    // namespace naming

    // Pulling important types into the main namespace
    using naming::id_type;
}    // namespace hpx
