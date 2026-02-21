//  Copyright (c) 2007-2026 Hartmut Kaiser
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

        HPX_CXX_EXPORT using component_type = std::int32_t;
        HPX_CXX_EXPORT using address_type = void*;

        HPX_CXX_EXPORT inline constexpr std::uint32_t invalid_locality_id =
            ~static_cast<std::uint32_t>(0);

        HPX_CXX_EXPORT inline constexpr std::int32_t component_invalid = -1;

        HPX_CXX_EXPORT struct HPX_EXPORT address;
        HPX_CXX_EXPORT struct HPX_EXPORT gid_type;
    }    // namespace naming

    HPX_CXX_EXPORT struct HPX_EXPORT id_type;
}    // namespace hpx
