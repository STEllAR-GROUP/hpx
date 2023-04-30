//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/type_support/extra_data.hpp>

namespace hpx::serialization::detail {

    struct allow_zero_copy_receive
    {
    };
}    // namespace hpx::serialization::detail

// This is explicitly instantiated to ensure that the id is stable across shared
// libraries.
template <>
struct hpx::util::extra_data_helper<
    hpx::serialization::detail::allow_zero_copy_receive>
{
    HPX_CORE_EXPORT static extra_data_id_type id() noexcept;
    static constexpr void reset(
        serialization::detail::allow_zero_copy_receive*) noexcept
    {
    }
};
