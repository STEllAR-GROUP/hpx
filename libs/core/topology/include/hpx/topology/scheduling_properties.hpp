//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/scheduling_properties.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/topology/topology.hpp>

#include <type_traits>

namespace hpx::execution::experimental {

    ///////////////////////////////////////////////////////////////////////////
    inline constexpr struct get_processing_units_mask_t final
      : hpx::functional::detail::tag_fallback<get_processing_units_mask_t>
    {
    private:
        // simply return machine affinity mask if get_processing_units_mask is
        // not supported
        template <typename Target>
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_invoke(
            get_processing_units_mask_t, Target&&) noexcept
        {
            return hpx::threads::create_topology().get_machine_affinity_mask();
        }
    } get_processing_units_mask{};

    template <>
    struct is_scheduling_property<get_processing_units_mask_t> : std::true_type
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    inline constexpr struct get_cores_mask_t final
      : hpx::functional::detail::tag_fallback<get_cores_mask_t>
    {
    private:
        // simply return machine affinity mask if get_cores_mask is not
        // supported
        template <typename Target>
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_invoke(
            get_cores_mask_t, Target&&) noexcept
        {
            return hpx::threads::create_topology().get_machine_affinity_mask();
        }
    } get_cores_mask{};

    template <>
    struct is_scheduling_property<get_cores_mask_t> : std::true_type
    {
    };
}    // namespace hpx::execution::experimental
