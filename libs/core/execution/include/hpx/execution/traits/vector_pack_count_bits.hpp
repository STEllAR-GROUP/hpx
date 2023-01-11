//  Copyright (c) 2016-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstddef>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::parallel::traits {

    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr std::size_t count_bits(
        bool value) noexcept
    {
        return value ? 1 : 0;
    }
}    // namespace hpx::parallel::traits

#if defined(HPX_HAVE_DATAPAR)

#if !defined(__CUDACC__)
#include <hpx/execution/traits/detail/eve/vector_pack_count_bits.hpp>
#include <hpx/execution/traits/detail/simd/vector_pack_count_bits.hpp>
#include <hpx/execution/traits/detail/vc/vector_pack_count_bits.hpp>
#endif

#endif
