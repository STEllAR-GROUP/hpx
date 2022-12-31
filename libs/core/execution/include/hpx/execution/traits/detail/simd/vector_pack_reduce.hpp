//  Copyright (c) 2022 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_EXPERIMENTAL_SIMD)

#include <hpx/execution/traits/detail/simd/vector_pack_simd.hpp>

namespace hpx::parallel::traits {

    ///////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi, typename Reduce>
    HPX_HOST_DEVICE HPX_FORCEINLINE T reduce(
        Reduce r, datapar::experimental::simd<T, Abi> const& val) noexcept
    {
        return datapar::experimental::reduce(val, r);
    }
}    // namespace hpx::parallel::traits

#endif
