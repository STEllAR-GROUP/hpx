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

    ////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    HPX_HOST_DEVICE HPX_FORCEINLINE auto choose(
        datapar::experimental::simd_mask<T, Abi> const& msk,
        datapar::experimental::simd<T, Abi> const& v_true,
        datapar::experimental::simd<T, Abi> const& v_false) noexcept
    {
        return datapar::experimental::choose(msk, v_true, v_false);
    }

    ////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    HPX_HOST_DEVICE HPX_FORCEINLINE void mask_assign(
        datapar::experimental::simd_mask<T, Abi> const& msk,
        datapar::experimental::simd<T, Abi>& v,
        datapar::experimental::simd<T, Abi> const& val) noexcept
    {
        datapar::experimental::mask_assign(msk, v, val);
    }
}    // namespace hpx::parallel::traits

#endif
