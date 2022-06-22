//  Copyright (c) 2022 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_STD_EXPERIMENTAL_SIMD)
#include <cstddef>

#include <experimental/simd>

namespace hpx { namespace parallel { namespace traits {
    ////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    HPX_HOST_DEVICE HPX_FORCEINLINE auto choose(
        std::experimental::simd_mask<T, Abi> const& msk,
        std::experimental::simd<T, Abi> const& v_true,
        std::experimental::simd<T, Abi> const& v_false)
    {
        std::experimental::simd<T, Abi> v;
        where(msk, v) = v_true;
        where(!msk, v) = v_false;
        return v;
    }

    ////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    HPX_HOST_DEVICE HPX_FORCEINLINE void mask_assign(
        std::experimental::simd_mask<T, Abi> const& msk,
        std::experimental::simd<T, Abi>& v,
        std::experimental::simd<T, Abi> const& val)
    {
        where(msk, v) = val;
    }
}}}    // namespace hpx::parallel::traits

#endif
