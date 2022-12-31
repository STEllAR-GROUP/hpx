//  Copyright (c) 2022 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)

namespace hpx::parallel::traits {

    ///////////////////////////////////////////////////////////////////////
    template <typename T, typename Reduce>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr T reduce(Reduce, T val) noexcept
    {
        return val;
    }
}    // namespace hpx::parallel::traits

#if !defined(__CUDACC__)
#include <hpx/execution/traits/detail/eve/vector_pack_reduce.hpp>
#include <hpx/execution/traits/detail/simd/vector_pack_reduce.hpp>
#include <hpx/execution/traits/detail/vc/vector_pack_reduce.hpp>

#endif

#endif
