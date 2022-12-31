//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)

namespace hpx::parallel::traits {

    ///////////////////////////////////////////////////////////////////////
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr int find_first_of(
        bool msk) noexcept
    {
        return msk ? 0 : -1;
    }
}    // namespace hpx::parallel::traits

#if !defined(__CUDACC__)
#include <hpx/execution/traits/detail/eve/vector_pack_find.hpp>
#include <hpx/execution/traits/detail/simd/vector_pack_find.hpp>
#include <hpx/execution/traits/detail/vc/vector_pack_find.hpp>
#endif

#endif
