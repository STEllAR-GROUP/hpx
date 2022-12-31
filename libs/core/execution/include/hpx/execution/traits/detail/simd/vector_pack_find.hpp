//  Copyright (c) 2021 Srinivas Yadav
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
    template <typename T, typename Abi>
    HPX_HOST_DEVICE HPX_FORCEINLINE int find_first_of(
        datapar::experimental::simd_mask<T, Abi> const& msk) noexcept
    {
        if (datapar::experimental::any_of(msk))
        {
            return datapar::experimental::find_first_set(msk);
        }
        return -1;
    }
}    // namespace hpx::parallel::traits

#endif
