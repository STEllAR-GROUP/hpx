//  Copyright (c) 2021 Srinivas Yadav
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
    ///////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    HPX_HOST_DEVICE HPX_FORCEINLINE int find_first_of(
        std::experimental::simd_mask<T, Abi> const& msk)
    {
        if (std::experimental::any_of(msk))
        {
            return std::experimental::find_first_set(msk);
        }
        return -1;
    }
}}}    // namespace hpx::parallel::traits

#endif
