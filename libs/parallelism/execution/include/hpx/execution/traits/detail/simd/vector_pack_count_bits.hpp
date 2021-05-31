//  Copyright (c) 2021 Srinivas Yadav
//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CXX20_EXPERIMENTAL_SIMD)
#include <cstddef>

#include <experimental/simd>

namespace hpx { namespace parallel { namespace traits {
    ///////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    HPX_HOST_DEVICE HPX_FORCEINLINE std::size_t count_bits(
        std::experimental::simd_mask<T, Abi> const& mask)
    {
        return std::experimental::popcount(mask);
    }
}}}    // namespace hpx::parallel::traits

#endif
