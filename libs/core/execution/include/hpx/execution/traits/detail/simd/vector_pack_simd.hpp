//  Copyright (c) 2022 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_EXPERIMENTAL_SIMD)

#include <cstddef>

#if defined(HPX_HAVE_DATAPAR_STD_EXPERIMENTAL_SIMD)
#include <experimental/simd>

namespace hpx::datapar::experimental {

    using std::experimental::fixed_size_simd;
    using std::experimental::is_simd_v;
    using std::experimental::native_simd;
    using std::experimental::simd;
    using std::experimental::simd_mask;

    using std::experimental::simd_abi::native;

    using std::experimental::memory_alignment_v;
    using std::experimental::vector_aligned;

    using std::experimental::all_of;
    using std::experimental::any_of;
    using std::experimental::find_first_set;
    using std::experimental::none_of;
    using std::experimental::popcount;
    using std::experimental::reduce;

    template <typename T, typename Abi>
    HPX_HOST_DEVICE HPX_FORCEINLINE auto choose(
        std::experimental::simd_mask<T, Abi> const& msk,
        std::experimental::simd<T, Abi> const& v_true,
        std::experimental::simd<T, Abi> const& v_false) noexcept
    {
        std::experimental::simd<T, Abi> v;
        where(msk, v) = v_true;
        where(!msk, v) = v_false;
        return v;
    }

    template <typename T, typename Abi>
    HPX_HOST_DEVICE HPX_FORCEINLINE void mask_assign(
        std::experimental::simd_mask<T, Abi> const& msk,
        std::experimental::simd<T, Abi>& v,
        std::experimental::simd<T, Abi> const& val) noexcept
    {
        where(msk, v) = val;
    }

    template <typename Vector, typename T>
    HPX_HOST_DEVICE HPX_FORCEINLINE auto set(
        Vector& vec, std::size_t index, T val) noexcept
    {
        vec[index] = val;
    }
}    // namespace hpx::datapar::experimental
#endif

#if defined(HPX_HAVE_DATAPAR_SVE)
#include <sve/sve.hpp>

namespace hpx::datapar::experimental {

    using namespace sve::experimental;
    using simd_abi::native;

    template <typename Vector, typename T>
    HPX_HOST_DEVICE HPX_FORCEINLINE auto set(
        Vector& vec, std::size_t index, T val) noexcept
    {
        vec.set(index, val);
    }
}    // namespace hpx::datapar::experimental

#endif

#endif
