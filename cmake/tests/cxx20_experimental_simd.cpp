//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// test for availability of std experimental simd (C++20)

#include <experimental/simd>

int main()
{
    auto v1 = std::experimental::simd<int>(12);
    auto v2 = std::experimental::native_simd<float>(24.0f);

    auto v3 = std::experimental::simd<int, std::experimental::simd_abi::scalar>(42);
    auto v4 = std::experimental::fixed_size_simd<float, 32>(3.14f);

    auto m1 = std::experimental::simd_mask<int>(true);
    auto m2 = std::experimental::native_simd_mask<float>(false);

    auto x1 = std::experimental::memory_alignment_v<
            std::experimental::native_simd<float>>;
    auto x2 = std::experimental::memory_alignment_v<
            std::experimental::simd<float, std::experimental::simd_abi::scalar>>;

    int arr[32] = {5};
    v1.copy_from(arr, std::experimental::element_aligned);
    v3.copy_from(arr, std::experimental::element_aligned);
    v1.copy_to(arr, std::experimental::element_aligned);
    v3.copy_to(arr, std::experimental::element_aligned);

    alignas(std::experimental::memory_alignment_v<
            std::experimental::native_simd<float>>) float arr2[32] = {42.f};
    v2.copy_from(arr2, std::experimental::vector_aligned);
    v2.copy_to(arr2, std::experimental::vector_aligned);

    (void) (x1);
    (void) (x2);
    (void) (v4);
    (void) (m1);
    (void) (m2);
    return 0;
}
