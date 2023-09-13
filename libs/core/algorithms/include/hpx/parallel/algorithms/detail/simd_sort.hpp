//  Copyright (c) 2023 Hari Hara Naveen S
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

// #include <avx512-16bit-qsort.hpp>
// #include <avx512-32bit-qsort.hpp>
#include <avx512-64bit-qsort.hpp>

#include <type_traits>


namespace hpx::parallel::util {
    // TODO : add support for _Float16
    // need compile time test as _Float16 is not always supported
    template <class T>
    struct is_simd_sortable
      : std::integral_constant<bool,
            std::is_same<uint16_t,
                typename std::remove_volatile<T>::type>::value ||
                std::is_same<int16_t,
                    typename std::remove_volatile<T>::type>::value ||
                std::is_same<uint32_t,
                    typename std::remove_volatile<T>::type>::value ||
                std::is_same<int32_t,
                    typename std::remove_volatile<T>::type>::value ||
                std::is_same<float,
                    typename std::remove_volatile<T>::type>::value ||
                std::is_same<uint64_t,
                    typename std::remove_volatile<T>::type>::value ||
                std::is_same<int64_t,
                    typename std::remove_volatile<T>::type>::value ||
                std::is_same<double,
                    typename std::remove_volatile<T>::type>::value>
    {
    };

    template <class T>
    constexpr bool is_simd_sortable_v = is_simd_sortable<T>::value;
}    // namespace hpx::parallel::util

#if (HPX_WITH_SIMD_SORT)

namespace hpx::parallel::util {
    template <typename T>
    void simd_quicksort(T* arr, int64_t arrsize)
    {
        static_assert(hpx::parallel::util::is_simd_sortable_v<T>);
        return avx512_qsort(arr, arrsize);
    }
}    // namespace hpx::parallel::util
#endif
