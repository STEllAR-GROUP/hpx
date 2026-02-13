//  Copyright (c) 2007-2025 Hartmut Kaiser
//  Copyright (c) 2019 Austin McCartney
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

namespace hpx::traits {

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_CORE_EXPORT template <typename Iterator, typename Enable = void>
    struct is_segmented_iterator;

    HPX_CXX_CORE_EXPORT template <typename Iter>
    using is_segmented_iterator_t = typename is_segmented_iterator<Iter>::type;

    HPX_CXX_CORE_EXPORT template <typename Iter>
    inline constexpr bool is_segmented_iterator_v =
        is_segmented_iterator<Iter>::value;

    HPX_CXX_CORE_EXPORT template <typename Iterator, typename Enable = void>
    struct is_segmented_local_iterator;

    HPX_CXX_CORE_EXPORT template <typename Iter>
    using is_segmented_local_iterator_t =
        typename is_segmented_local_iterator<Iter>::type;

    HPX_CXX_CORE_EXPORT template <typename Iter>
    inline constexpr bool is_segmented_local_iterator_v =
        is_segmented_local_iterator<Iter>::value;
}    // namespace hpx::traits
