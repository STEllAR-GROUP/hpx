//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/algorithms/config/defines.hpp>
#include <hpx/algorithm.hpp>

// different versions of clang-format produce different results
// clang-format off
#if HPX_ALGORITHMS_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/traits/segmented_iterator_traits.hpp is deprecated, \
    please include hpx/algorithms/traits/segmented_iterator_traits.hpp")
#else
#warning                                                                       \
    "The header hpx/traits/segmented_iterator_traits.hpp is deprecated, \
    please include hpx/algorithms/traits/segmented_iterator_traits.hpp"
#endif
#endif
// clang-format on
