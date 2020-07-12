//  Copyright (c) 2020 STE||AR Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/segmented_algorithms/config/defines.hpp>
#include <hpx/modules/segmented_algorithms.hpp>

#if HPX_SEGMENTED_ALGORITHMS_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message("The header hpx/util/zip_iterator.hpp is deprecated, please \
    include hpx/modules/segmented_algorithms.hpp instead")
#else
#warning "The header hpx/util/zip_iterator.hpp is deprecated, please \
    include hpx/modules/segmented_algorithms.hpp instead"
#endif
#endif
