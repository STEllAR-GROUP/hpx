//  Copyright (c) 2019 Auriane Reverdell
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/config/defines.hpp>
#include <hpx/modules/concepts.hpp>

#if HPX_CONCEPTS_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message("The header hpx/traits/has_xxx.hpp is deprecated, \
    please include hpx/modules/concepts.hpp instead")
#else
#warning "The header hpx/traits/has_xxx.hpp is deprecated, \
    please include hpx/modules/concepts.hpp instead"
#endif
#endif
