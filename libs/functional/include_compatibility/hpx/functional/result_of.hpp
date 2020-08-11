//  Copyright (c) 2020 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/config/defines.hpp>
#include <hpx/functional/invoke_result.hpp>

#if HPX_FUNCTIONAL_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message("The header hpx/functional/result_of.hpp is deprecated, \
    please include hpx/functional/invoke_result.hpp instead")
#else
#warning "The header hpx/functional/result_of.hpp is deprecated, \
    please include hpx/functional/invoke_result.hpp instead"
#endif
#endif
