//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/program_options/config/defines.hpp>
#include <hpx/modules/program_options.hpp>

#if defined(HPX_PROGRAM_OPTIONS_HAVE_DEPRECATION_WARNINGS)
#if defined(HPX_MSVC)
#pragma message("The header hpx/program_options.hpp is deprecated, \
    please include hpx/modules/program_options.hpp instead")
#else
#warning "The header hpx/program_options.hpp is deprecated, \
    please include hpx/modules/program_options.hpp instead"
#endif
#endif
