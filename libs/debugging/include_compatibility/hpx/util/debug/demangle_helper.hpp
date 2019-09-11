//  Copyright (c) 2019 Auriane Reverdell
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/debugging/config/defines.hpp>
#include <hpx/debugging/demangle_helper.hpp>

#if defined(HPX_DEBUGGING_HAVE_DEPRECATION_WARNINGS)
#if defined(HPX_MSVC)
#pragma message( \
    "The header hpx/util/debug/demangle_helper.hpp is deprecated, \
    please include hpx/debugging/demangle_helper.hpp instead")
#else
#warning \
    "The header hpx/util/debug/demangle_helper.hpp is deprecated, \
    please include hpx/debugging/demangle_helper.hpp instead"
#endif
#endif
