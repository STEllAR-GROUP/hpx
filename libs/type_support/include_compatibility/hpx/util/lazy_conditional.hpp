//  Copyright (c) 2019 Auriane Reverdell
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/type_support/config/defines.hpp>
#include <hpx/type_support/lazy_conditional.hpp>

#if defined(HPX_TYPE_SUPPORT_HAVE_DEPRECATION_WARNINGS)
#if defined(HPX_MSVC)
#pragma message( \
    "The header hpx/util/lazy_conditional.hpp is deprecated, \
    please include hpx/type_support/lazy_conditional.hpp instead")
#else
#warning \
    "The header hpx/util/lazy_conditional.hpp is deprecated, \
    please include hpx/type_support/lazy_conditional.hpp instead"
#endif
#endif
