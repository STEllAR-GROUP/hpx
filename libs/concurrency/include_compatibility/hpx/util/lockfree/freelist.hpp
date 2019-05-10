//  Copyright (c) 2019 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/concurrency/config/defines.hpp>
#include <hpx/concurrency/detail/freelist.hpp>

#if defined(HPX_CONCURRENCY_HAVE_DEPRECATION_WARNINGS)
#if defined(HPX_MSVC)
#pragma message( \
    "The header hpx/util/lockfree/freelist.hpp is deprecated, \
    please include hpx/concurrency/detail/freelist.hpp instead")
#else
#warning \
    "The header hpx/util/lockfree/freelist.hpp is deprecated, \
    please include hpx/concurrency/detail/freelist.hpp instead"
#endif
#endif
