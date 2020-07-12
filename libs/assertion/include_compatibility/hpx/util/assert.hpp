//  Copyright (c) 2013 Antoine Tran Tan
//  Copyright (c) 2001, 2002 Peter Dimov and Multi Media Ltd.
//  Copyright (c) 2007 Peter Dimov
//  Copyright (c) Beman Dawes 2011
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

//  Make HPX inspect tool happy: hpxinspect:noassert_macro
//                               hpxinspect:noinclude:HPX_ASSERT

#include <hpx/config.hpp>
#include <hpx/assertion/config/defines.hpp>
#include <hpx/assert.hpp>

#if HPX_ASSERTION_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/util/assert.hpp is deprecated,                             \
    please include hpx/assert.hpp instead")
#else
#warning                                                                       \
    "The header hpx/util/assert.hpp is deprecated,                        \
    please include hpx/assert.hpp instead"
#endif
#endif
