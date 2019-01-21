//  Copyright (c) 2013 Antoine Tran Tan
//  Copyright (c) 2001, 2002 Peter Dimov and Multi Media Ltd.
//  Copyright (c) 2007 Peter Dimov
//  Copyright (c) Beman Dawes 2011
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Make HPX inspect tool happy: hpxinspect:noassert_macro
//                               hpxinspect:noinclude:HPX_ASSERT
//                               hpxinspect:nodeprecatedname:BOOST_ASSERT

//  Note: There are no include guards. This is intentional.

#include <hpx/assert.hpp>

#include <hpx/config.hpp>

#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/util/assert.hpp is deprecated. Please use hpx/assert.hpp instead")
#else
#warning                                                                       \
    "The header hpx/util/assert.hpp is deprecated. Please use hpx/assert.hpp instead"
#endif
