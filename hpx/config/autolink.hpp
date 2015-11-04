//  Copyright (c) 2005-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#ifndef HPX_AUTOLINK_LIB_NAME
#error "Macro HPX_AUTOLINK_LIB_NAME not set (internal error)"
#endif

// enable auto-linking for supported platforms
#if defined(BOOST_MSVC) \
    || defined(__BORLANDC__) \
    || (defined(__MWERKS__) && defined(_WIN32) && (__MWERKS__ >= 0x3000)) \
    || (defined(__ICL) && defined(_MSC_EXTENSIONS) && (_MSC_VER >= 1200))

#if defined(HPX_DEBUG)
#pragma comment(lib, HPX_AUTOLINK_LIB_NAME "d" ".lib")
#else
#pragma comment(lib, HPX_AUTOLINK_LIB_NAME ".lib")
#endif

#endif

#undef HPX_AUTOLINK_LIB_NAME

