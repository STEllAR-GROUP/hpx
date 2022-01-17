//  Copyright (c) 2020 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CXX20_STD_ENDIAN)
#include <bit>
#endif

///////////////////////////////////////////////////////////////////////////////
/// \cond NODETAIL
namespace hpx {

#if defined(HPX_HAVE_CXX20_STD_ENDIAN)
    using std::endian;
#else
    enum class endian
    {
#ifdef _WIN32
        little = 0,
        big = 1,
        native = little
#else
        little = __ORDER_LITTLE_ENDIAN__,
        big = __ORDER_BIG_ENDIAN__,
        native = __BYTE_ORDER__
#endif
    };
#endif

}    // namespace hpx
