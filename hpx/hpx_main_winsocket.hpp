//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_MAIN_WINSOCKET_MAY_08_2020_1050AM)
#define HPX_MAIN_WINSOCKET_MAY_08_2020_1050AM

#include <hpx/config.hpp>

#if defined(HPX_WINDOWS)

namespace hpx { namespace detail
{
    HPX_EXPORT void init_winsocket();
}}

#endif
#endif
