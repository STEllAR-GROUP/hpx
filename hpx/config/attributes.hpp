//  Copyright (c) 2017 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_CONFIG_ATTRIBUTES_HPP
#define HPX_CONFIG_ATTRIBUTES_HPP

#include <hpx/config/defines.hpp>

#if defined(HPX_HAVE_CXX17_FALLTHROUGH_ATTRIBUTE)
#   define HPX_FALLTHROUGH [[fallthrough]];
#else
#   define HPX_FALLTHROUGH
#endif

#endif
