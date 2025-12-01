//  Copyright (c)      2021 ETH Zurich
//  Copyright (c)      2018 Mikael Simberg
//  Copyright (c) 2007-2024 Hartmut Kaiser
//  Copyright (c) 2010-2011 Phillip LeBlanc, Dylan Stark
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/preprocessor.hpp>

#if !defined(HPX_PREFIX)
#define HPX_PREFIX ""
#endif

#if defined(HPX_APPLICATION_NAME_DEFAULT) && !defined(HPX_APPLICATION_NAME)
#define HPX_APPLICATION_NAME HPX_APPLICATION_NAME_DEFAULT
#endif

#if !defined(HPX_APPLICATION_STRING)
#if defined(HPX_APPLICATION_NAME)
#define HPX_APPLICATION_STRING HPX_PP_STRINGIZE(HPX_APPLICATION_NAME)
#else
#define HPX_APPLICATION_STRING "unknown HPX application"
#endif
#endif
