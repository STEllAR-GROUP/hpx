//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/defines.hpp>
#include <hpx/preprocessor/identity.hpp>

#if defined(HPX_HAVE_CXX20_LAMBDA_CAPTURE)
#define HPX_CXX20_CAPTURE_THIS(...) HPX_PP_IDENTITY(__VA_ARGS__, this)
#else
#define HPX_CXX20_CAPTURE_THIS(...) __VA_ARGS__
#endif
