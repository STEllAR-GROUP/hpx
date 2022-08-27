//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if defined(HPX_HAVE_CXX_LAMBDA_CAPTURE_DECLTYPE)
#define HPX_FORWARD(T, ...) static_cast<decltype(__VA_ARGS__)&&>(__VA_ARGS__)
#else
#define HPX_FORWARD(T, ...) static_cast<T&&>(__VA_ARGS__)
#endif
