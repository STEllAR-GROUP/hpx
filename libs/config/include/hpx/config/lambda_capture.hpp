//  Copyright (c) 2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_CONFIG_LAMBDA_CAPTURE_HPP
#define HPX_CONFIG_LAMBDA_CAPTURE_HPP

#include <hpx/config/defines.hpp>

#include <utility>

#if defined(DOXYGEN)
/// Evaluates to `var = std::forward<decltype(var)>(var)` if the compiler supports
/// C++14 lambda captures. Defaults to `var`.
#define HPX_CAPTURE_FORWARD(var)

/// Evaluates to `var = std::move(var)` if the compiler supports C++14
/// lambda captures. Defaults to `var`.
#define HPX_CAPTURE_MOVE(var)
#else
#if defined(HPX_HAVE_CXX14_LAMBDAS)
#define HPX_CAPTURE_FORWARD(var)  var = std::forward<decltype(var)>(var)
#define HPX_CAPTURE_MOVE(var)     var = std::move(var)
#else
#define HPX_CAPTURE_FORWARD(var)  var
#define HPX_CAPTURE_MOVE(var)     var
#endif
#endif

#endif
