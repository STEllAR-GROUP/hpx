//  Copyright (c) 2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_CONFIG_LAMBDA_CAPTURE_HPP
#define HPX_CONFIG_LAMBDA_CAPTURE_HPP

#include <hpx/config/defines.hpp>

#include <utility>

#if defined(HPX_HAVE_CXX14_LAMBDAS)
#define HPX_CAPTURE_FORWARD(var, type)  var = std::forward<type>(var)
#define HPX_CAPTURE_MOVE(var)           var = std::move(var)
#else
#define HPX_CAPTURE_FORWARD(var, type)  var
#define HPX_CAPTURE_MOVE(var)           var
#endif

#endif
