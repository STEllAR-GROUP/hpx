//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_PREPROCESSED_FUTURE_WAIT_HPP)
#define HPX_LCOS_PREPROCESSED_FUTURE_WAIT_HPP

#if HPX_WAIT_ARGUMENT_LIMIT <= 5
#include <hpx/lcos/preprocessed/future_wait_5.hpp>
#elif HPX_WAIT_ARGUMENT_LIMIT <= 10
#include <hpx/lcos/preprocessed/future_wait_10.hpp>
#elif HPX_WAIT_ARGUMENT_LIMIT <= 20
#include <hpx/lcos/preprocessed/future_wait_20.hpp>
#elif HPX_WAIT_ARGUMENT_LIMIT <= 30
#include <hpx/lcos/preprocessed/future_wait_30.hpp>
#elif HPX_WAIT_ARGUMENT_LIMIT <= 40
#include <hpx/lcos/preprocessed/future_wait_40.hpp>
#elif HPX_WAIT_ARGUMENT_LIMIT <= 50
#include <hpx/lcos/preprocessed/future_wait_50.hpp>
#else
#error "HPX_WAIT_ARGUMENT_LIMIT out of bounds for preprocessed headers"
#endif

#endif
