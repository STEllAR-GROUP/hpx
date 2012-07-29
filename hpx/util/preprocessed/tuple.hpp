//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PREPROCESSED_TUPLE_HPP)
#define HPX_PREPROCESSED_TUPLE_HPP

#if HPX_TUPLE_LIMIT <= 10
#include <hpx/util/preprocessed/tuple_10.hpp>
#elif HPX_TUPLE_LIMIT <= 20
#include <hpx/util/preprocessed/tuple_20.hpp>
#elif HPX_TUPLE_LIMIT <= 30
#include <hpx/util/preprocessed/tuple_30.hpp>
#elif HPX_TUPLE_LIMIT <= 40
#include <hpx/util/preprocessed/tuple_40.hpp>
#elif HPX_TUPLE_LIMIT <= 50
#include <hpx/util/preprocessed/tuple_50.hpp>
#else
#error "HPX_TUPLE_LIMIT out of bounds for preprocessed headers"
#endif

#endif
