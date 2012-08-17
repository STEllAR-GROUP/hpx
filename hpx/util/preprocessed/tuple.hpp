//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PREPROCESSED_UTIL_TUPLE_HPP)
#define HPX_PREPROCESSED_UTIL_TUPLE_HPP

#if HPX_TUPLE_LIMIT  <= 5
#include <hpx/util/preprocessed/tuple_5.hpp>
#elif HPX_TUPLE_LIMIT <= 10
#include <hpx/util/preprocessed/tuple_10.hpp>
#elif HPX_TUPLE_LIMIT <= 15
#include <hpx/util/preprocessed/tuple_15.hpp>
#elif HPX_TUPLE_LIMIT <= 20
#include <hpx/util/preprocessed/tuple_20.hpp>
/*
#elif HPX_TUPLE_LIMIT <= 25
#include <hpx/util/preprocessed/tuple_25.hpp>
#elif HPX_TUPLE_LIMIT <= 30
#include <hpx/util/preprocessed/tuple_30.hpp>
#elif HPX_TUPLE_LIMIT <= 35
#include <hpx/util/preprocessed/tuple_35.hpp>
#elif HPX_TUPLE_LIMIT <= 40
#include <hpx/util/preprocessed/tuple_40.hpp>
#elif HPX_TUPLE_LIMIT <= 45
#include <hpx/util/preprocessed/tuple_45.hpp>
#elif HPX_TUPLE_LIMIT <= 50
#include <hpx/util/preprocessed/tuple_50.hpp>
*/
#else
#error "HPX_TUPLE_LIMIT out of bounds for preprocessed headers"
#endif

#endif
