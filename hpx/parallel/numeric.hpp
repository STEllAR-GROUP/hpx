//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_NUMERIC_JUN_02_2014_1151AM)
#define HPX_PARALLEL_NUMERIC_JUN_02_2014_1151AM

#include <hpx/config.hpp>

/// See N4310: 1.3/3
#include <numeric>

#include <hpx/parallel/algorithms/adjacent_difference.hpp>
#include <hpx/parallel/algorithms/exclusive_scan.hpp>
#include <hpx/parallel/algorithms/inclusive_scan.hpp>
#include <hpx/parallel/algorithms/reduce.hpp>
#include <hpx/parallel/algorithms/transform_exclusive_scan.hpp>
#include <hpx/parallel/algorithms/transform_inclusive_scan.hpp>
#include <hpx/parallel/algorithms/transform_reduce.hpp>
#include <hpx/parallel/algorithms/transform_reduce_binary.hpp>

#if defined(HPX_HAVE_TRANSFORM_REDUCE_COMPATIBILITY)
#include <hpx/parallel/algorithms/inner_product.hpp>
#endif

#endif
