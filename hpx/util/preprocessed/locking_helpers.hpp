//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PREPROCESSED_LOCKING_HELPERS_HPP)
#define HPX_PREPROCESSED_LOCKING_HELPERS_HPP

#if HPX_LOCK_LIMIT <= 10
#include <hpx/util/preprocessed/locking_helpers_10.hpp>
#elif HPX_LOCK_LIMIT <= 20
#include <hpx/util/preprocessed/locking_helpers_20.hpp>
#elif HPX_LOCK_LIMIT <= 30
#include <hpx/util/preprocessed/locking_helpers_30.hpp>
#elif HPX_LOCK_LIMIT <= 40
#include <hpx/util/preprocessed/locking_helpers_40.hpp>
#elif HPX_LOCK_LIMIT <= 50
#include <hpx/util/preprocessed/locking_helpers_50.hpp>
#else
#error "HPX_LOCK_LIMIT out of bounds for preprocessed headers"
#endif

#endif
