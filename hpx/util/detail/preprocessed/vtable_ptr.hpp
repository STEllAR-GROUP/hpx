//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2012-2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PREPROCESSED_UTIL_DETAIL_VTABLE_PTR_HPP)
#define HPX_PREPROCESSED_UTIL_DETAIL_VTABLE_PTR_HPP

#if HPX_FUNCTION_ARGUMENT_LIMIT  <= 5
#include <hpx/util/detail/preprocessed/vtable_ptr_5.hpp>
#elif HPX_FUNCTION_ARGUMENT_LIMIT <= 10
#include <hpx/util/detail/preprocessed/vtable_ptr_10.hpp>
#elif HPX_FUNCTION_ARGUMENT_LIMIT <= 15
#include <hpx/util/detail/preprocessed/vtable_ptr_15.hpp>
#elif HPX_FUNCTION_ARGUMENT_LIMIT <= 20
#include <hpx/util/detail/preprocessed/vtable_ptr_20.hpp>
/*
#elif HPX_FUNCTION_ARGUMENT_LIMIT <= 25
#include <hpx/util/detail/preprocessed/vtable_ptr_25.hpp>
#elif HPX_FUNCTION_ARGUMENT_LIMIT <= 30
#include <hpx/util/detail/preprocessed/vtable_ptr_30.hpp>
#elif HPX_FUNCTION_ARGUMENT_LIMIT <= 35
#include <hpx/util/detail/preprocessed/vtable_ptr_35.hpp>
#elif HPX_FUNCTION_ARGUMENT_LIMIT <= 40
#include <hpx/util/detail/preprocessed/vtable_ptr_40.hpp>
#elif HPX_FUNCTION_ARGUMENT_LIMIT <= 45
#include <hpx/util/detail/preprocessed/vtable_ptr_45.hpp>
#elif HPX_FUNCTION_ARGUMENT_LIMIT <= 50
#include <hpx/util/detail/preprocessed/vtable_ptr_50.hpp>
*/
#else
#error "HPX_FUNCTION_ARGUMENT_LIMIT out of bounds for preprocessed headers"
#endif

#endif
