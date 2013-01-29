//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2012-2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PREPROCESSED_RUNTIME_COMPONENTS_SERVER_RUNTIME_SUPPORT_HPP)
#define HPX_PREPROCESSED_RUNTIME_COMPONENTS_SERVER_RUNTIME_SUPPORT_HPP

#if HPX_ACTION_ARGUMENT_LIMIT  <= 5
#include <hpx/runtime/components/server/preprocessed/runtime_support_5.hpp>
#elif HPX_ACTION_ARGUMENT_LIMIT <= 10
#include <hpx/runtime/components/server/preprocessed/runtime_support_10.hpp>
#elif HPX_ACTION_ARGUMENT_LIMIT <= 15
#include <hpx/runtime/components/server/preprocessed/runtime_support_15.hpp>
#elif HPX_ACTION_ARGUMENT_LIMIT <= 20
#include <hpx/runtime/components/server/preprocessed/runtime_support_20.hpp>
/*
#elif HPX_ACTION_ARGUMENT_LIMIT <= 25
#include <hpx/runtime/components/server/preprocessed/runtime_support_25.hpp>
#elif HPX_ACTION_ARGUMENT_LIMIT <= 30
#include <hpx/runtime/components/server/preprocessed/runtime_support_30.hpp>
#elif HPX_ACTION_ARGUMENT_LIMIT <= 35
#include <hpx/runtime/components/server/preprocessed/runtime_support_35.hpp>
#elif HPX_ACTION_ARGUMENT_LIMIT <= 40
#include <hpx/runtime/components/server/preprocessed/runtime_support_40.hpp>
#elif HPX_ACTION_ARGUMENT_LIMIT <= 45
#include <hpx/runtime/components/server/preprocessed/runtime_support_45.hpp>
#elif HPX_ACTION_ARGUMENT_LIMIT <= 50
#include <hpx/runtime/components/server/preprocessed/runtime_support_50.hpp>
*/
#else
#error "HPX_ACTION_ARGUMENT_LIMIT out of bounds for preprocessed headers"
#endif

#endif
